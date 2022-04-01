# -*- coding: utf-8 -*-
from keras import layers, models, optimizers
from keras import backend as K
from keras.layers import GlobalAveragePooling2D, Reshape, Dense, multiply, Permute
from capsulelayers import CapsuleLayer, Length, ConvCapsuleLayer3D, FlattenCaps, PrimaryCap, Mish


K.set_image_data_format('channels_last')


def squeeze_excite_block(input, ratio=8):
    ''' Create a squeeze-excite block
    Args:
        input: input tensor
        filters: number of output filters
        k: width factor

    Returns: a keras tensor
    '''
    init = input
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    filters = init._keras_shape[channel_axis]

    # 空间采样
    se_shape = (1, 1, filters)

    # Squeeze
    se = GlobalAveragePooling2D()(init)
    se = Reshape(se_shape)(se)

    # Excitation
    se = Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
    se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)

    if K.image_data_format() == 'channels_first':
        se = Permute((3, 1, 2))(se)

    x = multiply([init, se])

    return x

def large_spatial(input):
    conv1 = layers.Conv2D(filters=128, kernel_size=5, strides=1, padding='valid',
                          name='spatial_conv1')(input)
    conv1 = layers.BatchNormalization(momentum=0.9, name='spatial_bn1')(conv1)
    conv1 = Mish()(conv1)

    conv2 = squeeze_excite_block(conv1)
    conv2 = layers.Conv2D(filters=64, kernel_size=5, strides=1, padding='valid',
                          name='spatial_conv2')(conv2)
    conv2 = layers.BatchNormalization(momentum=0.9, name='spatial_bn2')(conv2)
    conv2 = Mish()(conv2)

    conv2 = squeeze_excite_block(conv2)
    primarycaps = PrimaryCap(conv2, dim_vector=4, n_channels=8, kernel_size=9, strides=2, padding='valid')

    l_skip = primarycaps
    l = ConvCapsuleLayer3D(kernel_size=3, num_capsule=8, num_atoms=4, strides=1, padding='same', routings=3)(l_skip)
    l = ConvCapsuleLayer3D(kernel_size=3, num_capsule=8, num_atoms=4, strides=1, padding='same', routings=3)(l)
    l = layers.Add()([l, l_skip])

    l_skip = l
    la = ConvCapsuleLayer3D(kernel_size=3, num_capsule=8, num_atoms=8, strides=2, padding='same', routings=3)(l_skip)
    l = ConvCapsuleLayer3D(kernel_size=3, num_capsule=8, num_atoms=8, strides=2, padding='same', routings=3)(l_skip)
    l = ConvCapsuleLayer3D(kernel_size=3, num_capsule=8, num_atoms=8, strides=1, padding='same', routings=3)(l)
    l = layers.Add()([l, la])

    l = FlattenCaps()(l)

    return l

def spectral_local_spatial(input):
    conv1 = layers.Conv2D(filters=128, kernel_size=(1, 1), strides=(1, 1), padding='same',
                          name='spectral_conv1')(input)
    conv1 = layers.BatchNormalization(momentum=0.9, name='spectral_bn1')(conv1)
    conv1 = Mish()(conv1)

    conv2 = squeeze_excite_block(conv1)
    conv2 = layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same',
                          name='spectral_conv2')(conv2)
    conv2 = layers.BatchNormalization(momentum=0.9, name='spectral_bn2')(conv2)
    conv2 = Mish()(conv2)

    conv2 = squeeze_excite_block(conv2)
    primarycaps = PrimaryCap(conv2, dim_vector=4, n_channels=8, kernel_size=3, strides=2, padding='same')

    l_skip = primarycaps
    l = ConvCapsuleLayer3D(kernel_size=3, num_capsule=8, num_atoms=4, strides=1, padding='same', routings=3)(primarycaps)
    l = ConvCapsuleLayer3D(kernel_size=3, num_capsule=8, num_atoms=4, strides=1, padding='same', routings=3)(l)
    l = layers.Add()([l, l_skip])

    l_skip = l
    la = ConvCapsuleLayer3D(kernel_size=3, num_capsule=8, num_atoms=8, strides=2, padding='same', routings=3)(l_skip)
    l = ConvCapsuleLayer3D(kernel_size=3, num_capsule=8, num_atoms=8, strides=2, padding='same', routings=3)(l_skip)
    l = ConvCapsuleLayer3D(kernel_size=3, num_capsule=8, num_atoms=8, strides=1, padding='same', routings=3)(l)
    l = layers.Add()([l, la])

    l = FlattenCaps()(l)

    return l

class CapsnetBuilder_2D_noDecoder(object):
    @staticmethod
    def build(input_shape_spectral, input_shape_spatial, n_class, routings):
        x1 = layers.Input(shape=input_shape_spectral)
        x2 = layers.Input(shape=input_shape_spatial)

        la = spectral_local_spatial(x1)
        lb = large_spatial(x2)

        l = layers.Concatenate(axis=-2)([la, lb])

        digitcaps = CapsuleLayer(num_capsule=n_class, dim_capsule=16, routings=routings, name='digitcaps')(l)

        # Layer 5: classifier
        out_caps = Length(name='capsnet')(digitcaps)

        # Models for training
        train_model = models.Model([x1, x2], out_caps)
        return train_model

    @staticmethod
    def build_capsnet(input_shape_spectral, input_shape_spatial, n_class, routings):
        return CapsnetBuilder_2D_noDecoder.build(input_shape_spectral, input_shape_spatial, n_class, routings)

def margin_loss(y_true, y_pred):
    """
    Margin loss for Eq.(4). When y_true[i, :] contains not just one `1`, this loss should work too. Not test it.
    :param y_true: [None, n_classes]
    :param y_pred: [None, num_capsule]
    :return: a scalar loss value.
    """
    L = y_true * K.square(K.maximum(0., 0.9 - y_pred)) + \
        0.5 * (1 - y_true) * K.square(K.maximum(0., y_pred - 0.1))

    return K.mean(K.sum(L, 1))

def main():
    model = CapsnetBuilder_2D_noDecoder.build_capsnet(input_shape_spectral=(7, 7, 200), input_shape_spatial=(27, 27, 7), n_class=16, routings=3)
    model.compile(optimizer=optimizers.Adam(lr=0.001),
                  loss=[margin_loss],
                  metrics={'capsnet': 'accuracy'})
    model.summary(positions=[.33, .61, .75, 1.])

if __name__ == '__main__':
    main()