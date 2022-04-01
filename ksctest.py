# -*- coding: utf-8 -*-
# 整个模型训练测试验证代码，并保存最优模型，打印测试数据
import numpy as np
import scipy.io as sio
from keras.utils.np_utils import to_categorical
import time
from sklearn import preprocessing
from sklearn.decomposition import PCA
import MSCapsNet
import os
import argparse
from dividedataset import indexToAssignment, selectNeighboringPatch, sampling
import collections
from sklearn import metrics
import modelStatsRecord, averageAccuracy, zeroPadding


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# setting the hyper parameters
parser = argparse.ArgumentParser(description="Capsule Network on KSC.")
parser.add_argument('--epochs', default=200, type=int)
parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--nb_classes', default=13, type=int)
parser.add_argument('--lr', default=0.001, type=float,
                    help="Initial learning rate")
parser.add_argument('-r', '--routings', default=3, type=int,
                    help="Number of iterations used in routing algorithm. should > 0")
parser.add_argument('--input_dimension', default=176, type=int,
                    help="Number of dimensions for input datasets.")
parser.add_argument('--n_components', default=7, type=int,
                    help="Number of principal components for input datasets.")
parser.add_argument('--save_dir', default='./result/KSC-7-27-3%-SE-pca7')
args = parser.parse_args()
print(args)

ITER = 3
seeds = [1220, 1221, 1222]

PATCH_LENGTH_SPECTRAL = 3
PATCH_LENGTH_SPATIAL = 13
img_rows = img_cols = 7 and 27

TOTAL_SIZE = 5211
VAL_SIZE = 162
TRAIN_SIZE = 162
TEST_SIZE = TOTAL_SIZE - TRAIN_SIZE
VALIDATION_SPLIT = 0.97  # 3% for trainnig and 97% for validation and testing
# 0.995 34
# 0.99 61
# 0.97 162
# 0.95 268
# 0.93 373
# 0.9  528

# 加载数据
mat_data = sio.loadmat('./datasets/ksc/KSC.mat')
data_IN = mat_data['KSC']
# 标签数据
mat_gt = sio.loadmat('./datasets/ksc/KSC_gt.mat')
gt_IN = mat_gt['KSC_gt']

new_gt_IN = gt_IN

# 对数据进行reshape处理之后，进行scale操作
data = data_IN.reshape(np.prod(data_IN.shape[:2]), np.prod(data_IN.shape[2:]))
gt = new_gt_IN.reshape(np.prod(new_gt_IN.shape[:2]), )

# 标准化操作，即将所有数据沿行沿列均归一化道0-1之间
data = preprocessing.scale(data)

pca = PCA(n_components=args.n_components)
data_spatial = pca.fit_transform(data)
print(data.shape)

whole_data_spectral = data.reshape(data_IN.shape[0], data_IN.shape[1], data_IN.shape[2])
whole_data_spatial = data_spatial.reshape(data_IN.shape[0], data_IN.shape[1], args.n_components)

padded_data_spectral = zeroPadding.zeroPadding_3D(whole_data_spectral, PATCH_LENGTH_SPECTRAL)
padded_data_spatial = zeroPadding.zeroPadding_3D(whole_data_spatial, PATCH_LENGTH_SPATIAL)

train_data_spectral = np.zeros((TRAIN_SIZE, 2 * PATCH_LENGTH_SPECTRAL + 1, 2 * PATCH_LENGTH_SPECTRAL + 1, args.input_dimension))
train_data_spatial = np.zeros((TRAIN_SIZE, 2 * PATCH_LENGTH_SPATIAL + 1, 2 * PATCH_LENGTH_SPATIAL + 1, args.n_components))

test_data_spectral = np.zeros((TEST_SIZE, 2 * PATCH_LENGTH_SPECTRAL + 1, 2 * PATCH_LENGTH_SPECTRAL + 1, args.input_dimension))
test_data_spatial = np.zeros((TEST_SIZE, 2 * PATCH_LENGTH_SPATIAL + 1, 2 * PATCH_LENGTH_SPATIAL + 1, args.n_components))

KAPPA = []
OA = []
AA = []
TRAINING_TIME = []
TESTING_TIME = []
ELEMENT_ACC = np.zeros((ITER, args.nb_classes))


for index_iter in range(ITER):
    print("# %d Iteration" % (index_iter + 1))
    # # 1 Iteration

    # save the best validated model
    best_weights_path = args.save_dir + '/KSC_best_weights_' + str(index_iter + 1) + '.hdf5'

    # 通过sampling函数拿到测试和训练样本
    np.random.seed(seeds[index_iter])
    train_indices, test_indices = sampling(VALIDATION_SPLIT, gt)

    y_train = gt[train_indices] - 1
    y_train = to_categorical(np.asarray(y_train))

    y_test = gt[test_indices] - 1
    y_test = to_categorical(np.asarray(y_test))

    # 重构spectral部分训练数据
    train_assign_spectral = indexToAssignment(train_indices, whole_data_spectral.shape[0], whole_data_spectral.shape[1], PATCH_LENGTH_SPECTRAL)
    for i in range(len(train_assign_spectral)):
        train_data_spectral[i] = selectNeighboringPatch(padded_data_spectral, train_assign_spectral[i][0], train_assign_spectral[i][1], PATCH_LENGTH_SPECTRAL)
    # 重构spatial部分训练数据
    train_assign_spatial = indexToAssignment(train_indices, whole_data_spatial.shape[0], whole_data_spatial.shape[1], PATCH_LENGTH_SPATIAL)
    for i in range(len(train_assign_spatial)):
        train_data_spatial[i] = selectNeighboringPatch(padded_data_spatial, train_assign_spatial[i][0], train_assign_spatial[i][1], PATCH_LENGTH_SPATIAL)

    # 重构spectral部分测试数据
    test_assign_spectral = indexToAssignment(test_indices, whole_data_spectral.shape[0], whole_data_spectral.shape[1], PATCH_LENGTH_SPECTRAL)
    for i in range(len(test_assign_spectral)):
        test_data_spectral[i] = selectNeighboringPatch(padded_data_spectral, test_assign_spectral[i][0], test_assign_spectral[i][1], PATCH_LENGTH_SPECTRAL)
    # 重构spatial部分测试数据
    test_assign_spatial = indexToAssignment(test_indices, whole_data_spatial.shape[0], whole_data_spatial.shape[1], PATCH_LENGTH_SPATIAL)
    for i in range(len(test_assign_spatial)):
        test_data_spatial[i] = selectNeighboringPatch(padded_data_spatial, test_assign_spatial[i][0], test_assign_spatial[i][1], PATCH_LENGTH_SPATIAL)

    x_train_spectral = train_data_spectral.reshape(train_data_spectral.shape[0], train_data_spectral.shape[1], train_data_spectral.shape[2], args.input_dimension)
    x_train_spatial = train_data_spatial.reshape(train_data_spatial.shape[0], train_data_spatial.shape[1], train_data_spatial.shape[2], args.n_components)

    x_test_spectral_all = test_data_spectral.reshape(test_data_spectral.shape[0], test_data_spectral.shape[1], test_data_spectral.shape[2], args.input_dimension)
    x_test_spatial_all = test_data_spatial.reshape(test_data_spatial.shape[0], test_data_spatial.shape[1],  test_data_spatial.shape[2], args.n_components)

    # 在测试数据集上进行验证和测试的划分
    x_val_spectral = x_test_spectral_all[-VAL_SIZE:]
    x_val_spatial = x_test_spatial_all[-VAL_SIZE:]
    y_val = y_test[-VAL_SIZE:]

    x_test_spectral = x_test_spectral_all[:-VAL_SIZE]
    x_test_spatial = x_test_spatial_all[:-VAL_SIZE]
    y_test = y_test[:-VAL_SIZE]

    gt_test = gt[test_indices] - 1
    gt_test = gt_test[:-VAL_SIZE]

    eval_model = MSCapsNet.CapsnetBuilder_2D_noDecoder.build_capsnet(
        input_shape_spectral=(x_test_spectral.shape[1], x_test_spectral.shape[2], x_test_spectral.shape[3]),
        input_shape_spatial=(x_test_spatial.shape[1], x_test_spatial.shape[2], x_test_spatial.shape[3]),
        n_class=len(np.unique(np.argmax(y_test, 1))),
        routings=args.routings)
    eval_model.load_weights(best_weights_path)

    # 测试
    tic2 = time.clock()
    y_pred = eval_model.predict([x_test_spectral, x_test_spatial])
    toc2 = time.clock()
    pred_test = y_pred.argmax(axis=1)
    collections.Counter(pred_test)

    overall_acc = metrics.accuracy_score(pred_test, gt_test)
    confusion_matrix = metrics.confusion_matrix(pred_test, gt_test)
    each_acc, average_acc = averageAccuracy.AA_andEachClassAccuracy(confusion_matrix)
    kappa = metrics.cohen_kappa_score(pred_test, gt_test)
    KAPPA.append(kappa)
    OA.append(overall_acc)
    AA.append(average_acc)
    TESTING_TIME.append(toc2 - tic2)
    ELEMENT_ACC[index_iter, :] = each_acc

    print('Test time:', toc2 - tic2)
    print('Test accuracy:', overall_acc)

    print("Finished.")
    print("# %d Iteration" % (index_iter + 1))

# 自定义输出类
modelStatsRecord.outputStats_assess_test(KAPPA, OA, AA, ELEMENT_ACC, TESTING_TIME, args.nb_classes,
                             args.save_dir + '/KSC_test.txt',
                             args.save_dir + '/KSC_test_element.txt')
