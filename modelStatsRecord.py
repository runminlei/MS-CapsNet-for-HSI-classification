# -*- coding: utf-8 -*-
import numpy as np


def outputStats_assess_test(KAPPA_AE, OA_AE, AA_AE, ELEMENT_ACC_AE, TESTING_TIME_AE, CATEGORY, path1, path2):


    f = open(path1, 'a')

    sentence0 = 'KAPPAs, mean_KAPPA ± std_KAPPA for each iteration are:' + str(KAPPA_AE) + str(np.mean(KAPPA_AE)) + ' ± ' + str(np.std(KAPPA_AE)) + '\n'
    f.write(sentence0)
    sentence1 = 'OAs, mean_OA ± std_OA for each iteration are:' + str(OA_AE) + str(np.mean(OA_AE)) + ' ± ' + str(np.std(OA_AE)) + '\n'
    f.write(sentence1)
    sentence2 = 'AAs, mean_AA ± std_AA for each iteration are:' + str(AA_AE) + str(np.mean(AA_AE)) + ' ± ' + str(np.std(AA_AE)) + '\n'
    f.write(sentence2)
    sentence4 = 'Total average Testing time is:' + str(TESTING_TIME_AE) + str(np.mean(TESTING_TIME_AE)) + '\n'
    f.write(sentence4)

    element_mean = np.mean(ELEMENT_ACC_AE, axis=0)
    element_std = np.std(ELEMENT_ACC_AE, axis=0)
    sentence5 = "Mean of all elements in confusion matrix:" + str(np.mean(ELEMENT_ACC_AE, axis=0)) + '\n'
    f.write(sentence5)
    sentence6 = "Standard deviation of all elements in confusion matrix" + str(np.std(ELEMENT_ACC_AE, axis=0)) + '\n'
    f.write(sentence6)

    f.close()

    print_matrix = np.zeros((CATEGORY), dtype=object)
    for i in range(CATEGORY):
        print_matrix[i] = str(element_mean[i]) + " ± " + str(element_std[i])

    np.savetxt(path2, print_matrix.astype(str), fmt='%s', delimiter="\t",
               newline='\n')
