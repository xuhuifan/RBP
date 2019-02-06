import numpy as np
import scipy
import copy
import time
from scipy.stats import poisson
from scipy.stats import norm




def patchPara_ini(patchNum, dimLength, lambdas):
    # Usage: initialize the position parameters of the bounding boxes
    # Input:
    # patchNum: number of bounding boxes
    # dimLength: length for each dimension
    # lambdas: parameter controls of the size of bounding boxes

    # Output:
    # patchParas: positions of the bounding boxes, size: K x D x 2

    patchParas = np.zeros([patchNum, len(dimLength), 2])

    case_1 = 2
    if case_1 == 1:
        patchParas[:, :, 0] = np.zeros(len(dimLength))
        patchParas[:, :, 1] = dimLength
    elif case_1 == 2:
        patchParas[0, :, 0] = np.zeros(len(dimLength))
        patchParas[0, :, 1] = dimLength

        for jj in np.arange(1, patchNum):
            patchParas[jj] = patch_shape_gen(dimLength, lambdas)

    return patchParas


def patch_shape_gen(dimLength, lambdas):
    # Usage: generate the positions of one bounding box
    # Input:
    # dimLength: length for each dimension
    # lambdas: parameter controls of the size of bounding boxes

    # Output:
    # patch_shape: positions of one bounding box, size: D x 2

    patch_shape = np.zeros([len(dimLength), 2])

    s_not_equal_0 = (np.random.uniform(size=len(dimLength))>(1/(1+lambdas*dimLength)))
    patch_shape[s_not_equal_0, 0] = np.random.uniform(low=0, high=dimLength[s_not_equal_0])

    patch_shape[:, 1] = np.random.exponential(lambdas, size=len(dimLength))
    violates_index = (np.sum(patch_shape, axis=1)>dimLength)
    patch_shape[violates_index, 1] = (dimLength-patch_shape[:, 0])[violates_index]

    return patch_shape



def LL_cal(dataR_val, omega_val, pos_and_negative):
    # Usage: calculate the log-likelihood for dataR_val
    # Input:
    # dataR_val: observational data
    # omega_val: intensity values for the observational data
    if pos_and_negative==1:
        return -(((dataR_val==1)|(dataR_val==0))*np.log(1+np.exp(-omega_val)))-(dataR_val==0)*omega_val
    elif pos_and_negative==2:
        oms = omega_val + 1e-6
        return -(((dataR_val==1)|(dataR_val==0))*np.log((1+np.exp(-oms))))+(dataR_val==1)*np.log(1-np.exp(-oms))+(dataR_val==0)*(np.log(2)-oms)



def train_test_split(dataR, train_test_ratio):
    # Usage: split the data into training test through the train_test_ratio
    # Input:
    # dataR: observational data
    # train_test_ratio: the ratio of testing data in each row

    colNum = dataR.shape[0]
    colsel = int(colNum*train_test_ratio)
    dataR_test = np.ones(dataR.shape, dtype=int)*(-1)
    dataR = dataR.astype(int)
    for ti in range(colNum):
        col_index = np.random.choice(colNum, colsel, replace=False)
        dataR_test[ti, col_index] = dataR[ti, col_index]
        dataR[ti, col_index] = (-1)
    return dataR, dataR_test


def mh_continuous_propose(current_s_or_e_position, dim_d, lambdas_s, start_end_indicator):
    # Usage: use the prior distribution of the patch positions to propose the new values
    # Input:
    # current_s_or_e_position: the current value of the initial position or terminating position
    # dim_d: the dim_d-th dimension
    # lambdas: parameter control the size of the bounding boxes
    # start_end_indicator == 0: update the starter; start_end_indicator == 1: update the ender.

    # Output:
    # pp_position: the new proposal position for the initial position or terminating position


    random_dist = np.random.exponential(lambdas_s)
    if start_end_indicator == 0:
        if random_dist>(current_s_or_e_position):
            pp_position = 0
        else:
            pp_position = current_s_or_e_position-random_dist
        return np.array([pp_position, current_s_or_e_position-pp_position])
    elif start_end_indicator == 1:
        if random_dist>(dim_d-current_s_or_e_position):
            pp_position = dim_d-current_s_or_e_position
        else:
            pp_position = random_dist
        return pp_position


