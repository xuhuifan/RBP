
#from partition_class import *
import numpy as np
import scipy
import copy
import matplotlib.pyplot as plt
from matplotlib import cm
import time
from scipy.stats import poisson
from scipy.stats import norm
from matplotlib.patches import Circle, Wedge, Polygon
from matplotlib.collections import PatchCollection
from sklearn import linear_model
from scipy.stats import invgamma


def Friedman_pre_process(xdata_train, ydata_train, xdata_test, ydata_test):

    ydata_train_min = np.min(ydata_train)
    ydata_train_max = np.max(ydata_train)
    ydata_train_mean = (ydata_train_max+ydata_train_min)/2
    dd = ydata_train_max-ydata_train_min

    ydata_train = (ydata_train-ydata_train_mean)/dd
    ydata_test = (ydata_test-ydata_train_mean)/dd


    # set up the hyper-parameters
    regr = linear_model.LinearRegression()
    regr.fit(xdata_train, ydata_train)
    y_train_predict = regr.predict(xdata_train)
    variance_hat = np.var(y_train_predict-ydata_train)

    hyper_sigma_1 = 1.5
    percentile_val = 0.9

    val1 = invgamma.ppf(percentile_val, a = hyper_sigma_1, scale=1)
    hyper_sigma_2 = variance_hat/val1
    # invgamma.cdf(variance_hat, a=hyper_sigma_1, scale=hyper_sigma_2)
    # Calculate the standard deviation for least square regression

    return xdata_train, ydata_train, xdata_test, ydata_test, ydata_train_mean, dd, hyper_sigma_1, hyper_sigma_2, variance_hat



def pre_process_data(xdata, ydata, train_test_ratio):
    # Usage: pre-process the data
    # Input:
    # xdata, ydata: the original full data

    # Output:
    # xdata_train, ydata_train, xdata_test, ydata_test: the train and test data
    # ydata_train_mean: the "mean" of the original training data
    # dd: the difference between the maximum and minimum ydata_train
    # hyper_sigma_1, hyper_sigma_2: hyper-parameters for infer the variance
    # variance_hat: rough estimated variance for ydata_train

    test_index = np.random.choice(len(ydata), int(np.ceil(len(ydata)*train_test_ratio)), replace=False)
    train_index = np.delete(np.arange(len(ydata)), test_index)

    xdata_train = xdata[train_index]
    ydata_train = ydata[train_index]
    xdata_test = xdata[test_index]
    ydata_test = ydata[test_index]

    ydata_train_min = np.min(ydata_train)
    ydata_train_max = np.max(ydata_train)
    ydata_train_mean = (ydata_train_max+ydata_train_min)/2
    dd = ydata_train_max-ydata_train_min

    ydata_train = (ydata_train-ydata_train_mean)/dd
    ydata_test = (ydata_test-ydata_train_mean)/dd


    # set up the hyper-parameters
    regr = linear_model.LinearRegression()
    regr.fit(xdata_train, ydata_train)
    y_train_predict = regr.predict(xdata_train)
    variance_hat = np.var(y_train_predict-ydata_train)

    hyper_sigma_1 = 1.5
    percentile_val = 0.9

    val1 = invgamma.ppf(percentile_val, a = hyper_sigma_1, scale=1)
    hyper_sigma_2 = variance_hat/val1
    # invgamma.cdf(variance_hat, a=hyper_sigma_1, scale=hyper_sigma_2)
    # Calculate the standard deviation for least square regression

    return xdata_train, ydata_train, xdata_test, ydata_test, ydata_train_mean, dd, hyper_sigma_1, hyper_sigma_2, variance_hat


def patchPara_ini(patchNum, dimLength, lambdas):

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
    patch_shape = np.zeros([len(dimLength), 2])

    s_not_equal_0 = (np.random.uniform(size=len(dimLength))>(1/(1+lambdas*dimLength)))
    patch_shape[s_not_equal_0, 0] = np.random.uniform(low=0, high=dimLength[s_not_equal_0])

    patch_shape[:, 1] = np.random.exponential(lambdas, size=len(dimLength))
    violates_index = (np.sum(patch_shape, axis=1)>dimLength)
    patch_shape[violates_index, 1] = (dimLength-patch_shape[:, 0])[violates_index]

    return patch_shape




def ll_cal(ydata, mus, variances):
    ll_calval = np.sum(norm.logpdf(ydata, mus, variances**(0.5))) # need to consider the test case

    return ll_calval



def mh_continuous_propose(current_s_or_e_position, dim_d, lambdas_s, start_end_indicator):
    # start_end_indicator == 0: update the starter; start_end_indicator == 1: update the ender.
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





def incorporate_likelihood_position_propose(jj_patchPara, dimLength_s, lambdas_s, d_dimension, start_end_indicator, jj_xdatass_d, jj_ydatass, jj_temp_val, jj_omega, total_variance):
    # start_end_indicator == 0: update the starter; start_end_indicator == 1: update the ender.

    # use posterior probability to sample the position of patches

    # if ((start_end_indicator==0)&(jj_patchPara[d_dimension, 1]>0))|((start_end_indicator==1)&(jj_patchPara[d_dimension, 0]<(dimLength_s[d_dimension]-1))):

    if (start_end_indicator==0)&(jj_patchPara[d_dimension, 1]==0):
        return 0
    elif (start_end_indicator==1)&(jj_patchPara[d_dimension, 0]==(dimLength_s[d_dimension]-1)):
        return (dimLength_s[d_dimension]-1)
    else:
        if start_end_indicator == 0:

            candidate_set = np.arange(0, jj_patchPara[d_dimension, 1], dtype=int).reshape((1, -1))
            index_judge = (jj_xdatass_d>candidate_set)

            log_likeli_position = np.sum(norm.logpdf(jj_ydatass.reshape((-1, 1)), jj_temp_val.reshape((-1, 1))+jj_omega*index_judge, total_variance**(0.5)), axis=0)
            log_prior_position = np.log(lambdas_s)*(np.arange(len(candidate_set[0])-1, -1, -1))+np.log(1-lambdas_s)
            log_prior_position[0] -= np.log(1-lambdas_s)

        elif start_end_indicator == 1:

            candidate_set = np.arange(jj_patchPara[d_dimension, 0], dimLength_s[d_dimension], dtype=int).reshape((1, -1))
            index_judge = (jj_xdatass_d<candidate_set)

            log_likeli_position = np.sum(norm.logpdf(jj_ydatass.reshape((-1, 1)), jj_temp_val.reshape((-1, 1))+jj_omega*index_judge, total_variance**(0.5)), axis=0)
            log_prior_position = np.log(lambdas_s)*np.arange(len(candidate_set[0]))

        ll_posterior = log_likeli_position + log_prior_position
        posterior_prob = np.exp(ll_posterior - np.max(ll_posterior))

        selected_value = np.random.choice(len(candidate_set[0]), p = posterior_prob/np.sum(posterior_prob))

        return candidate_set[0, selected_value]

