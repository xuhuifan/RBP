import numpy as np
import scipy
import matplotlib.pyplot as plt
import time
import sys
import scipy.io
import pandas as pd
from scipy.stats import uniform
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor

from sklearn import linear_model

from utilities import *
from partition_class import *
from benchmark_comparison import *


def Friedman_function_gen(dimNum,dataNum):
    # Usage: generate the synthetic data through the Friedman's function
    # Input:
    # dimNum: number of dimensions for feature data
    # dataNum: number of data points

    xdata = uniform.rvs(size=(dataNum, dimNum))
    ydata = 10*np.sin(xdata[:, 0]*np.pi*xdata[:, 1])+20*((xdata[:, 2]-0.5)**2)+10*xdata[:, 3]+5*xdata[:, 4]+norm.rvs(size=dataNum)

    return xdata, ydata

def test_run():
    dataNum = 1000
    dimNum = 5
    lambdas = 3
    taus = 0.04

    run_mainfunction(dataNum, dimNum, lambdas, taus)



def run_mainfunction(dataNum, dimNum, lambdas, taus):

    xdata, ydata = Friedman_function_gen(dimNum, dataNum)
    train_test_ratio = 0.5 # the ratio of test data
    xdata_train, ydata_train, xdata_test, ydata_test, ydata_train_mean, dd, hyper_sigma_1, hyper_sigma_2, variance_hat = pre_process_data(xdata, ydata, train_test_ratio)

    dimLength = np.ones(dimNum)*1.0 # the length vector for all the dimensions


    IterationTime = 1000

    RBP = PartitionPatch(dimLength, dataNum, taus, lambdas, np.mean(ydata_train), variance_hat)
    print('Expected number of boxes is: '+str(RBP.patchNum))

    predicted_value_train_seq = np.zeros((IterationTime, len(ydata_train)))
    predicted_value_test_seq = np.zeros((IterationTime, len(ydata_test)))
    train_RMAE_seq = np.zeros(IterationTime)
    test_RMAE_seq = np.zeros(IterationTime)
    Numbox = np.zeros(IterationTime)
    for tt in range(IterationTime):
        RBP.Metropolis_Hastings_omegas(xdata_train, ydata_train)
        RBP.Metropolis_Hastings_A(xdata_train, ydata_train)
        RBP.hyperparameter_update(xdata_train, ydata_train, hyper_sigma_1, hyper_sigma_2)
        RBP.sample_patchNum(xdata_train, ydata_train)

        tvals_train = RBP.total_judge(xdata_train)
        predicted_value_train_seq[tt] = np.dot(tvals_train, RBP.omegas)
        tvals_test = RBP.total_judge(xdata_test)
        predicted_value_test_seq[tt] = np.dot(tvals_test, RBP.omegas)

        train_RMAE_seq[tt] = np.mean(abs(predicted_value_train_seq[tt]-ydata_train)*dd)
        test_RMAE_seq[tt] = np.mean(abs(predicted_value_test_seq[tt]-ydata_test)*dd)
        Numbox[tt] = RBP.patchNum
        if np.mod(tt+1, 100)==0:
            print('============= Iteration '+str(tt+1) + ' finished. =============')
            print('Number of boxes is: '+ str(RBP.patchNum))

    final_RMAE_train = np.mean(abs(np.mean(predicted_value_train_seq[:, int(IterationTime/2):], axis=1).reshape((-1))-ydata_train.reshape((-1)))*dd)
    final_RMAE_test = np.mean(abs(np.mean(predicted_value_test_seq[:, int(IterationTime/2):], axis=1).reshape((-1))-ydata_test.reshape((-1)))*dd)


    print('RBP-RT last train: '+str(train_RMAE_seq[-1])+', test: '+str(test_RMAE_seq[-1]))

    print('RBP-RT train: '+str(final_RMAE_train)+', test: '+str(final_RMAE_test))
    compare_table = benchmark_compare(xdata_train, xdata_test, ydata_train, ydata_test, dd)

    np.savez_compressed('result/trial_1.npz', xdata_train = xdata_train, xdata_test = xdata_test, ydata_train= ydata_train,
                        ydata_test=ydata_test, train_RMAE_seq=train_RMAE_seq, test_RMAE_seq = test_RMAE_seq,
                        f_RMAE_train = final_RMAE_train, f_RMAE_test = final_RMAE_test, Numbox = Numbox,
                        predicted_value_train_seq = predicted_value_train_seq, predicted_value_test_seq=predicted_value_test_seq,
                        bechmark_table = compare_table)

    # plt.plot(train_RMAE_seq)
    # plt.plot(test_RMAE_seq)
    # plt.legend(['Train', 'Test'])
    # plt.show()


if __name__ == '__main__':
    test_run()