import numpy as np
import scipy
import matplotlib.pyplot as plt
import time
import sys
import scipy.io
import pandas as pd
import glob
import scipy.io as sio

from utilities import *
from partition_class import *



def run_mainfunction():

    pathss = 'data/'
    filess = glob.glob(pathss + '*.mat')
    identities = np.random.rand()
    for file_i in filess:
        train_test_ratio = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        for tt_r in train_test_ratio:
            mat_contents = sio.loadmat(file_i)
            dataR = mat_contents['datas']

            # define the train test ratio
            dataR, dataR_test = train_test_split(dataR, tt_r)

            print('\n')
            print('++++ new data: '+file_i)

            lambdas = 1
            taus = 3.0
            dimLength = np.array([1.0, 1.0]) # denote the length for each dimensions

            IterationTime = 200

            dataNum = dataR.shape[0]

            # initialize the partition structure
            # pos_and_negative: 1; pos: 2
            pos_and_negative = 1
            RBP = PartitionPatch(dimLength, dataNum, lambdas, taus, pos_and_negative)

            auc_seq = []
            precision_seq = []

            for tt in range(IterationTime):

                RBP.Metropolis_Hastings_omegas(dataR)
                RBP.Metropolis_Hastings_A(dataR)
                RBP.Metropolis_Hastings_coor(dataR)
                RBP.sample_patchNum(dataR)

                auc_val, precision_val = RBP.AUC_PRECISION_cal(dataR_test)
                auc_seq.append(auc_val)
                precision_seq.append(precision_val)

                # if np.mod(tt+1, 50)==0:
                #     print('============= Iteration '+str(tt+1) + ' finished. =============')
                #     print('auc is: '+str(auc_val), ', precision is: '+ str(precision_val))
                #     print('number of boxes is: '+str(RBP.patchNum))

            dataname = file_i[10:(-4)]
            np.savez_compressed('data/dataname+'_'+str(identities)+'_testratio_'+str(tt_r)+'.npz', RBP=RBP, auc_seq=auc_seq, precision_seq=precision_seq,
                                dataR = dataR, dataR_test = dataR_test)



if __name__ == '__main__':
    run_mainfunction()