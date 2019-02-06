import numpy as np
import copy
from scipy.stats import norm
from scipy.stats import invgamma
from sklearn.metrics import roc_auc_score, average_precision_score

from utilities import *


class PartitionPatch:


    def __init__(self, dimLength, dataNum, lambdas, taus, pos_and_negative):
        # Input:
        # patchNum: Number of boxes
        # dimLength: length vector for each dimensions
        # dataNum: number of data points
        # lambdas: parameter control the size of boxes

        # generate boxes parameters: K x D x position: patchPara[k,d,p] indicates the initial position (p=0) or length (p=1) in d-th dimension for k-th box
        self.patchNum = int(taus*(np.prod(1+lambdas*dimLength)))
        self.patchPara = patchPara_ini(self.patchNum, dimLength, lambdas)
        self.xx = np.random.uniform(size=dataNum)
        self.yy = np.random.uniform(size=dataNum)

        self.pos_and_negative = pos_and_negative
        if pos_and_negative == 1:
            self.omegas = np.random.normal(size=self.patchNum)
        elif pos_and_negative == 2:
            self.omegas = np.random.exponential(size=self.patchNum)


        self.tau = taus
        self.lambdas = lambdas

        self.dimLength = dimLength
        self.dataNum = dataNum




    def Metropolis_Hastings_A(self, dataR):
        # Usage: update the positions of the boxes
        # Input:
        # dataR: training relational data N x N

        t_val = self.total_judge(self.xx, self.yy)

        for jj in np.arange(1, self.patchNum):

            current_total_val = np.sum(t_val * self.omegas.reshape((1, 1, -1)), axis=2)
            temp_total_val_missing = current_total_val - t_val[:, :, jj] * self.omegas[jj]

            for dim_d in range(self.patchPara.shape[1]):
                for start_end_in in [0, 1]:
                    proposal_patchPara_j = copy.copy(self.patchPara[jj])
                    if start_end_in == 0:
                        new_position = mh_continuous_propose(np.sum(self.patchPara[jj, dim_d]), self.dimLength[dim_d],
                                                             self.lambdas, start_end_in)
                        proposal_patchPara_j[dim_d] = new_position
                    else:
                        new_locates = mh_continuous_propose(self.patchPara[jj, dim_d, 1 - start_end_in],
                                                            self.dimLength[dim_d], self.lambdas, start_end_in)
                        proposal_patchPara_j[dim_d, start_end_in] = new_locates

                    lower = proposal_patchPara_j[:, 0]
                    upper = np.sum(proposal_patchPara_j, axis=1)

                    x_judge = (self.xx > lower[0]) * (self.xx<= upper[0])
                    y_judge = (self.yy > lower[1]) * (self.yy<= upper[1])
                    temp_t_val = x_judge[:, np.newaxis]*y_judge[np.newaxis, :]

                    change_index = (t_val[:, :, jj]!=temp_t_val)

                    old_total_val = temp_total_val_missing[change_index] + t_val[:, :, jj][change_index]*self.omegas[jj]
                    new_total_val = temp_total_val_missing[change_index] + temp_t_val[change_index]*self.omegas[jj]

                    if np.log(np.random.rand())<(np.sum(LL_cal(dataR[change_index], new_total_val, self.pos_and_negative))-np.sum(LL_cal(dataR[change_index], old_total_val, self.pos_and_negative))):
                        self.patchPara[jj] = proposal_patchPara_j
                        t_val[:, :, jj] = temp_t_val


    def Metropolis_Hastings_omegas(self, dataR):
        # Usage: update the intensities \omega of the boxes
        # Input:
        # dataR: training relational data N x N

        t_val = self.total_judge(self.xx, self.yy)
        omega_dataR = np.sum(t_val*self.omegas.reshape((1, 1, -1)), axis=2)
        if self.pos_and_negative==1:
            new_omegas = np.random.normal(size=self.patchNum)
        elif self.pos_and_negative == 2:
            new_omegas = np.random.exponential(size=self.patchNum)

        for jj in range(self.patchNum):
            # origin omega
            tt_index = t_val[:, :, jj]
            LL_origin = np.sum(LL_cal(dataR[tt_index], omega_dataR[tt_index], self.pos_and_negative))

            # new omega
            LL_new = np.sum(LL_cal(dataR[tt_index], omega_dataR[tt_index]-self.omegas[jj]+new_omegas[jj], self.pos_and_negative))

            if np.log(np.random.rand())<(LL_new-LL_origin):
                omega_dataR = omega_dataR+(new_omegas[jj]-self.omegas[jj])*tt_index
                self.omegas[jj] = new_omegas[jj]

    def Metropolis_Hastings_coor(self, dataR):
        # Usage: update the coordinates for all the nodes
        # Input:
        # dataR: training relational data N x N

        new_x = np.random.uniform(size = self.dataNum)
        new_y = np.random.uniform(size = self.dataNum)

        old_t_val = self.total_judge(self.xx, self.yy)
        new_t_val = self.total_judge(new_x, self.yy)
        np.sum(abs(old_t_val.astype(int)-new_t_val.astype(int)))
        old_total_val = np.sum(old_t_val*self.omegas.reshape((1, 1, -1)), axis=2)
        new_total_val = np.sum(new_t_val*self.omegas.reshape((1, 1, -1)), axis=2)


        change_index = (np.log(np.random.uniform(size=self.dataNum))<(np.sum(LL_cal(dataR, new_total_val, self.pos_and_negative), axis=1)-np.sum(LL_cal(dataR, old_total_val, self.pos_and_negative), axis=1)))
        self.xx[change_index] = new_x[change_index]


        old_t_val = self.total_judge(self.xx, self.yy)
        new_t_val = self.total_judge(self.xx, new_y)
        old_total_val = np.sum(old_t_val*self.omegas.reshape((1, 1, -1)), axis=2)
        new_total_val = np.sum(new_t_val*self.omegas.reshape((1, 1, -1)), axis=2)

        change_index = (np.log(np.random.uniform(size=self.dataNum))<(np.sum(LL_cal(dataR, new_total_val, self.pos_and_negative), axis=0)-np.sum(LL_cal(dataR, old_total_val, self.pos_and_negative), axis=0)))
        self.yy[change_index] = new_y[change_index]


    def total_judge(self, xx_coor, yy_coor):
        # Usage: judge the belonging of each relation
        # Input:
        # xx_coor, yy_coor: coordinates for all the N nodes (xx_coor in dimension 1, yy_coor in dimension 2)

        # Input:
        # tvals: indicator matrix, size: N x N x K

        lower = self.patchPara[:, :, 0]
        upper = np.sum(self.patchPara, axis=2)

        x_judge = (xx_coor.reshape((-1, 1))>lower[:, 0].reshape((1, -1)))*(xx_coor.reshape((-1, 1))<=upper[:, 0].reshape((1, -1)))
        y_judge = (yy_coor.reshape((-1, 1))>lower[:, 1].reshape((1, -1)))*(yy_coor.reshape((-1, 1))<=upper[:, 1].reshape((1, -1)))
        tvals = x_judge[:, np.newaxis, :]*y_judge[np.newaxis, :, :]

        return tvals


    def sample_patchNum(self, dataR):
        # Usage: update the number of boxes
        # Input:
        # dataR: training relational data N x N

        t_val = self.total_judge(self.xx, self.yy)
        total_val = np.sum(t_val*self.omegas.reshape((1, 1, -1)), axis=2)
        # theta_star = self.tau*np.prod((self.lambdas + (1 - self.lambdas) * self.dimLength))
        theta_star_log = np.log(self.tau)+np.sum(np.log(1+self.lambdas*self.dimLength))

        if (np.random.rand() > 0.5)|(self.patchNum<1):  # propose to add a patch

            # initialize a new patch
            new_patch_shape = patch_shape_gen(self.dimLength, self.lambdas)
            lower = new_patch_shape[:, 0]
            upper = np.sum(new_patch_shape, axis=1)

            x_judge = (self.xx > lower[0]) * (self.xx <= upper[0])
            y_judge = (self.yy > lower[1]) * (self.yy <= upper[1])
            temp_t_val = x_judge[:, np.newaxis] * y_judge[np.newaxis, :]

            new_omega = np.random.normal()

            old_LL = np.sum(LL_cal(dataR[temp_t_val], total_val[temp_t_val], self.pos_and_negative))
            new_LL = np.sum(LL_cal(dataR[temp_t_val], total_val[temp_t_val]+new_omega, self.pos_and_negative))

            if np.log(np.random.rand())<(new_LL + theta_star_log-np.log(self.patchNum+1) - old_LL):
                # add one patch
                self.patchPara = np.concatenate((self.patchPara, new_patch_shape[np.newaxis, :,:]), axis = 0)
                self.omegas = np.hstack((self.omegas, new_omega))
                self.patchNum +=1
        else: # propose to remove a patch
            select_i = np.random.choice(self.patchNum)
            temp_t_val = t_val[:, :, select_i]

            old_LL = np.sum(LL_cal(dataR[temp_t_val], total_val[temp_t_val], self.pos_and_negative))
            new_LL = np.sum(LL_cal(dataR[temp_t_val], total_val[temp_t_val]-self.omegas[select_i], self.pos_and_negative))

            if np.log(np.random.rand())<(new_LL -theta_star_log+np.log(self.patchNum) - old_LL):
                # remove one patch
                self.patchPara = np.delete(self.patchPara, select_i, axis=0)
                self.omegas= np.delete(self.omegas, select_i, axis=0)
                self.patchNum -=1


    def AUC_PRECISION_cal(self, dataR_test):
        # Usage: calculate the AUC, Precision
        # Input:
        # dataR: testing relational data N x N

        t_val = self.total_judge(self.xx, self.yy)
        # total_val = np.sum(t_val*self.omegas.reshape((1, 1, -1)), axis=2)
        total_val = np.dot(t_val, self.omegas)
        prob_total = np.exp(LL_cal(dataR_test, total_val, self.pos_and_negative))
        # prob_total = 1/(1+np.exp(-total_val))
        auc_val = roc_auc_score(dataR_test[dataR_test!=(-1)], prob_total[dataR_test!=(-1)])
        precision_val = average_precision_score(dataR_test[dataR_test!=(-1)], prob_total[dataR_test!=(-1)])

        return auc_val, precision_val