
import numpy as np
import copy
from scipy.stats import norm
from scipy.stats import invgamma, gamma, poisson

from utilities import *


class PartitionPatch:


    def __init__(self, dimLength, dataNum, taus, lambdas, mus, variances):
        # Usage: Initialize the partition structure
        # Input:
        # dimLength: length vector for each dimensions
        # dataNum: number of data points
        # taus: paramter control the number of boxes
        # lambdas: parameter control the size of boxes

        # mus: prior for the intensity variable
        # variances: variance for the ydata

        # generate boxes parameters: K x D x position: patchPara[k,d,p] indicates the initial position (p=0) or length (p=1) in d-th dimension for k-th box

        self.patchNum = int(taus * np.prod(1 + lambdas * dimLength))  # number of boxes

        self.patchPara = patchPara_ini(self.patchNum, dimLength, lambdas)
        self.omegas = np.zeros(self.patchNum)

        self.mus = mus
        self.variances = variances

        self.tau = taus
        self.lambdas = lambdas
        self.dimLength = dimLength

        self.dimNum = len(dimLength)
        self.dataNum = dataNum



    def Metropolis_Hastings_A(self, xdata, ydata):
        # Usage: update the positions of the boxes
        # Input:
        # Training data: xdata: N x D; ydata: N x 1

        t_val = self.total_judge(xdata)

        for jj in np.arange(1, self.patchNum):

            current_total_val = np.dot(t_val, self.omegas)

            temp_total_val_missing = current_total_val - t_val[:, jj] * self.omegas[jj]

            for dim_d in range(len(self.dimLength)):
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

                    judges = np.prod((xdata>lower.reshape((1, -1)))&(xdata<=upper.reshape((1, -1))), axis=1)
                    change_index = (t_val[:, jj]!=judges)

                    old_total_val = temp_total_val_missing[change_index] + t_val[change_index, jj]*self.omegas[jj]
                    new_total_val = temp_total_val_missing[change_index] + judges[change_index]*self.omegas[jj]

                    old_change_ll = ll_cal(ydata[change_index], old_total_val, self.variances)
                    new_change_ll = ll_cal(ydata[change_index], new_total_val, self.variances)

                    if (np.log(np.random.rand()) < (new_change_ll - old_change_ll)):
                        self.patchPara[jj] = proposal_patchPara_j


    def Metropolis_Hastings_omegas(self, xdata, ydata):
        # Usage: update the intensities \omega of the boxes

        t_val = self.total_judge(xdata)
        prior_mu = self.mus
        # prior_variance = ((np.max(ydata)-np.min(ydata))**2)/(4*4)
        prior_variance = (0.5/3)**2

        for jj in range(self.patchNum):

            temp_total_val = np.dot(t_val, self.omegas)
            y_differ = ydata[t_val[:, jj]] - (temp_total_val[t_val[:, jj]] - self.omegas[jj])

            posterior_variance = (prior_variance ** (-1) + len(y_differ) / self.variances) ** (-1)
            posterior_mean = posterior_variance * (prior_mu / prior_variance + np.sum(y_differ) / self.variances)

            self.omegas[jj] = norm.rvs(loc=posterior_mean, scale=posterior_variance ** (0.5))

    def hyperparameter_update(self, xdata, ydata, hyper_sigma_1, hyper_sigma_2):
        # Usage: update the variance for ydata
        # Input:
        # hyper_sigma_1, hyper_sigma_2: parameters for the prior distribution of variance: Inverse-Gamma distribution

        t_val = self.total_judge(xdata)

        temp_total_val = np.dot(t_val, self.omegas)
        y_differ = ydata - temp_total_val

        posterior_alpha = hyper_sigma_1 + len(y_differ)/2
        posterior_beta = hyper_sigma_2 + np.sum(y_differ**2)/2

        self.variances = invgamma.rvs(a = posterior_alpha, scale = posterior_beta)

    def hyper_sample(self):

        lambda_counts = np.sum((np.sum(self.patchPara, axis=2)<self.dimLength)+(self.patchPara[:, :, 0]>0))

        new_lambda = invgamma.rvs(a=3, scale=0.2, loc=0.95)

        conVal = np.sum(np.log(1+new_lambda*self.dimLength))
        new_numLL = poisson.logpmf(self.patchNum, self.tau*np.exp(conVal))

        conVal_old = np.sum(np.log(1+self.lambdas*self.dimLength))
        old_numLL = poisson.logpmf(self.patchNum, self.tau*np.exp(conVal_old))

        new_PosLL = conVal-new_lambda*np.sum(self.patchPara[:, :, 1])+lambda_counts*np.log(new_lambda)
        old_PosLL = conVal_old-self.lambdas*np.sum(self.patchPara[:, :, 1])+lambda_counts*np.log(self.lambdas)

        if np.log(np.random.rand())<(new_PosLL+new_numLL-old_PosLL-old_numLL):
            self.lambdas = new_lambda


    def sample_patchNum(self, xdata, ydata):
        # Usage: update the number of boxes

        t_val = self.total_judge(xdata)
        temp_total_val = np.dot(t_val, self.omegas)
        theta_star_log = np.log(self.tau)+np.sum(np.log(1+self.lambdas*self.dimLength))

        if (np.random.rand() > 0.5)|(self.patchNum<1):  # propose to add a patch

            # initialize a new patch
            new_patch_shape = patch_shape_gen(self.dimLength, self.lambdas)
            lower = new_patch_shape[:, 0]
            upper = np.sum(new_patch_shape, axis=1)

            judges = np.prod((xdata > lower.reshape((1, -1))) & (xdata <= upper.reshape((1, -1))), axis=1)
            prior_variance = (0.5 / 3) ** 2
            prior_mu = self.mus
            new_omega = np.random.normal(prior_mu, prior_variance)

            old_total_val = temp_total_val[judges]
            new_total_val = temp_total_val[judges] +  new_omega

            old_change_ll = ll_cal(ydata[judges], old_total_val, self.variances)
            new_change_ll = ll_cal(ydata[judges], new_total_val, self.variances)

            if np.log(np.random.rand())<(new_change_ll + theta_star_log-np.log(self.patchNum+1) - old_change_ll):
                # add one patch
                self.patchPara = np.concatenate((self.patchPara, new_patch_shape[np.newaxis, :,:]), axis = 0)
                self.omegas = np.hstack((self.omegas, new_omega))
                self.patchNum +=1
        else: # propose to remove a patch
            select_i = np.random.choice(self.patchNum)
            temp_t_val = t_val[:, select_i]

            old_total_val = temp_total_val[temp_t_val]
            new_total_val = temp_total_val[temp_t_val] - self.omegas[select_i]

            old_change_ll = ll_cal(ydata[temp_t_val], old_total_val, self.variances)
            new_change_ll = ll_cal(ydata[temp_t_val], new_total_val, self.variances)

            if np.log(np.random.rand())<(new_change_ll -theta_star_log+np.log(self.patchNum) - old_change_ll):
                # remove one patch
                self.patchPara = np.delete(self.patchPara, select_i, axis=0)
                self.omegas= np.delete(self.omegas, select_i, axis=0)
                self.patchNum -=1


    def total_judge(self, xdata): # jj = -1 denotes to calculate the whole likelihood
        # Usage: judge the belonging of each feature data

        # Input:
        # indicator matrix, size: N x K

        lower = self.patchPara[:, :, 0]
        upper = np.sum(self.patchPara, axis=2)
        judges = (xdata[:, np.newaxis, :]>lower[np.newaxis, :, :])*(xdata[:, np.newaxis, :]<=upper[np.newaxis, :, :])

        return np.prod(judges, axis=2).astype(bool)
