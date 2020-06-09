import os
from functools import partial
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import pyro
import torch 
import seaborn as sns 
import pyro.distributions as dist 
from pyro import infer, optim
from pyro.infer.mcmc import HMC, MCMC
from pyro.infer import EmpiricalMarginal
from pyro.infer import SVI, Trace_ELBO
from torch.distributions import constraints
from pyro.infer.autoguide import AutoMultivariateNormal, init_to_mean

import csv
import random
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
import sys

def model(data, x_index, y_index, z_indices, feature_names): ## Note [1] 
    '''
    percent change in average movement,percent change in total Cases,Income,White_Percentage,
    Black_Percentage,Native_Percentage,Asian_Percentage,Hawaiian_Percentage,
    Other_Percentage,Multi_Percentage,Republican_Percentage,Democrat_Percentage,
    Independent_Percentage,Population_Density,Male_Percentage,
    Female_Percentage,Under_25_Percentage,25_49_Percentage,
    50_74_Percentage,Over_74_Percentage,Number_of_Cases,
    Percent_w_Corona
    '''

    #print("in model")
    #skip_indices = [9,12,15,19]
    size_z = len(z_indices)
    l_vs_y = torch.ones([size_z+1], dtype=torch.float)
    #l_vs_x = torch.ones([size_z], dtype=torch.float)

    

    
    l_vs_y[0] = pyro.sample("{}_weight_y".format(feature_names[x_index]),dist.Normal(0,10))
    for i in range(size_z):
        cur_z_index = z_indices[i]
        #if cur_z_index in skip_indices:
        #    continue
        #else:
        #l_vs_x[i] = pyro.sample("{}_weight_x".format(feature_names[cur_z_index]),dist.Normal(0,10))
        l_vs_y[i+1] = pyro.sample("{}_weight_y".format(feature_names[cur_z_index]),dist.Normal(0,10))

    #sigma_x = pyro.sample("{}_sigma".format(feature_names[x_index]),dist.Uniform(0., 10.))
    sigma_y = pyro.sample("{}_sigma".format(feature_names[y_index]),dist.Uniform(0., 10.))

    #movement = pyro.sample(feature_names[x_index], dist.Normal(data[:,z_indices].dot(l_vs_x),sigma_x), obs=data[:,0])

    

    x_z_indices = torch.zeros([len(z_indices)+1], dtype=torch.long)
    x_z_indices[0]=x_index
    x_z_indices[1:] = torch.tensor(z_indices,dtype=torch.long)
    #print("shape of dot output: {}".format(data[:,x_z_indices].mm(l_vs_y).shape))
    for i in pyro.plate("data_loop", data.shape[0], subsample_size=5000):
        #print("in loop: {}".format(i))
        covid_spread = pyro.sample("{}_{}".format(feature_names[y_index],i), dist.Normal(data[:,x_z_indices].mm(l_vs_y.unsqueeze(1)),sigma_y), obs=data[i,1])

data_path = '../dat/Data_Files/average_movement_Cases_data_7_day.csv'

data_fields = []
with open(data_path) as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    first = True
    for row in readCSV:
        if first:
            data_fields=row
            first = False
            break

z_indices = [3,4,5,6,7,8,10,11]

for i in range(11):
    cur_epoch = 100 * i
    cur_str = "../out/pyro_reg/checkpoints/epoch_{}_model.pth".format(cur_epoch)
    print(torch.load(cur_str))
    #guide=np.load(cur_str)
    guide = AutoMultivariateNormal(model, init_loc_fn=init_to_mean)
    
    for k in guide.state_dict():
        print("here")
        print(k)
    guide.load_state_dict(torch.load(cur_str))

    predictive = Predictive(model, guide=guide, num_samples=num_samples)
    svi_mvn_samples = {k: v.reshape(num_samples).detach().cpu().numpy()
                    for k, v in predictive(log_gdp, is_cont_africa, ruggedness).items()
                    if k != "obs"}
                    
    

    num_var = len(list(svi_mvn_samples.keys))
    x_1 = int(num_var/2)
    x_2 = num_var - x_1

    
    path = "../out/pyro_reg/posteriors"

    #fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(12, 10))
    #fig.suptitle("Marginal Posterior density - Regression Coefficients", fontsize=16)
    for k in svi_mvn_samples:
        fig,ax=plt.subplot(2, 2, 1)
        #site = sites[i]
        sns.distplot(svi_mvn_samples[k], ax=ax, label="SVI (Multivariate Normal) \nMarginal Posterior density")
        ax.set_title(k)
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper right');
        plt.close()