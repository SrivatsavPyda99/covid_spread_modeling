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
from pyro.infer.autoguide import AutoMultivariateNormal, init_to_mean, AutoDiagonalNormal
from pyro.infer import Predictive


import csv
import random
from sklearn import datasets, linear_model
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error as mse
import sys

def eval(epoch,data, m, sd, x_index,z_indices,y_index):
    x_z_indices = torch.LongTensor(size=[len(z_indices) + 1])
    x_z_indices[0] = x_index
    x_z_indices[1:] = torch.LongTensor(z_indices).int()

    out_rand = torch.zeros(data.shape[0])
    out_model = torch.zeros(data.shape[0])

    for i in range(data.shape[0]):
        rand_weights = torch.FloatTensor(np.random.normal(0,10,size=len(z_indices)+2))
        sample_model_weights = m.sample_latent()
        #print(data.shape)
        #r = data[:,x_z_indices]
        out_rand[i] = data[i,x_z_indices].dot(rand_weights[:-1]) + \
            rand_weights[-1]
        out_model[i] = data[i,x_z_indices].dot(sample_model_weights[:-1]) + \
            sample_model_weights[-1]


    
    '''
    num_rvs = sd['loc'].shape[0]
    for i in range(num_rvs):
        weight_1 = torch.FloatTensor(np.random.normal(sd['loc'][i],sd['scale_unconstrained'][i],size=data.shape[0]))
        weight_2 = torch.FloatTensor(np.random.normal(0,10,size=data.shape[0]))

        if i < num_rvs-1:
            var = data[x_z_indices[i]]
        else:
            var = 1

        out1+= weight_1 * var
        out2+= weight_2 * var
    '''
    
    print("MSE for random weights: {}".format(mse(out_rand.detach().numpy(),
                                                data[:,y_index].detach().numpy())))
    print("MSE for learned weights at epoch {}: {}".format(epoch, mse(out_model.detach().numpy(),
                                                            data[:,y_index].detach().numpy())))
# saves the model and optimizer states to disk
def save_posteriors(epoch, m,m_path,checkpoint_path,x_index,z_indices, y_index, data_fields,data,num_samples=100):
    x_z_indices = torch.ones([len(z_indices) + 1]).int()
    x_z_indices[0] = x_index
    x_z_indices[1:] = torch.Tensor(z_indices).int()
    names = []
    for i in range(x_z_indices.shape[0]):
        names.append(data_fields[x_z_indices[i]])
    names.append("{}_sigma".format(data_fields[y_index]))

    print("saving posteriors to {}...".format(m_path))

    td = m.state_dict()
    eval(epoch, data, m,td, x_index,z_indices,y_index)
    torch.save(td, checkpoint_path)
    #print(td)
    #sys.exit(0)

    for i in range(len(names)):
        cur_name = names[i]
        cur_mn = td['loc'][i]
        cur_std = td['scale_unconstrained'][i]
        cur_dist = dist.Normal(cur_mn,cur_std)

        pts = torch.ones([num_samples])
        for j in range(num_samples):
            cur = cur_dist.sample()
            pts[j] = cur


        fig,ax=plt.subplots()
        #site = sites[i]
        sns.distplot(pts, ax=ax, 
                label="SVI")
        ax.set_title(cur_name)
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper right');
        plt.savefig("{}/{}".format(m_path,cur_name))
        plt.close()

    #sys.exit(0)
    #torch.save(m.state_dict(), m_path)

# saves the model and optimizer states to disk
def save_checkpoint(m,o,m_path,o_path):
    print("saving model to {}...".format(m_path))
    torch.save(m.state_dict(), m_path)
    #print(m.state_dict())
    #sys.exit(0)
    print("saving optimizer states to {}...".format(o_path))
    o.save(o_path)
    print("done saving model and optimizer checkpoints to disk.")

# loads the model and optimizer states from disk
def load_checkpoint(m,o,m_path,o_path):
    print("loading model from {}...".format(m_path))
    m.load_state_dict(torch.load(m_path))
    print("loading optimizer states from {}...".format(o_path))
    o.load(o_path)
    print("done loading model and optimizer states.")

#assert pyro.__version__.startswith('0.3')
def histogram_feature(feature_name, feature,num_bins):
    plt.hist(feature, bins=num_bins)

    plt.savefig('outcome_matching/histogram_{}.png'.format(feature_name))
    plt.close()

def digitize(data):
    bins_map = dict()
    for i in range(data.shape[1]):

        bins = get_bins(data,i)
        #bins = np.arange(data[:,i].min(),data[:,i].max(), data[:,i].std() * bin_scaling_factor)
        print("number of bins: {}".format(bins.shape[0]))
        data[:,i] = np.digitize(data[:,i], bins)
        bins_map[data_fields[i]] = bins

    #histogram_feature("pct average movement", data[:,0],len(bins_map[data_fields[0]]))
    return bins_map,data

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
        #if i ==3:
        #    l_vs_y[i+1] = pyro.sample("{}_weight_y".format(feature_names[cur_z_index]),dist.Normal(0,0))
        #    continue
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
    for i in pyro.plate("data_loop", data.shape[0], subsample_size=1000):
        #print("in loop: {}".format(i))
        covid_spread = pyro.sample("{}_{}".format(feature_names[y_index],i), dist.Normal(data[:,x_z_indices].mm(l_vs_y.unsqueeze(1)),sigma_y), obs=data[i,1])

def guide(data, x_index, y_index, z_indices, feature_names):
    size_z = len(z_indices)
    l_vs_y = torch.ones([size_z+1], dtype=torch.float)
    #l_vs_x = torch.ones([size_z], dtype=torch.float)

    l_vs_y_loc = torch.ones([size_z+1], dtype=torch.float)
    l_vs_y_scale = torch.ones([size_z+1], dtype=torch.float)

    #l_vs_x_loc = torch.ones([size_z], dtype=torch.float)
    #l_vs_x_scale = torch.ones([size_z], dtype=torch.float)

    l_vs_y_loc[0] = pyro.param('{}_weight_y_loc'.format(feature_names[x_index]), torch.tensor(0.))
    l_vs_y_scale[0] = pyro.param('{}_weight_y_scale'.format(feature_names[x_index]), torch.tensor(1.), constraint=constraints.positive)

    for i in range(size_z):
        cur_z_index = z_indices[i]
        #l_vs_x_loc[i] = pyro.param('{}_weight_x_loc'.format(feature_names[cur_z_index]), torch.tensor(0.))
        #l_vs_x_scale[i] = pyro.param('{}_weight_x_scale'.format(feature_names[cur_z_index]), torch.tensor(1.),constraint=constraints.positive)
        l_vs_y_loc[i+1] = pyro.param('{}_weight_y_loc'.format(feature_names[cur_z_index]), torch.tensor(0.))
        l_vs_y_scale[i+1] = pyro.param('{}_weight_y_scale'.format(feature_names[cur_z_index]), torch.tensor(1.),constraint=constraints.positive)

    #sigma_x_loc = pyro.param('{}_sigma_loc'.format(feature_names[x_index]), torch.tensor(1.),
    #                         constraint=constraints.positive)
    sigma_y_loc = pyro.param('{}_sigma_loc'.format(feature_names[y_index]), torch.tensor(1.),
                             constraint=constraints.positive)

    l_vs_y[0] = pyro.sample("{}_weight_y".format(feature_names[x_index]),dist.Normal(l_vs_y_loc[0],l_vs_y_scale[0]))
    for i in range(size_z):
        cur_z_index = z_indices[i]
        #l_vs_x[i] = pyro.param("{}_weight_x".format(feature_names[cur_z_index]), dist.Normal(l_vs_x_loc[i],l_vs_x_scale[i]))
        l_vs_y[i+1] = pyro.sample("{}_weight_y".format(feature_names[cur_z_index]),dist.Normal(l_vs_y_loc[i+1],l_vs_y_scale[i+1]))
        
    #sigma_x = pyro.sample("{}_sigma".format(feature_names[x_index]),dist.Normal(sigma_x_loc,1))
    sigma_y = pyro.sample("{}_sigma".format(feature_names[y_index]),dist.Normal(sigma_y_loc,1))
def normalize(data):
    means = []
    std_devs = []
    for i in range(data.shape[1]):
        cur_mean = data[:,i].mean()
        cur_stddev = data[:,i].std()
        means.append(cur_mean)
        std_devs.append(cur_stddev)
        data[:,i] = (data[:,i] - cur_mean)/cur_stddev
    return data,means,std_devs


data_path = '../dat/Data_Files/average_movement_Cases_data_7_day.csv'
checkpoint_path = '../out/pyro_reg/checkpoints_full_normalized'
posterior_path = '../out/pyro_reg/posteriors_full_normalized'
run_number = 1
loadFromCheckpoint = False



data_fields = []
l = []
with open(data_path) as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    first = True
    for row in readCSV:
        if first:
            data_fields=row
            first = False
            continue
        #print("here")
        l.append(row)

data = np.array(l).astype(np.float64)
#data = data[:10,:]
data,means,stddevs = normalize(data)
# Remove zeros in data

remove_indices = []

#print("min income: {}".format(data[:,2].min()))

for i in range(data.shape[0]):
    if data[i,2] == 0.0:
        #print("the median income of county {} is 0".format(i))
        #print("the data for this row: {}".format(data[i,:]))
        remove_indices.append(i)

data = np.delete(data,remove_indices,axis=0)


# Log scale income

#data[:,2] = np.log(data[:,2])

# Digitize data

# bins_map, data = digitize(data)

z_indices = [2,3,4,5,6,7,8,10,11,13,14,16,17,18]
#z_indices = [3,4,5,6,7,8,10,11]
#z_indices = list(range(data.shape[1]))[2:]
y_index = 1
x_index = 0

data = torch.tensor(data, dtype=torch.float)

print("data shape: {}".format(data.shape))





#svi = SVI(model,
#          guide,
#          optim.Adam({"lr": .01}),
#          loss=Trace_ELBO())

num_iters = 5000
opt = optim.Adam({"lr": .01})

#m = model(data, x_index, y_index, z_indices, data_fields)
#guide = AutoMultivariateNormal(model, init_loc_fn=init_to_mean)
#for k in guide.state_dict():
#    print("here")
#    print(k)
#modelt = model(data, x_index, y_index, z_indices, data_fields)
#guidet=guide(data, x_index, y_index, z_indices, data_fields)
guide = AutoDiagonalNormal(model)
svi = SVI(model,
          guide,
          opt,
          loss=Trace_ELBO())
#for k in guide.state_dict():
#    print("here")
#    print(k)
if loadFromCheckpoint:
    load_checkpoint(svi.guide, opt, "{}/epoch_{}_model.pth".format(checkpoint_path,run_number),"{}/epoch_{}_opt.pth".format(checkpoint_path,run_number))

for i in range(num_iters):
    #elbo=svi.step(None)
    elbo = svi.step(data, x_index, y_index, z_indices, data_fields)
    if i % 10 == 0:
        #predictive = Predictive(model, guide=guide, num_samples=15)
        #print(svi.guide.state_dict())
        #print(predictive(data, x_index, y_index, z_indices, data_fields))
        #svi_mvn_samples = {k: v.reshape(num_samples).detach().cpu().numpy()
        #            for k, v in predictive(data, x_index, y_index, z_indices, data_fields).items()
        #            if "percent change in total Cases" not in k}
        #for k in svi_mvn_samples:
        #    print("{}: {}".format(k,svi_mvn_samples[k]))
        model_path = "{}/epoch_{}_model.pth".format(checkpoint_path,i)
        opt_path = "{}/epoch_{}_opt.pth".format(checkpoint_path,i)
        save_path = "{}/epoch_{}".format(posterior_path,i)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        #save_checkpoint(svi.guide,opt,model_path,opt_path)
        save_posteriors(i,svi.guide, save_path,model_path,x_index,z_indices, y_index, data_fields,data)
    
    #print(svi.guide.get_posterior())
    print("Elbo loss after epoch {}: {}".format(i+1,elbo))


#predictive = Predictive(model, guide=guide, num_samples=num_samples)
#svi_mvn_samples = {k: v.reshape(num_samples).detach().cpu().numpy()
#                   for k, v in predictive(log_gdp, is_cont_africa, ruggedness).items()
#                   if k != "obs"}
#fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(12, 10))
#fig.suptitle("Marginal Posterior density - Regression Coefficients", fontsize=16)
#for i, ax in enumerate(axs.reshape(-1)):
#    site = sites[i]
#    sns.distplot(svi_mvn_samples[site], ax=ax, label="SVI (Multivariate Normal)")
#    sns.distplot(hmc_samples[site], ax=ax, label="HMC")
#    ax.set_title(site)
#handles, labels = ax.get_legend_handles_labels()
#fig.legend(handles, labels, loc='upper right');