'''
File specification: this file intends to indentify the average treatment effect of following social distancing guidelines on
the change in percentage of active cases. We seek to use the backdoor criterion to estimate the effect using a nearest neighbor
regression style matching. We define a "close enough" match to be within 0.1 standard deviations of the distribution.
'''

import csv
import numpy as np
import random
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
import sys
import matplotlib.pyplot as plt

def get_indices(data,y_index,x_index,z_indices):
    x_map_to_z = dict()
    x_z_index_map = dict()
    x_index_map = dict()

    z_indices = np.array(z_indices).astype(int)
    x_z_indices = np.empty([z_indices.shape[0]+1]).astype(int)
    y_x_z_indices = np.empty([x_z_indices.shape[0]+1]).astype(int)
    x_z_indices[0] = x_index
    x_z_indices[1:] = z_indices
    y_x_z_indices[1] = y_index
    y_x_z_indices[0] = x_index
    y_x_z_indices[2:] = z_indices

    x_y_indices = np.array([x_index,y_index]).astype(int)

    for i in range(data.shape[0]):
        x_z_vec = tuple(data[i,x_z_indices])
        x = data[i,int(x_index)]
        if x not in x_index_map:
            x_index_map[x] = list()

        if x_z_vec not in x_z_index_map:
            x_z_index_map[x_z_vec] = list()

            if x not in x_map_to_z:
                x_map_to_z[x] = list()
            
            x_map_to_z[x].append(tuple(data[i,z_indices]))

        x_z_index_map[x_z_vec].append(i)
        x_index_map[x].append(i)
    

    return x_map_to_z,x_z_index_map, x_index_map

#def c_ATE()
def get_match(x, z, x_z_map):
    #print(z)
    #print(x_z_map[x])
    if z in x_z_map[x]:
        return True
    return False


def get_do_expectations(data, x_index, y_index, z_indices,bins_map, data_fields):
    x_map_to_z,x_z_index_map, x_index_map = get_indices(data,y_index,x_index,z_indices)
    cnt_z, cnt_x_z,cnt_y_x_z, cnt_x_y = get_counts(data, y_index, x_index, z_indices)

    ATE = np.zeros([len(bins_map[data_fields[x_index]])-1])

    for i in range(ATE.shape[0]):
        
        x_2 = i+2
        x_1 = i+1

        print("Calculating E[y|do(x={})] - E[y|do(x={})]".format(x_2,x_1))

        x_z_index_map_cur = x_z_index_map.copy()

        total_cnt = 0
        total_val = 0
        for j in x_index_map[x_1]:
            cur_z = data[j,z_indices]

            if get_match(x_2,tuple(cur_z),x_map_to_z):
                total_cnt +=1
                x_z_vec = np.empty([len(z_indices)+1]).astype(int)
                x_z_vec[0] = x_2
                x_z_vec[1:] = cur_z
                x_z_vec = tuple(list(x_z_vec))

                # pick a random index from the x_2 z indices

                rand_num = random.randint(0, len(x_z_index_map_cur[x_z_vec])-1)


                # delete that particular match from the x_2 z indices
                cur_index = x_z_index_map_cur[x_z_vec].pop(rand_num)
                if not x_z_index_map_cur[x_z_vec]:
                    x_map_to_z[x_2].remove(tuple(cur_z))

                # add y_val

                total_val += data[cur_index,y_index] - data[i,y_index]

        ex = total_val/total_cnt
            
        print("E[y|do(x={})] - E[y|do(x={})] = {}".format(x_2,x_1, ex))

        ATE[i] = ex
    return ATE

        
def get_counts(data, y_index, x_index, z_indices):
    cnt_y_x_z = dict()
    cnt_x_z = dict()
    cnt_z = dict()
    cnt_x_y=dict()
    z_indices = np.array(z_indices).astype(int)
    x_z_indices = np.empty([z_indices.shape[0]+1]).astype(int)
    y_x_z_indices = np.empty([x_z_indices.shape[0]+1]).astype(int)
    x_z_indices[0] = x_index
    x_z_indices[1:] = z_indices
    y_x_z_indices[1] = y_index
    y_x_z_indices[0] = x_index
    y_x_z_indices[2:] = z_indices
    x_y_indices = np.array([x_index,y_index])

    #z_indices = tuple(z_indices)
    #x_z_indices = tuple(x_z_indices)
    #y_x_z_indices = tuple(y_x_z_indices)

    for i in range(data.shape[0]):
        z_vec = tuple(data[i,z_indices])
        x_z_vec = tuple(data[i,x_z_indices])
        y_x_z_vec = tuple(data[i,y_x_z_indices])
        x_y_vec = tuple(data[i,x_y_indices])
        
        if z_vec in cnt_z:
            cnt_z[z_vec] += 1
            if x_z_vec in cnt_x_z:
                cnt_x_z[x_z_vec] += 1
                if y_x_z_vec in cnt_y_x_z:
                    cnt_y_x_z[y_x_z_vec] += 1
                else:
                    cnt_y_x_z[y_x_z_vec] = 1
            else:
                cnt_x_z[x_z_vec] = 1
                cnt_y_x_z[y_x_z_vec] = 1
        else:
            cnt_z[z_vec] = 1
            cnt_x_z[x_z_vec] = 1
            cnt_y_x_z[y_x_z_vec] = 1
        if x_y_vec in cnt_x_y:
            cnt_x_y[x_y_vec] +=1
        else:
            cnt_x_y[x_y_vec] =1
        
    return cnt_z, cnt_x_z,cnt_y_x_z, cnt_x_y


def get_bins(data, index):
    data_cur = data[:,index]
    mean = np.mean(data_cur)
    standard_deviation = np.std(data_cur)
    distance_from_mean = abs(data_cur - mean)
    max_deviations = 2
    not_outlier = distance_from_mean < max_deviations * standard_deviation
    
    #print("not outlier shape: {}".format(not_outlier.shape))
    #print("data_cur")
    no_outliers = data_cur[not_outlier]
    max_no_outliers = no_outliers.max()
    min_no_outliers = no_outliers.min()
    outliers = 1 - not_outlier.astype(int)
    print("number of outliers: {}".format(outliers.sum()))
    print("proportion of data captured without loss: {}".format(not_outlier.sum()/outliers.shape[0]))

    if index == 1:
        bin_scaling_factor = 0.1
    else:
        bin_scaling_factor = 0.1

    #bins = get_bins(data,i)
    bins = np.arange(min_no_outliers,max_no_outliers, standard_deviation * bin_scaling_factor)

    return bins
    

#def calc_p_z(data, index_y, index_x, index_z, y_bins, num_bins=25):
    # Calculate expected value of y by calculating which bins happen at which percentage and then
    # weighting by the corresponding amount.

#    data_y = data[:,index_y]
#    data_x = data[:,index_x]
#    data_z = data[:,index_z]

def plot_reg(x, y, x_name, y_name):
    plt.scatter(x, y,  color='black')
    #plt.plot(diabetes_X_test, diabetes_y_pred, color='blue', linewidth=3)

    #plt.xticks(())
    #plt.yticks(())

    plt.xlabel(x_name)
    plt.ylabel(y_name)
    
    plt.axis([x.min(),x.max(),y.min(),y.max()])

    plt.savefig('outcome_matching/new_2_{}_vs_{}.png'.format(y_name,x_name))
    plt.close()


'''
E[y|do(x)] = \sum_{y}y * P[y|do(x)] = \sum_{y} y * \sum_z P[y|x,z]P(z)
'''
def calc_causal_effect(num_iter, data, x_val, index_y, index_x, index_z, bins_map, data_fields):
    # Steps
    # Calculate highest number of bins
    # March each of the covariates up by 1 unless they've already hit their peaks,
    # And then only march what's left
    max_num_bins = 0 
    bins = []

    for index in index_z:
        bins.append(len(bins_map[data_fields[index]]))
        #numbins = len(v)
        #if numbins > max_num_bins:
        #    max_num_bins = numbins

    print("bins: {}".format(bins))
    def cur_field_maxed(index, field, bins_map):
        if index > len(bins_map[field]):
            return True
        return False

    def calc_p_z(data,index_z,z_vec):
        #cnt = (data[:,index_z]==z_vec).astype(int).sum(axis=1)
        #cnt = (cnt == z_vec.shape[0]).astype(int).sum()
        #if cnt > 0:
        #    print("cnt {}, data shape {}".format(cnt,data.shape[0]))
        cnt = get_cnt(data,index_z,z_vec)
        #if cnt > 0:
        #    print("cnt {}, data shape {}".format(cnt,data.shape[0]))
        #    print(z_vec)
        return cnt/data.shape[0], cnt

    def get_cnt(data,index_z,z_vec):
        cnt = (data[:,index_z]==z_vec).all(axis=1)
        cnt = cnt.astype(int).sum()
        #cnt = 0
        #for i in range(data.shape[0]):
            #print("data: {}".format(data[i,index_z]))
            #print("z_vec: {}".format(z_vec))
            
        #    new = int((data[i,index_z].astype(int)==z_vec).all())
        #    if(new ==1):
        #        print("upping cnt")
            #if (data[i,index_z] == [5., 2., 1., 1., 1., 1., 7., 2.]).all():
            #    sys.exit(0)
        #    cnt = cnt + new
        #cnt = ((data[:,index_z]==z_vec).astype(int).all()).sum()
        #cnt = (cnt == z_vec.shape[0]).astype(int).sum()
        return cnt


    def calc_p_y_given_x_z(data, index_y, index_x,index_z,y_val,x_val, z_vec, bins_map):
        num_dim_z = z_vec.shape[0]

        p_x_z_indices =np.empty([num_dim_z+1])
        p_x_z_vec =np.empty([num_dim_z+1])
        p_x_z_indices = index_z
        p_x_z_vec = z_vec
        p_x_z_indices = np.insert(p_x_z_indices,0,index_x)
        p_x_z_vec = np.insert(p_x_z_vec,0,x_val)


        

        cnt_p_x_z = get_cnt(data,p_x_z_indices,p_x_z_vec)
        if cnt_p_x_z == 0:
            return 0.0

        cnt_p_y_x_z = np.zeros([len(bins_map[index_y])]).astype(float)
        for i in range(len(bins_map[index_y])):
            y_weight = bins_map[index_y][i]
            p_y_x_z_indicez = np.empty([len(index_z)+1])
            p_y_x_z_indices = index_z
            p_y_x_z_vec = np.array(z_vec)
            p_y_x_z_indices = np.insert(p_y_x_z_indices,0,index_y)
            p_y_x_z_indices = np.insert(p_y_x_z_indices,0,index_x)
            p_y_x_z_vec = np.insert(p_y_x_z_vec,0,y_val)
            p_y_x_z_vec = np.insert(p_y_x_z_vec,0,x_val)
            cnt_p_y_x_z[i] = get_cnt(data,p_y_x_z_indices,p_y_x_z_vec) * y_val

        #print("cnt_p_x_z: {}".format(cnt_p_x_z))
        #print("cnt_p_y_x_z: {}".format(cnt_p_y_x_z))
        #print("p_y_x_z_indices: {}".format(p_y_x_z_indices))
        #print("p_y_x_z_vec: {}".format(p_y_x_z_vec))
        
        prob = cnt_p_y_x_z/cnt_p_x_z

        #if prob > 1:
        #    print("prob was too high, system broke: {}".format(prob))
        #    sys.exit(0)
        return prob

    
    #print("z_vec: {}".format(z_vec))
    e_y_do_x = 0.0
    y_ind = 0
    for y_val in bins_map[data_fields[index_y]]:
        y_ind = y_ind+1
        it = 0
        p_y_do_x = 0.0
        p_z_s = 0
        total_cnt = 0
        for z_vec in np.ndindex(tuple(bins)):
            z_vec_mod = np.array(z_vec)+1
            it = it + 1
            #print(z_vec)
            p_z,cnt = calc_p_z(data,index_z,z_vec_mod)
            if p_z == 0.0:
                #print("p_z = 0 for z = {}".format(z_vec))
                #print("causal prob so far: {}".format(p_y_do_x))
                continue
            
            
            p_z_s = p_z_s + p_z
            p_y_given_x_z = calc_p_y_given_x_z(data, index_y, index_x,index_z,y_ind,x_val, z_vec_mod)
            p_y_do_x = p_y_do_x + p_y_given_x_z * p_z
            total_cnt = total_cnt + cnt
            #z_vec[vec_bin_num] = z_vec[vec_bin_num]+1
            #vec_bin_num = vec_bin_num+1
        #print("sum of p_zs: {}".format(p_z_s))
        #print("number of iterations covered vs expected: {},{}".format(it, num_iter))
        #print("total cnt: {}".format(total_cnt))
        
        e_y_do_x = e_y_do_x + y_val * p_y_do_x
        print("Finished an iteration through z variables")
    return e_y_do_x



data_path = 'Data_Files/average_movement_Cases_data_7_day.csv'
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

remove_indices = []
#print("min income: {}".format(data[:,2].min()))
for i in range(data.shape[0]):
    if data[i,2] == 0.0:
        #print("the median income of county {} is 0".format(i))
        #print("the data for this row: {}".format(data[i,:]))
        remove_indices.append(i)

data = np.delete(data,remove_indices,axis=0)
data[:,2] = np.log(data[:,2])

# change outcome to log scale

#data[:,1] = np.log(data[:,1])

def histogram_feature(feature_name, feature,num_bins):
    plt.hist(feature, bins=num_bins)

    plt.savefig('outcome_matching/histogram_{}.png'.format(feature_name))
    plt.close()

# digitize data
bins_map = dict()
for i in range(data.shape[1]):
    if i ==1:
        continue
    bins = get_bins(data,i)
    #bins = np.arange(data[:,i].min(),data[:,i].max(), data[:,i].std() * bin_scaling_factor)
    print("number of bins: {}".format(bins.shape[0]))
    data[:,i] = np.digitize(data[:,i], bins)
    bins_map[data_fields[i]] = bins

histogram_feature("pct average movement", data[:,0],len(bins_map[data_fields[0]]))

#z_indices = [2,3,4,5,6,7,8,10,11,13]
#z_indices = [3,4,5,6,7,8,10,11]
z_indices = [2,3,4,5,6,7,8,10,11,13,14,15,16,17,18,19]
#z_indices = list(range(data.shape[1]))[2:]
y_indices = 1
x_indices = 0

tot = 1
for index in z_indices:
    numb = len(bins_map[data_fields[index]])
    tot = tot * numb

print("total number of combos: {}".format(tot))

def calc_ATE(num_iter, data, xval_1,xval_2,y_indices, x_indices, z_indices, bins_map, data_fields):
    causal_effect_xval_1 = calc_causal_effect(num_iter, data, xval_1, y_indices, x_indices, z_indices, bins_map, data_fields)
    causal_effect_xval_2 = calc_causal_effect(num_iter, data, xval_2, y_indices, x_indices, z_indices, bins_map, data_fields)
    return causal_effect_xval_2 - causal_effect_xval_1

#for i in range(10):
#    print(data[i,:])

num_x_bins = len(bins_map[data_fields[x_indices]])
x_0 = 1
x_1 = 2

#for i in range(len(bins_map[x_indices])):
#    x_val = 1 + i
#    print("E[y|do(x={})] = {}".format(i, causal_effect_xval_1 = calc_causal_effect(tot, data, xval, y_indices, x_indices, z_indices, bins_map, data_fields)))
#print("E[y|do(x={})] - E[y|do(x={})] = {}".format(x_1,x_0,calc_ATE(tot, data,x_0,x_1,y_indices, x_indices, z_indices, bins_map, data_fields)))

cnt_z, cnt_x_z,cnt_y_x_z,cnt_x_y = get_counts(data,y_indices,x_indices,z_indices)

e_y_do_x = np.zeros([len(bins_map[data_fields[x_indices]])])
#cnt_x_y = np.zeros([len(bins_map[data_fields[x_indices]]), len(bins_map[data_fields[y_indices]])])

'''
for y_x_z_vec in cnt_y_x_z:
    y_x_z_vec_l = list(y_x_z_vec)
    #print("y_x_z_vec: {}".format(y_x_z_vec_l))
    x_z_vec_l = y_x_z_vec_l.copy()
    y = x_z_vec_l.pop(1)
    z_vec_l = x_z_vec_l.copy()
    x = z_vec_l.pop(0)
    x_z_vec = tuple(x_z_vec_l)
    z_vec = tuple(z_vec_l)

    #x = y_x_z_vec_l[0]
    #y = y_x_z_vec_l[1]

    #y_val = bins_map[data_fields[y_indices]][int(y-1)]
    y_val = y

    #print("y_x_z_vec: {}".format(y_x_z_vec_l))
    #print("x_z_vec: {}".format(x_z_vec))
    #print("z_vec: {}".format(z_vec))

    p_y_given_x_z = cnt_y_x_z[y_x_z_vec]/cnt_x_z[x_z_vec]
    p_z = cnt_z[z_vec]/data.shape[0]


    e_y_do_x[int(x)-1] += y_val * p_y_given_x_z * p_z
'''
'''
for y_x_z_vec in cnt_y_x_z:
    y_x_z_vec_l = list(y_x_z_vec)
    x_z_vec_l = y_x_z_vec_l.copy()
    y = x_z_vec_l.pop(1)
    z_vec_l = x_z_vec_l.copy()
    x = z_vec_l.pop(0)
    x_z_vec = tuple(x_z_vec_l)
    z_vec = tuple(z_vec_l)

    e_y_do_x[int(x)-1] += y
    cnt_x[int(x)-1] += 1
'''


'''
for i in range(data.shape[0]):
    x = int(data[i,0])
    y = int(data[i,1])
    e_y_do_x[int(x)-1]+=y
    cnt_x_y[int(x)-1, int(y)-1] +=1


e_y_do_x = e_y_do_x/cnt_x
'''

'''
z_indices = np.array(z_indices).astype(int)
x_z_indices = np.empty([z_indices.shape[0]+1]).astype(int)
y_x_z_indices = np.empty([x_z_indices.shape[0]+1]).astype(int)
x_z_indices[0] = x_indices
x_z_indices[1:] = z_indices
y_x_z_indices[1] = y_indices
y_x_z_indices[0] = x_indices
y_x_z_indices[2:] = z_indices
x_y_indices = np.array([x_indices,y_indices]).astype(int)

#z_indices = tuple(z_indices)
#x_z_indices = tuple(x_z_indices)
#y_x_z_indices = tuple(y_x_z_indices)

for i in range(data.shape[0]):
    z_vec = tuple(data[i,z_indices])
    x_z_vec = tuple(data[i,x_z_indices])
    y_x_z_vec = tuple(data[i,y_x_z_indices])
    x_y_vec = tuple(data[i,x_y_indices])
'''
'''
for x in range(len(bins_map[x_indices])):
    cur_x_val = x+1
    for y in range(len(bins_map[y_indices])):
        cur_y_val = y+1
        for z_vec in cnt_z:
            x_y_vec = tuple([cur_x_val,cur_y_val])
            x_y_vec_l = np.array(list(x_y_vec))
            y_x_z_vec = np.empty([x_y_vec_l.shape[0] + len(z_indices)])
            y_x_z_vec[:2] = x_y_vec_l
            y_x_z_vec[2:] = np.array(list(z_vec)))

            e_y_do_x[x] += cnt_y_x_z[y_x_z_vec] * bins_map[y_indices][cur_y_val]/ cnt_x_y[x_y_vec]
    #e_y_do_x[x]



for i in range(e_y_do_x.shape[0]):
    print("E[y|do(x={})]={}".format(i+1,e_y_do_x[i]))
    print("count: {}".format(cnt_x_y[i]))


plot_reg(np.array(range(e_y_do_x.shape[0]))+1, e_y_do_x, "percent average movement", "percent changes in total cases")
#print(a)
print("min count: {}".format(np.min(cnt_x)))

'''
print("max y: {}".format(np.max(data[:,1])))
print("min y: {}".format(np.min(data[:,1])))
print("bins map test: {}".format(len(bins_map[data_fields[x_indices]])))
ATE = get_do_expectations(data, x_indices, y_indices, z_indices,bins_map, data_fields)


plot_reg(np.array(range(ATE.shape[0]))+2, ATE, "percent average movement", "ATE")