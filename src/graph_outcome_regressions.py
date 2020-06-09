import csv
import numpy as np
import random
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

def plot_reg(x, y, x_name, y_name):
    plt.scatter(x, y,  color='black')
    #plt.plot(diabetes_X_test, diabetes_y_pred, color='blue', linewidth=3)

    #plt.xticks(())
    #plt.yticks(())

    plt.xlabel(x_name)
    plt.ylabel(y_name)
    
    plt.axis([x.min(),x.max(),y.min(),y.max()])

    plt.savefig('outcome_regressions/{}_vs_{}.png'.format(x_name,y_name))
    plt.close()

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
print("data shape: {}".format(data.shape))

#indices = list(range(data.shape[1]))
#indices_features = list(np.ones([data.shape[1]]).astype(int))
#indices_features[1] = 0
#print("sum of indices_features: {}".format(sum(indices_features)))

for i in range(data.shape[1]):
    if i == 1:
        continue
    plot_reg(data[:,i],data[:,1],data_fields[i],data_fields[1])