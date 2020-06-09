import csv
import numpy as np
import random
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

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

data = np.array(l).astype(float)

remove_indices = []
print("min income: {}".format(data[:,2].min()))
for i in range(data.shape[0]):
    if data[i,2] == 0.0:
        print("the median income of county {} is 0".format(i))
        print("the data for this row: {}".format(data[i,:]))
        remove_indices.append(i)

data = np.delete(data,remove_indices,axis=0)
#data[:,2] = np.log(data[:,2])
#data[:,2] = np.log(data[:,2] + np.finfo(float).eps)
#data[:,1] = np.log(data[:,1])

data,means,stddevs = normalize(data)
#print(data_fields)
#print("number of data points: {}".format(data.shape[0]))

ind_excl = list(range(len(data_fields)))
ind_excl.remove(21)
ind_excl.remove(20)
ind_excl.remove(1)
#print(ind_excl)


outcomes = data[:,1]
features = data[:,ind_excl]
#features = data[:,[0,3,4,5,6,7,8,9,10,11,12]]

feature_names = ['percent change in average movement', 
                'White_Percentage', 
                'Black_Percentage', 
                'Native_Percentage', 
                'Asian_Percentage', 
                'Hawaiian_Percentage', 
                'Other_Percentage', 
                'Multi_Percentage', 
                'Republican_Percentage', 
                'Democrat_Percentage', 
                'Independent_Percentage']

#means_features = means[[0,3,4,5,6,7,8,9,10,11,12]]
#mean_outcome = means[1]


# Split into train and test datasets

pct_train = 0.9

num_train = int(pct_train * outcomes.shape[0])
#print("size of outcomes: {}".format(outcomes.shape[0]))
#print("num train: {}".format(num_train))
indices = random.sample(range(outcomes.shape[0]),num_train)
indices_train = np.zeros([outcomes.shape[0]]).astype(bool)
indices_train[indices] = 1
indices_test = 1 - indices_train

outcomes_train = outcomes[indices_train].astype(np.float64)
outcomes_test = outcomes[indices_test].astype(np.float64)
features_train = features[indices_train,:].astype(np.float64)
features_test = features[indices_test,:].astype(np.float64)


# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(features_train, outcomes_train)

# Make predictions using the testing set
#print(features_test.type())
y_pred = regr.predict(features_test)


# The coefficients
print('Coefficients: \n', regr.coef_)
index = 0
for i in ind_excl:
    print("Coefficient for {}: {}".format(data_fields[i], regr.coef_[index]))
    index=index+1

# The mean squared error
print('Mean squared error: %.2f'
      % mean_squared_error(outcomes_test, y_pred))

# The coefficient of determination: 1 is perfect prediction
print('Coefficient of determination: %.2f'
      % r2_score(outcomes_test, y_pred))


print("Some statistics about the outcomes: ")
print("mean of oucomes: {}".format(outcomes_test.mean()))
print("stddev of oucomes: {}".format(outcomes_test.std()))