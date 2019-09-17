import numpy as np 
import math 
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
import pandas as pd
import statistics
import csv


#Load data
ratings2 = np.genfromtxt("C:/Users/Niek/Desktop/Yelp/Review.csv", usecols=(1, 2, 3), delimiter=';', dtype='int')
ratings2 = np.delete(ratings2, 0, axis=0)

#Start Cross-Validation
np.random.seed(1234)
F = KFold(n_splits=5, shuffle=True)
dat = F.split(ratings2)

#Train data, step size, penalization, test data, number of times customer must appear in data before we apply MF, dimensionality (parameters)
def matrix_factorization_train(data, eeta, l, test, n_naive, k, iterations): #Train data, step size, penalization, test data, num
    
        #Find the indices of items that appear less than n_naive times in the training data
        unique, counts = np.unique(data[:,0], return_counts=True)
        unique = dict(zip(unique, counts))
        index = [k for k,v in unique.items() if int(v) in range(0, n_naive) ]
        index2 = [k for k,v in unique.items() if int(v) > n_naive -1  ]

        #Training data containing all items that occure more than n_naive times and fix indexing
        data_iter = pd.DataFrame(data,
                        columns = ["col0", "col1", "col2"])        
        
        data_iter = data_iter[~data_iter['col0'].isin(index)]
        data_iter = data_iter.sort_values('col0')
        data_iter['col3'] = pd.factorize(data_iter.col0)[0] + 1
        data_iter = data_iter.sort_values('col1')
        data_iter['col4'] = pd.factorize(data_iter.col1)[0] + 1
        
        #Convert training data to Pandas
        data = pd.DataFrame(data,
                        columns = ["col0", "col1", "col2"])        
        
        #Initialize c and r (customers, restaurant)
        d = len(np.unique(data_iter['col0']))
        b = len(np.unique(data_iter['col1'])) 
        c_mat = np.random.uniform(low=-0.01, high=0.01, size=d * k).reshape((d, k))
        r_mat = np.random.uniform(low=-0.01, high=0.01, size=k * b).reshape((k, b))

       
        mse = []
        mse_test =[]
 
        for i in range(iterations):
 
            #Loop through data and use simple gradient descent
            for rating in np.array(data_iter):

                j = int(rating[3]-1)
                k = int(rating[4]-1)
                r = rating[2]
                pred = np.dot(c_mat[j, :], r_mat[:, k])
                error = r - pred if r > 0 else 0
                
                #calculate gradients
                c_gradient = c_mat[j, :] + eeta * (2 * error * r_mat[:, k] - l * c_mat[j, :])
                r_gradient = r_mat[:, k] + eeta * (2 * error * c_mat[j, :] - l * r_mat[:, k])
               
                c_mat[j, :] = c_gradient
                r_mat[:, k] = r_gradient

                
     
    
            #calculate RMSE training data
            c = c_mat[(data_iter['col3']-1).astype(int),:]
            r =  r_mat[:,(data_iter['col4']-1).astype(int)]
         
            #Make predictions use matrix factorization
            pred = np.zeros(shape=len(data))
            pred[data['col0'].isin(index2)] = np.sum(c * r.T, axis = 1)
            
            #For low-appearance customers, just predict global mean
            pred[pred == 0] = np.mean(data['col2'])
            mse_it = math.sqrt(np.sum(pow(data['col2'] - np.array(pred), 2)) / len(pred))
 
 
            #calculate RMSE test data 
            idx = pd.DataFrame(ratings2[test],
                               columns = ["cola", "colb", "colc"])
          
            #find the observations that we want to predict using matrix factorization
            from_mf = idx[idx['cola'].isin(data_iter['col0'])]
            from_mf = from_mf[from_mf['colb'].isin(data_iter['col1'])]
         
            #find correct index by merging
            temp3 = pd.merge( from_mf,data_iter.drop_duplicates(subset = 'col0'), right_on = ['col0'], left_on = ['cola'], how = 'left')['col3']

            temp4 = pd.merge( from_mf, data_iter.drop_duplicates(subset = 'col1'), right_on = ['col1'], left_on = ['colb'], how = 'left')['col4']
            
            #predict using matrix factorization
            c = c_mat[(temp3-1).astype(int),:]
            r =  r_mat[:,(temp4-1).astype(int)]
            pred = np.zeros(shape=len(idx))
            pred[idx['cola'].isin(data_iter['col0']) & idx['colb'].isin(data_iter['col1']) ] = np.sum(c * r.T, axis = 1)
            
            #when matrix factorization is unwanted, predict global mean from training data
            pred[pred == 0] = np.mean(data['col2'])

            test_mse_it =  math.sqrt(np.sum(pow(idx['colc'] -  np.array(pred), 2)) / len(pred))

            #Append to lists for output
            mse_test.append(test_mse_it)
            mse.append(mse_it)
        
            print("Iteration " + str(i) + " : \n Train RMSE: " + str(mse_it)  + " \n TEST RMSE: " + str(test_mse_it) )
           
        return([c_mat, r_mat, mse, mse_test])
    
    
def my_main():

    print("Start of the matrix factorisation")
    
    fold = 1
    eeta = 0.01 #step size
    l = 0.005 #penalization factor
    k = 10 #parameters
    N = 3 #naive
    iterations = 20
    
    mf_mse = []
    mf_test_mse = []

    #for every fold
    for train, test in dat:

        print("Fold:", fold)
        matrix = matrix_factorization_train(ratings2[train], eeta, l, test, N, k, iterations)
        fold += 1
 
        mf_mse.append(matrix[2])   
        mf_test_mse.append(matrix[3])
        
        print("The RMSE of the matrix factorisation:")
  

    return([mf_mse, mf_test_mse, iterations])


outcome = my_main()

#training and test RMSE
mse1 = outcome[0]
mse2 = outcome[1]

#save Data
with open("rmse_train.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerows(mse1)

with open("rmse_test.csv", "w") as f:
    wr = csv.writer(f)
    wr.writerow(mse2)
        
#Average over all folds
mean_mse1 = [statistics.mean(k) for k in zip(mse1[0],mse1[1], mse1[2], mse1[3], mse1[4])]
mean_mse2 = [statistics.mean(k) for k in zip(mse2[0],mse2[1], mse2[2], mse2[3], mse2[4])]


df=pd.DataFrame({'x': range(2, outcome[2] + 1),  'Training RMSE': mean_mse1[1:outcome[2]], 'Test RMSE': mean_mse2[1:outcome[2]]})
 
# create picture of losses
plt.plot( 'x', 'Training RMSE', data=df, marker='')
plt.plot( 'x', 'Test RMSE', data=df, marker='')
plt.title('Error Matrix Factorisation', size = 16)
plt.ylabel('Error', size = 12)
plt.xlabel('Iterations', size = 12)

plt.legend()

plt.savefig('Training_error1.png')


