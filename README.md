# Recomender-System
Recommender System

This code contains a recommendation system for the Yelp dataset review data: https://www.kaggle.com/yelp-dataset/yelp-dataset. It uses matrix factorization and a simple gradient descent update rule. It uses cross-validation to verify the results. 

The data are extremely sparse, which means that many customers have only rated a few places. Many of the customers that are found in the test set, do not appear in the training data. To circumvent this, we alter our approach. We introduce a new variable (n_naive), which indicates how many times a customer must appear in the training data before we just plug in the grand mean as the prediction for this customer. In a similar way the grand mean is used for unseen customers (customers that appear in the test set, but not in the training set). 

The test error is lower than the training error. This happens due to a more frequent imputation of the global mean in the test set than in the training set. The training error relies more on the matrix factorization process than the test error. The sparsity of the data is a problem here. 

# Parameters
- eeta: step size
- l: penalization factor to prevent overfitting
- n_naive: how many times must a customer must appear in the training set before we apply matrix factorization for this specfic customer.    It is generally a good idea to leave the customers that appear only once (n_naive = 2). 
- k: the dimensionality of matrix factorization
- iterations: how many times the training set is iterated over 

