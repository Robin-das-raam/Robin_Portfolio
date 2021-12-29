# Robin_Portfolio
Machine Learning Engineer Portfolio



# [Project 1: Predicting the Auction Sale Price of Bulldozers using Machine Learning.](https://github.com/Robin-das-raam/Machine-Learning-Algorithm/tree/main/Kaggle_Projects)
## Project Overview
The data used in this project was collected from  Kaggle Bluebook for Bulldozers competition.

* The project goal was to predict a number, so it's regression problem.
* Data used in this project was real world data. So there was lots of missing data.
* Explored the data and preprocessed the data.
* Created a Baseline model to see how well the model fitted .
* Feature Engineering: Created new Features to minimize the evaluation metric.
* Optimized Random Forest Regressor using RandomizedSearchCV and GridSearchCV to reach the best model.
* Evaluated the model with Root Mean Squared Log Error. 
* Tried to figure out the most important features.  
![Important_Features](https://github.com/Robin-das-raam/Machine-Learning-Algorithm/blob/main/Kaggle_Projects/Impotance%20Feat.png)


# [Project 2: Fraudulent Click Prediction.](https://github.com/Robin-das-raam/Machine-Learning-Algorithm/blob/main/Ensemble/XGBoost.ipynb)
## Project Overview
Click fraud is the practice of repeatedly clicking on an advertisement hosted on a website with the intention of generating revenue for the host website or draining revenue from the advertiser. The goal of this project is to Predict the fraud click.

**The problem was released on Kaggle and the original dataset contains observations of about 240 million clicks**, and whether a given click resulted in a download or not (1/0).On Kaggle, the data is split into train.csv and train_sample.csv (100,000 observations). Due to lack of computation power in my system and for better speed i have used **train_sample.csv** though the full training data will obviously produce better results.

In this project i applied **XGBOOST** boosting algorithm to solve this interesting **Classification** problem.
* Explored the data and preprocessed the data.
* Created a Baseline model to see how well the model fitted .
* Feature Engineering: Created new Features to minimize the evaluation metric.
* Optimized XGBoost classifier using GridSearchCV to reach the best model.
* Evaluated the model with classification report ,confusion matrix ,roc_auc_score and roc_curve. 
* Tried to figure out the most important features. 


# [Project 3: Build a RFM clustering and choose the best set of customers for online retail shop](https://github.com/Robin-das-raam/Machine-Learning-Algorithm/tree/main/Unsupervised%20Learning/KMeans)
## Project Overview

Online retail is a transnational data set which contains all the transactions occurring between 01/12/2010 and 09/12/2011 for a UK-based and registered non-store online retail.The company mainly sells unique all-occasion gifts. Many customers of the company are wholesalers.

Used the online reatil trasnational dataset to build a RFM clustering and choose the best set of customers.
* Explored the data and preprocessed the data.
* Used Boxplot to find out the outlier
* for checking the tendency of clusters used Hopkins Stat function
* applied K Means algorithm
* evaluate the model with Silhouette Analysis 
![](https://github.com/Robin-das-raam/Machine-Learning-Algorithm/blob/main/Unsupervised%20Learning/KMeans/kmean.png)

![](https://github.com/Robin-das-raam/Machine-Learning-Algorithm/blob/main/Unsupervised%20Learning/KMeans/kmeans_boxplot.png)







