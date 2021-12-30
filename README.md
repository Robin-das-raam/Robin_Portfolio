
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
![Kmeans clusters image](kmean.png)

![](https://github.com/Robin-das-raam/Machine-Learning-Algorithm/blob/main/Unsupervised%20Learning/KMeans/kmeans_boxplot.png)


# [Project 4: Automate Nucleus Detection Using U_NET image Segmentation](https://github.com/Robin-das-raam/Deep-Learning/blob/main/FInding%20Nucleus%20using%20U_Net.ipynb)
## Project Overview:
This project was **Kaggle** Data Science Bowl 2018 Challenge with a Heading of **Spot Nuclei Speed Cure**.The project goal is to find Nuclei for helping the cureing process and it faster..
**Objective** - Automate the generation of the image masks
**Approach** - Used **U_NET CNN** for segmentation task to generate these mask with **Tensorflow**
* Preprocessed the image data for creating image mask of dimension 128x128
* Used **IOU** metric for evaluation
* Optimized **U_NET** with **CallBacks** function to reach the best model.


# [Project 5: Workout estimation and reps Counting using OpenCV.](https://github.com/Robin-das-raam/Computer-Vision-OpenCV/tree/main/Workout%20estimation%20and%20reps%20Counter)
## Project Overview:
### The goal of this project is to track various gym workout position and count the reputation of perfectly done exercise. If the exercise not done properly then it will be not counted..

* This project is based on **Python** and **OpenCV**
* created a PoseDetector Module using MediaPipe
* PoseDetector module can find pose and draw line on pose landmarks
* PoseDetector module has a function to calculate the angle between defined pose landmarks
* each kind of exercise have a predefined angle between pose landmarks for counting as perfect reputation 


# [Project 6: Virtual Painter](https://github.com/Robin-das-raam/Computer-Vision-OpenCV/tree/main/Virtual%20Painter)

## Project Overview:
In this project build an application for drawing on screen using real time video using python and OpenCV.
* created a HandTracking module using mediapipe
* Handtracking module can track the hand and get the position of fingers
* draw landmarks 
* used two fingers to select the color and eraser
* used one fingers to draw on screen 

# [Project 7: Anomaly Detection in Time Series Data.](https://github.com/Robin-das-raam/Time-Series-Analysis-/blob/main/Anomaly%20Detection%20in%20Time%20Series.ipynb)

## Project Overview
The goal of this project is to detect anomaly in time Series data.In this project i used Catfish Sales dataset which provides the sales from 1996 to 2000.
* Explored the data to find out the pattern and seasonality
* Created artificial anomaly data and add them into original dataset.
* Preprocessed the data to remove the trend by first differencing
* Created a baseline model with **SARIMA** 
* Trained the baseline model and test the model
* Evaluated the model with **MAPE and RMSE** 
* For predicting anomaly applied **Deviation Method** and **Seasonal Method**
* Used Mean of other month for removing the effect of anomaly 
* Applied **Rolling Forecast** Method for better Prediction

# [Project 8: Autonomous driving with Reinforcement Learning](https://github.com/Robin-das-raam/Reinforcement-Learning-/blob/main/autonomous%20driving.ipynb)

## Project Overview
Train a reinforcement learning agent to drive a race car on Car racing environment with Open AI Gym.
* used **CarRacing-v0** Box environment
* applied **PPO** algorithm from Stable_Baseline
* applied **CNN policy** to create model
* evaluated the model with **evaluate_policy** 




