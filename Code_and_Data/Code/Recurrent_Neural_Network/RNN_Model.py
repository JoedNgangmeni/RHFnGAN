import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from sklearn.model_selection import train_test_split

import math, sklearn, os, random
from sklearn.datasets import fetch_california_housing
# from ucimlrepo import fetch_ucirepo
from sklearn.datasets import load_iris


from datetime import datetime

# Setting up file locations
base_dir = os.path.dirname(os.path.abspath(__file__))  # Gets the directory where your script is
outpDir = "Output"
specOut =""

# Obtain current date for naming convention
current_date = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")

# LOOP CONTROLLERS
myDatasets = ['cali', 'air', 'fb' , 'aba', 'income', 'diabetes', 'cancer', 'wine', 'HAR' ]
N_ESTIMATORS = [1, 10, 50, 100, 150, 200]
DEPTH = [1, 3, 5, 7, 10, 15, 20]
MAX_RUNS = 50


# QUICK TEST CONTROLS
# MAX_RUNS = 3
# N_ESTIMATORS = [50]
# DEPTH = [1]
# myDatasets = [  'wine' ]


regressionDatasets = ['cali', 'air', 'fb' , 'aba']
classificationDatasets = ['income', 'diabetes', 'cancer', 'wine', 'HAR']

iris = load_iris()

for dataset in myDatasets:
    if dataset in regressionDatasets:
        if dataset == 'cali':
            specOut = "caliOutput"
            # https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html
            data = sklearn.datasets.fetch_california_housing(download_if_missing=True, as_frame=True)
            X = tf.convert_to_tensor(data.data)
            y = tf.convert_to_tensor(data.target)

        # elif dataset == 'air':
        #     specOut = "airOutput"
        #     # https://archive.ics.uci.edu/dataset/360/air+quality
        #     data = fetch_ucirepo(id=360) 
        #     temp = data.data.features.drop('PT08.S3(NOx)', axis =1)
        #     temp = temp.drop('Time', axis =1)
        #     temp = temp.drop('Date', axis =1)
        #     temp = temp.drop('NO2(GT)', axis =1)
        #     temp = temp.drop('PT08.S4(NO2)', axis =1)
        #     y = tf.convert_to_tensor(data.data.features['NOx(GT)'])
        #     X = tf.convert_to_tensor(temp.drop('NOx(GT)', axis =1))


        # elif dataset == 'fb':
        #     specOut = "fbOutput"
        #     # https://archive.ics.uci.edu/dataset/363/facebook+comment+volume+dataset
        #     pathToFile = os.path.join(base_dir, '../../Datasets/fb_git_repo/dataset/Training/')
        #     trainingVariants =['Features_Variant_1.csv', 'Features_Variant_2.csv', 'Features_Variant_3.csv', 'Features_Variant_4.csv', 'Features_Variant_5.csv']
        #     fileName = random.choice(trainingVariants)
        #     data = pd.read_csv(pathToFile+ fileName)
        #     X = tf.convert_to_tensor(data.iloc[:,:-1])
        #     y = tf.convert_to_tensor(data.iloc[:,-1])
        #     # y = y.values.ravel()

        
        # elif dataset =='aba':
        #     specOut = "abaOutput"
        #     # https://archive.ics.uci.edu/dataset/1/abalone
        #     data = fetch_ucirepo(id=1) 
        #     data['data']['features'].loc[:, 'Sex'] = data['data']['features']['Sex'].astype('category')

        #     # Create one-hot encoding
        #     one_hot_encoded = pd.get_dummies(data['data']['features']['Sex'], prefix='Sex')

        #     # Concatenate the one-hot encoded columns with the original DataFrame
        #     data['data']['features'] = pd.concat([data['data']['features'], one_hot_encoded], axis=1)

        #     # Drop the original 'Sex' column
        #     data['data']['features'] = data['data']['features'].drop('Sex', axis=1)

        #     # # data (as pandas dataframes) 
        #     X = tf.convert_to_tensor(data.data.features)
        #     y = tf.convert_to_tensor(data.data.targets)
        #     y = tf.convert_to_tensor(y.values.ravel())



        print('dataset name:' + dataset)
        print("Shape of X:", X.shape)
        print("Shape of y:", y.shape)
        # print("Type of y:", type(y))
        # print("y:", y)
        # print('\n\n')
        # print('X head: {X.head()}\n\n')
        # print('y head: {y}\n\n')




# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3) as tensorflow
        
        dataset = tf.data.Dataset.from_tensor_slices((X, y))
        print(dataset)
        xu
        dataset = X.shuffle(len(X))
        # Split the dataset into training and testing sets
        split_ratio = 0.7  # 80% training, 20% testing

        train_size = int(len(X) * split_ratio)


        X_train = dataset.take(train_size) #70%
        X_test = dataset.skip(train_size)  #30%
        y_train = dataset.take(train_size)


        

        
        # print(X_train.shape)
        # print(X_train[0].shape)
