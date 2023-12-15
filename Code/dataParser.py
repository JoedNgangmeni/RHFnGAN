import pandas as pd
import math, sklearn, os, random, sklearn.datasets
from ucimlrepo import fetch_ucirepo

base_dir = os.path.dirname(os.path.abspath(__file__))  # Gets the directory where your script is


def getCaliData():
    """
     Fetch California housing data from scikit - learn.
     
     
     @return X : 2D NumPy array shape = ( n_samples, n_features)
     @return y : 1D NumPy array shape = (n_samples,)
    """
    # https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html
    data = sklearn.datasets.fetch_california_housing(download_if_missing=True, as_frame=True)
    X = data.data
    y = data.target
    return X,y

def getAirData():
    """
     Get air data from UCI and return X and y. This is useful for testing
     
     
     @return X : 2D NumPy array shape = ( n_samples, n_features)
     @return y : 1D NumPy array shape = (n_samples,)
    """
    # https://archive.ics.uci.edu/dataset/360/air+quality
    data = fetch_ucirepo(id=360) 
    temp = data.data.features.drop('PT08.S3(NOx)', axis =1)
    temp = temp.drop('Time', axis =1)
    temp = temp.drop('Date', axis =1)
    temp = temp.drop('NO2(GT)', axis =1)
    temp = temp.drop('PT08.S4(NO2)', axis =1)
    y = data.data.features['NOx(GT)'] 
    X = temp.drop('NOx(GT)', axis =1)
    return X,y

def getFbData():
    """getFbData AI is creating summary for getFbData

    [extended_summary]

    Returns:
        [type]: [description]
    """
    # https://archive.ics.uci.edu/dataset/363/facebook+comment+volume+dataset
    pathToFile = os.path.join(base_dir, '../Datasets/fb_git_repo/dataset/Training/')
    trainingVariants =['Features_Variant_1.csv', 'Features_Variant_2.csv', 'Features_Variant_3.csv', 'Features_Variant_4.csv', 'Features_Variant_5.csv']
    fileName = random.choice(trainingVariants)
    data = pd.read_csv(pathToFile+ fileName)
    X_train = data.iloc[:,:-1]
    y_train = data.iloc[:,-1]

    pathToFile = os.path.join(base_dir, '../Datasets/fb_git_repo/dataset/Testing/TestSet/')
    testVariants =['Test_Case_1.csv', 'Test_Case_2.csv', 'Test_Case_3.csv', 'Test_Case_4.csv', 'Test_Case_5.csv', 'Test_Case_6.csv', 'Test_Case_7.csv', 'Test_Case_8.csv', 'Test_Case_9.csv', 'Test_Case_10.csv']
    fileName = random.choice(testVariants)
    data = pd.read_csv(pathToFile+ fileName)
    X_test = data.iloc[:,:-1]
    y_test = data.iloc[:,-1]

    return X_train, y_train, X_test, y_test
