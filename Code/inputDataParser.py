import pandas as pd, numpy as np
import math, sklearn, os, random, sklearn.datasets


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
    return X, y

def getAirData():
    """
     Get air data from UCI and return X and y.
     
     
     @return X : 2D NumPy array shape = ( n_samples, n_features)
     @return y : 1D NumPy array shape = (n_samples,)
    """
    # https://archive.ics.uci.edu/dataset/360/air+quality
    pathToFile = os.path.join(base_dir, '../Datasets/extra_datasets_and_zips/air_quality/AirQualityUCI.csv')
    data = pd.read_csv(pathToFile, sep=';', header=0)

    columns_to_fix = ['CO(GT)', 'C6H6(GT)', 'T', 'RH', 'AH']

    for col in columns_to_fix:
        data[col] = data[col].str.replace(',', '.').astype(float)

    colsToDrop = ['Date', 'Time', 'NO2(GT)', 'PT08.S4(NO2)', 'PT08.S3(NOx)' , 'Unnamed: 15', 'Unnamed: 16']
    data = data.drop(columns=colsToDrop, axis=1)
    data = data.dropna()
    y = data['NOx(GT)'] 
    X = data.drop('NOx(GT)', axis=1)
    return X, y

def getFbData():
    """
    The getFbData function returns a tuple of two numpy arrays:
        X_train, y_train
    
    :return: (x_train, y_train)
    :doc-author: Trelent
    """
    # https://archive.ics.uci.edu/dataset/363/facebook+comment+volume+dataset
    pathToFile = os.path.join(base_dir, '../Datasets/fb_git_repo/dataset/Training/')
    trainingVariants =['Features_Variant_1.csv', 'Features_Variant_2.csv', 'Features_Variant_3.csv', 'Features_Variant_4.csv', 'Features_Variant_5.csv']
    fileName = random.choice(trainingVariants)
    data = pd.read_csv(pathToFile+ fileName)
    X_train = data.iloc[:,:-1]
    y_train = data.iloc[:,-1]

    # pathToFile = os.path.join(base_dir, '../Datasets/fb_git_repo/dataset/Testing/TestSet/')
    # testVariants =['Test_Case_1.csv', 'Test_Case_2.csv', 'Test_Case_3.csv', 'Test_Case_4.csv', 'Test_Case_5.csv', 'Test_Case_6.csv', 'Test_Case_7.csv', 'Test_Case_8.csv', 'Test_Case_9.csv', 'Test_Case_10.csv']
    # fileName = random.choice(testVariants)
    # data = pd.read_csv(pathToFile+ fileName)
    # X_test = data.iloc[:,:-1]
    # y_test = data.iloc[:,-1]
    # return X_train, y_train, X_test, y_test
    return X_train, y_train



def getAbaData():
    """
    The getAbaData function returns the Abalone dataset from UCI Machine Learning Repository.
    The data is a collection of physical measurements of abalones, and the target variable is age in years.
    The function returns two pandas dataframes: X (the features) and y (the targets). 
    
    :return: X (the features) and y (the targets)
    :doc-author: Trelent
    """

    # https://archive.ics.uci.edu/dataset/1/abalone
    abaHeader = ['Sex', 'Length', 'Diameter', 'Height', 'WholeWeight', 'ShuckedWeight', 'VisceraWeight', 'ShellWeight', 'Rings']
    pathToFile = os.path.join(base_dir, '../Datasets/extra_datasets_and_zips/uci-abalone/abalone.data.csv')
    data = pd.read_csv(pathToFile, sep=',', header=None, names=abaHeader) 
    data.loc[:, 'Sex'] = data['Sex'].astype('category')

    # Create one-hot encoding
    one_hot_encoded = pd.get_dummies(data['Sex'], prefix='Sex')

    # Concatenate the one-hot encoded columns with the original DataFrame
    data = pd.concat([data, one_hot_encoded], axis=1)

    # Drop the original 'Sex' column
    data = data.drop('Sex', axis=1)

    # # data (as pandas dataframes) 
    X = data.drop('Rings', axis=1)  # Exclude the target column 'Rings'
    y = data['Rings']   
    
    return X, y

def getIncomeData():
    """
    The getIncomeData function returns the income data from the UCI Machine Learning Repository.
    The function takes no arguments and returns two pandas DataFrames: X, which contains all of the features, 
    and y, which contains all of the targets. The target values are binary (YES or NO) and indicate whether an individual's income is greater than 50K.
    
    :return: A tuple of (X (the features), and y (the targets))
    :doc-author: Trelent
    """
    # https://archive.ics.uci.edu/dataset/2/adult
    # fetch dataset 

    incomeHeader = ['age', 'workclass', 'fnlwgt', 'education', 
                    'educationNum', 'maritalStatus', 'occupation', 'relationship', 
                    'race', 'sex', 'capitalGain', 'capitalLoss', 'hoursPerWeek', 'nativeCountry', 'income']
    
    pathToFile = os.path.join(base_dir, '../Datasets/extra_datasets_and_zips/adult/adult.data')
    data = pd.read_csv(pathToFile, sep=',', header=None, names=incomeHeader) 

    for nonNumFeat in data.columns:
        if nonNumFeat == 'income':
            continue
        if (data.dtypes[nonNumFeat] != np.int64) and (data.dtypes[nonNumFeat] != np.float64):
            # Create one-hot encoding
            one_hot_encoded = pd.get_dummies(data[nonNumFeat], prefix=nonNumFeat)

            # Concatenate the one-hot encoded columns with the original DataFrame
            data = pd.concat([data, one_hot_encoded], axis=1)

            # Drop the original column
            data = data.drop(nonNumFeat, axis=1)

    # data (as pandas dataframes) 
    X = data.drop('income', axis=1) 
    y = data['income'] 

    y[y == '<=50K.'] = '<=50K' 
    y[y == '>50K.'] = '>50K' 
    y[y == '<=50K'] = 'NO' 
    y[y == '>50K'] = "YES" 
    return X, y
            


def getDiabetesData():
    """
    The getDiabetesData function returns the X and y values for the diabetes dataset.
    The X value is a pandas dataframe containing all of the features, while y is a pandas series containing only the target variable.
    
    
    :return: A tuple of (X (the features), and y (the targets))
    :doc-author: Trelent
    """
    # https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database/?select=diabetes.csv
    pathToFile = os.path.join(base_dir, '../Datasets/diabetes.csv')
    data = pd.read_csv(pathToFile)
    X = data.iloc[:,:-1]
    y = data.iloc[:,-1]
    # y = y.values.ravel()
    return X, y

def getCancerData():
    """
    The getCancerData function returns the breast cancer dataset from UCI's machine learning repository.
    The data is returned as a tuple of two pandas dataframes, X and y.  The first contains the features, while
    the second contains the targets (labels).  The labels are in a numpy array format.
    
    :return: A tuple of (X (the features), and y (the targets))
    :doc-author: Trelent
    """
    # https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic
    # fetch dataset 

    cancerHeader = ['ID', 'diagnosis', 'radius1', 'texture1', 'perimeter1', 'area1', 'smoothness1', 
                    'compactness1', 'concavity1', 'concavePoints1', 'symmetry1', 'fractalDimension1',
                       'radius2', 'texture2', 'perimeter2', 'area2', 'smoothness2', 
                    'compactness2', 'concavity2', 'concavePoints2', 'symmetry2', 'fractalDimension2',
                      'radius3', 'texture3', 'perimeter3', 'area3', 'smoothness3', 
                    'compactness3', 'concavity3', 'concavePoints3', 'symmetry3', 'fractalDimension3' ]
    
    pathToFile = os.path.join(base_dir, '../Datasets/extra_datasets_and_zips/breast+cancer+wisconsin+diagnostic/wdbc.data')
    data = pd.read_csv(pathToFile, sep=',', header=None, names=cancerHeader) 
    
    # data (as pandas dataframes) 
    X = data.drop('diagnosis', axis=1) 
    y = data['diagnosis'] 

    return X, y

def getWineData():
    """
    The getWineData function fetches the wine quality dataset from UCI repository.
    The data is returned as a tuple of two pandas dataframes, X and y. 
    X contains the features (independent variables) and y contains the targets (dependent variable). 
    
    
    :return: A tuple of (X (the features), and y (the targets))
    :doc-author: Trelent
    """
    # https://archive.ics.uci.edu/dataset/186/wine+quality
    # fetch dataset     
    
    pathToFile = os.path.join(base_dir, '../Datasets/extra_datasets_and_zips/wine+quality/winequality-red.csv')
    data = pd.read_csv(pathToFile, sep=';', header=0)
    
    # data (as pandas dataframes) 
    X = data.drop('quality', axis=1) 
    y = data['quality']
    return X, y

def getHeartDisease():
    """
    The getHeartDisease function returns the heart disease dataset from UCI's machine learning repository.
    X contains the features (independent variables) and y contains the targets (dependent variable). 
    
    
    :return: a tuple of two pandas dataframes, X and y
    :doc-author: Trelent
    """
    # https://archive.ics.uci.edu/dataset/45/heart+disease
    # fetch dataset  

    clevelandHeader = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
                       'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'num']
    pathToFile = os.path.join(base_dir, '../Datasets/extra_datasets_and_zips/heart+disease/processed.cleveland.data')
    data = pd.read_csv(pathToFile, sep=',', header=None, names=clevelandHeader)
    
    # remove any rows with NaN or missing values
    data = data.dropna()
    data = data[data.apply(lambda row: "?" not in row.values, axis=1)]

    X = data.drop('num', axis=1)
    y = data['num']

    return X, y


def getHARData():
    """
    The getHARData function returns the Human Activity Recognition dataset.

    :return: (x_train,y_train)
    :doc-author: Trelent
    """
    # https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones
    traindata = pd.read_csv(os.path.join(base_dir, '../Datasets/hartrain.csv') , delimiter = ',')
    X_train = traindata.iloc[:,:-1]
    y_train = traindata.iloc[:,-1]

    # testdata = pd.read_csv(os.path.join(base_dir, '../Datasets/hartest.csv') , delimiter = ',')
    # X_test = testdata.iloc[:,:-1]
    # y_test = testdata.iloc[:,-1]
    # return X_train, y_train, X_test, y_test
    return X_train, y_train,

def getAllData():
    """
    The getAllData function returns a tuple of tuples. Each inner tuple contains two elements:
        1) A pandas dataframe containing the dataset's features and labels
        2) A string describing the dataset's name
    
    :return: A tuple of tuples
    :doc-author: Trelent
    """
    a, b = getCaliData()
    c, d = getAirData()
    e, f = getFbData()
    g, h = getAbaData()
    i, j = getIncomeData()
    k, l = getDiabetesData()
    m, n = getCancerData()
    o, p = getWineData()
    q, r = getHARData()
    return a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r


def getRegData(dataset: str): 
    if dataset == 'cali':
        X,y = getCaliData()
    elif dataset == 'air':
        X,y = getAirData()
    elif dataset == 'fb':
        X,y = getFbData()
    elif dataset == 'aba':
        X,y = getAbaData()
    return X, y

def getClsData(dataset: str): 
    if dataset == 'income':
        X,y = getIncomeData()
    elif dataset == 'diabetes':
        X,y = getDiabetesData()
    elif dataset == 'cancer':
        X,y = getCancerData()
    elif dataset == 'wine':
        X,y = getWineData()
    elif dataset == 'HAR':
        X,y = getHARData()
    elif dataset == 'heart':
        X,y = getHeartDisease()
    return X, y
