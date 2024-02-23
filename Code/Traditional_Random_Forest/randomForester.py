import math, os, time
import pandas as pd, numpy as np
import inputDataParser as parse
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def isEmpty(file_path):
    return os.stat(file_path).st_size == 0

def regressionRuns(model: str, task: str, allDatasets: list, regDatasets: list, ESTNUM: int, DEPTH: int, MAX_RUNS: int, rawDataPath: str, aggDataPath: str ):
    """
    The regressionRuns function is used to run the Random Forest Regressor on a given dataset.
   
    :param model: str: Specify the type of model to be used ('RF', 'XGB', etc...)
    :param task: str: Specify the type of task being performed ('reg')
    :param allDatasets: list: a list with the abreviated names of all available datasets
    :param regDatasets: list: Specify which datasets to run the regression on
    :param ESTNUM: int: Set the number of trees in the forest
    :param DEPTH: int: Set the depth of each tree in the forest
    :param MAX_RUNS: int: Set the number of times we want to run the model with that number of trees and depth
    :param rawDataPath: str: Specify the path to the directory where you want your raw data saved
    :param aggDataPath: str: Specify the path to where the aggregated data will be stored
    :return: None
    :doc-author: Trelent
    """
    for dataset in allDatasets:
        if dataset in regDatasets: 

            X,y = parse.getRegData(dataset)

            for numEstimators in ESTNUM:
                for depth in DEPTH:
                    runNumber = 1

                    while (runNumber < MAX_RUNS + 1):
                        print(f'\n{dataset} Run number:\t{runNumber}')


                        # Set file name system for raw data
                        saveRawDataHere = os.path.join(rawDataPath, dataset, f'_{numEstimators}_{depth}_{dataset}_{model}_{task}_')

                        # add header to raw and agg file
                        with open(saveRawDataHere, 'a') as raw_file:
                            if isEmpty(saveRawDataHere):
                                raw_file.write(f"numTrees\ttreeDepth\tr2\trmse\tmse\tmae\tbuildTime\n") 
                        
                        # Set file name system for agg data
                        saveAggDataHere = os.path.join(aggDataPath, f'_{dataset}_{model}_{task}_')
                        
                        # add header to agg data file 
                        with open(saveAggDataHere, 'a') as agg_file:
                            if isEmpty(saveAggDataHere):
                                agg_file.write(f"numTrees\ttreeDepth\tr2\trmse\tmse\tmae\tbuildTime\n")

                        # run and time forest building
                        # start_time = time.time()
                        # r2, rmse, mse, mae = growRegressor(numEstimators, depth, X, y)
                        # finish_time = time.time()
                        # buildtime = finish_time - start_time

                        # # write data to file
                        # print(f'saving data in {saveRawDataHere}')
                        # with open(saveRawDataHere, 'a') as raw_file:
                        #     raw_file.write(f"{numEstimators}\t{depth}\t{r2}\t{rmse}\t{mse}\t{mae}\t{buildtime}\n")

                        # # increment counter    
                        runNumber += 1

def classificationRuns(model: str, task: str, allDatasets: list, clsDatasets: list, ESTNUM: int, DEPTH: int, MAX_RUNS: int, rawDataPath: str, aggDataPath: str ):
    """
    The regressionRuns function is used to run the Random Forest Regressor on a given dataset.
   
    :param model: str: Specify the type of model to be used ('RF', 'XGB', etc...)
    :param task: str: Specify the type of task being performed ('cls')
    :param allDatasets: list: a list with the abreviated names of all available datasets
    :param regDatasets: list: Specify which datasets to run the regression on
    :param ESTNUM: int: Set the number of trees in the forest
    :param DEPTH: int: Set the depth of each tree in the forest
    :param MAX_RUNS: int: Set the number of times we want to run the model with that number of trees and depth
    :param rawDataPath: str: Specify the path to the directory where you want your raw data saved
    :param aggDataPath: str: Specify the path to where the aggregated data will be stored
    :return: None
    :doc-author: Trelent
    """
    for dataset in allDatasets:
        if dataset in clsDatasets: 

            X,y = parse.getClsData(dataset)

            for numEstimators in ESTNUM:
                for depth in DEPTH:
                    runNumber = 1
                    while (runNumber < MAX_RUNS + 1):
                        print(f'\n{dataset} Run number:\t{runNumber}')

                        # Set file name system for raw data
                        saveRawDataHere = os.path.join(rawDataPath, dataset, f'_{numEstimators}_{depth}_{dataset}_{model}_{task}_')

                        # add header to raw and agg file
                        with open(saveRawDataHere, 'a') as raw_file:
                            if isEmpty(saveRawDataHere):
                                raw_file.write(f"numTrees\ttreeDepth\tf1\taccuracy\tprecision\trecall\tbuildTime\n") 
                        
                        # Set file name system for agg data
                        saveAggDataHere = os.path.join(aggDataPath, f'_{dataset}_{model}_{task}_')
                        
                        # add header to agg data file 
                        with open(saveAggDataHere, 'a') as agg_file:
                            if isEmpty(saveAggDataHere):
                                agg_file.write(f"numTrees\ttreeDepth\tf1\taccuracy\tprecision\trecall\tbuildTime\n")

                        # # run and time forest building
                        # start_time = time.time()
                        # f1, accuracy, precision, recall, conf_matrix = growClassifier(numEstimators, depth, X, y)
                        # finish_time = time.time()
                        # buildtime = finish_time - start_time

                        # # write data to file
                        # print(f'saving data in {saveRawDataHere}')
                        # with open(saveRawDataHere, 'a') as raw_file:
                        #     raw_file.write(f"{numEstimators}\t{depth}\t{f1}\t{accuracy}\t{precision}\t{recall}\t{buildtime}\n")

                        # increment counter    
                        runNumber += 1

def growRegressor(NUMTREES: int, DEPTH: int, X: pd.DataFrame , y: np.ndarray):
    """
    The growRegressor function takes in the number of trees, depth, and data as input.
    It then splits the data into training and testing sets. It initializes a random forest regressor with
    the given parameters (number of trees, depth). It trains the model on the training set and makes predictions on 
    the test set. Finally it calculates R^2 score, MSE, RMSE and MAEs
    
    :param NUMTREES: int: Specify the number of trees in the forest
    :param DEPTH: int: Determine the depth of each tree in the forest
    :param X: pd.DataFrame: Pass in the dataframe of features
    :param y: np.ndarray: Pass the target variable to the function

    :return: r^2 score, rmse, mse and mae
    :doc-author: Trelent
    """
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, shuffle=True)

    print(f'\nBuilding regression forest with {NUMTREES} trees each {DEPTH} deep\n')

    # Initialize the random forest
    rf = RandomForestRegressor(n_estimators=NUMTREES, max_depth=DEPTH, max_features='sqrt', bootstrap=True)

    # Train the model
    rf.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = rf.predict(X_test)

    # Calculate R^2 score
    r2 = r2_score(y_test, y_pred)

    # Calculate MSE
    mse = mean_squared_error(y_test, y_pred)

    # Calculate RMSE
    rmse = math.sqrt(mse)

    # Calculate MAE
    mae = mean_absolute_error(y_test, y_pred)

    return r2, rmse, mse, mae 


def growClassifier(NUMTREES: int, DEPTH: int, X: pd.DataFrame , y: np.ndarray):
    """
    The growClassifier function takes in the number of trees, depth, and data as input.
    It then splits the data into training and testing sets. It initializes a random forest classifier with 
    the given parameters (number of trees, depth). The model is trained on the training set and predictions are made on 
    the test set. Accuracy is measured using accuracy_score from sklearn's metrics module. Precision is measured using precision_score from sklearn's metrics module. Recall is measured using recall_score from sklearn's metrics module.
    
    :param NUMTREES: int: Set the number of trees in the random forest
    :param DEPTH: int: Set the maximum depth of each tree in the forest
    :param X: pd.DataFrame: Pass the dataframe of features to the function
    :param y: np.ndarray: Pass in the labels for the data
    :return: The following: f1, accuracy, precision, recall, conf_matrix
    :doc-author: Trelent
    """
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, shuffle=True)

    print(f'\nBuilding classification forest with {NUMTREES} trees each {DEPTH} deep\n')

    # Initialize the random forest
    rf = RandomForestClassifier(n_estimators=NUMTREES, max_depth=DEPTH, max_features='sqrt', bootstrap=True)

    # Train the model
    rf.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = rf.predict(X_test)

    # measure Accuracy 
    accuracy = accuracy_score(y_test, y_pred)
    # print('acc:\t', accuracy)

    # measure precision 
    precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    # print('prec:\t', precision)

    # measure recall 
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    # print('rec:\t', recall)


    # measure f1 
    f1 = f1_score(y_test, y_pred, average='weighted' , zero_division=0)
    # print('f1:\t', f1, '\n')


    # create confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    flatConfMatrix = conf_matrix.ravel()

    return f1, accuracy, precision, recall, conf_matrix



def remakeRegressionRuns(model: str, task: str, allDatasets: list, regDatasets: list, ESTNUM: int, DEPTH: int, MAX_RUNS: int, rawDataPath: str, aggDataPath: str ):
    """
    The regressionRuns function is used to run the Random Forest Regressor on a given dataset.
   
    :param model: str: Specify the type of model to be used ('RF', 'XGB', etc...)
    :param task: str: Specify the type of task being performed ('reg')
    :param allDatasets: list: a list with the abreviated names of all available datasets
    :param regDatasets: list: Specify which datasets to run the regression on
    :param ESTNUM: int: Set the number of trees in the forest
    :param DEPTH: int: Set the depth of each tree in the forest
    :param MAX_RUNS: int: Set the number of times we want to run the model with that number of trees and depth
    :param rawDataPath: str: Specify the path to the directory where you want your raw data saved
    :param aggDataPath: str: Specify the path to where the aggregated data will be stored
    :return: None
    :doc-author: Trelent
    """
    for dataset in allDatasets:
        if dataset in regDatasets: 

            X,y = parse.getRegData(dataset)

            runNumber = 1

            while (runNumber < MAX_RUNS + 1):
                print(f'\n{dataset} Run number:\t{runNumber}')


                # Set file name system for raw data
                saveRawDataHere = os.path.join(rawDataPath, dataset, f'_{ESTNUM}_{DEPTH}_{dataset}_{model}_{task}_')

                # add header to raw and agg file
                with open(saveRawDataHere, 'a') as raw_file:
                    if isEmpty(saveRawDataHere):
                        raw_file.write(f"numTrees\ttreeDepth\tr2\trmse\tmse\tmae\tbuildTime\n") 
                
                # Set file name system for agg data
                saveAggDataHere = os.path.join(aggDataPath, f'_{dataset}_{model}_{task}_')
                
                # add header to agg data file 
                with open(saveAggDataHere, 'a') as agg_file:
                    if isEmpty(saveAggDataHere):
                        agg_file.write(f"numTrees\ttreeDepth\tr2\trmse\tmse\tmae\tbuildTime\n")

                # run and time forest building
                start_time = time.time()
                r2, rmse, mse, mae = growRegressor(ESTNUM, DEPTH, X, y)
                finish_time = time.time()
                buildtime = finish_time - start_time

                # write data to file
                print(f'saving data in {saveRawDataHere}')
                with open(saveRawDataHere, 'a') as raw_file:
                    raw_file.write(f"{ESTNUM}\t{DEPTH}\t{r2}\t{rmse}\t{mse}\t{mae}\t{buildtime}\n")

                # increment counter    
                runNumber += 1



def critRuns(model: str, regDatasets: list, clsDatasets: list, MAX_RUNS: int, rawDataPath:str, aggDataPath: str ):

    allDatasets = regDatasets+clsDatasets

    for cr

    for dataset in allDatasets:
        if dataset in regDatasets: 
            task = 'reg'
            X,y = parse.getRegData(dataset)
        
        elif dataset in clsDatasets:
            task = 'cls'
            X,y = parse.getClsData(dataset)

            runNumber = 1

            while (runNumber < MAX_RUNS + 1):
                print(f'\n{dataset} Run number:\t{runNumber}')

                # Set file name system for raw data
                saveRawDataHere = os.path.join(rawDataPath, dataset, f'_{myCrit}_{depth}_{dataset}_{model}_{task}_')

                # add header to raw and agg file
                with open(saveRawDataHere, 'a') as raw_file:
                    if isEmpty(saveRawDataHere):
                        raw_file.write(f"numTrees\ttreeDepth\tr2\trmse\tmse\tmae\tbuildTime\n") 
                
                # Set file name system for agg data
                saveAggDataHere = os.path.join(aggDataPath, f'_{dataset}_{model}_{task}_')
                
                # add header to agg data file 
                with open(saveAggDataHere, 'a') as agg_file:
                    if isEmpty(saveAggDataHere):
                        agg_file.write(f"numTrees\ttreeDepth\tr2\trmse\tmse\tmae\tbuildTime\n")

                # run and time forest building
                start_time = time.time()
                r2, rmse, mse, mae = growRegressor(numEstimators, depth, X, y)
                finish_time = time.time()
                buildtime = finish_time - start_time

                # write data to file
                print(f'saving data in {saveRawDataHere}')
                with open(saveRawDataHere, 'a') as raw_file:
                    raw_file.write(f"{numEstimators}\t{depth}\t{r2}\t{rmse}\t{mse}\t{mae}\t{buildtime}\n")

                # # increment counter    
                runNumber += 1

def growCrit():
    print('hello')