import math, os, time, random
import pandas as pd, numpy as np
import inputDataParser as parse
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, log_loss


def isEmpty(file_path):
    """
    The isEmpty function takes in a file path and returns True if the file is empty, False otherwise.
        Args:
            file_path (str): The path to the desired textfile.
    
    :param file_path: Specify the file path of the file to be checked
    :return: True if the file is empty
    :doc-author: Trelent
    """
    return os.stat(file_path).st_size == 0

def regressionRuns(model: str, task: str, allDatasets: list, regDatasets: list, ESTNUM: int, DEPTH: int, MAX_RUNS: int, rawDataPath: str, aggDataPath: str ):

    for dataset in allDatasets:
        if dataset in regDatasets: 

            X,y = parse.getRegData(dataset)
            for depth in range(DEPTH+1):

                runNumber = 1
                depth +=1
                while (runNumber < MAX_RUNS + 1):
                    print(f'\nRun number:\t{runNumber}')

                    # run forest building
                    regResults = growRegressor(ESTNUM, depth, X, y)

                    column_names = regResults.columns.tolist()

                    # Join column names with tab separators
                    header = '\t'.join(column_names)

                    # Set file name system for raw data
                    saveRawDataHere = os.path.join(rawDataPath, dataset, f'_{ESTNUM}_{depth}_{dataset}_{model}_{task}_')

                    # add header to raw and agg file
                    with open(saveRawDataHere, 'a') as raw_file:
                        if isEmpty(saveRawDataHere):
                            raw_file.write(f"{header}\n") 
                    
                    # Set file name system for agg data
                    saveAggDataHere = os.path.join(aggDataPath, f'_{dataset}_{model}_{task}_')
                    
                    # add header to agg data file 
                    with open(saveAggDataHere, 'a') as agg_file:
                        if isEmpty(saveAggDataHere):
                            agg_file.write(f"{header}\n")

                    
                    # write data to file
                    print(f'saving data in {saveRawDataHere}')
                    regResults.to_csv(saveRawDataHere, mode='a', index=False, header=False, sep='\t')
                    # increment counter    
                    runNumber += 1

def classificationRuns(model: str, task: str, allDatasets: list, clsDatasets: list, ESTNUM: int, startDEPTH: int, endDEPTH: int, MAX_RUNS: int, rawDataPath: str, aggDataPath: str ):

    for dataset in allDatasets:
        if dataset in clsDatasets: 

            # Get the data 
            X,y = parse.getClsData(dataset)
            while startDEPTH < endDEPTH: 

                runNumber = 1
                while (runNumber < MAX_RUNS + 1):
                    print(f'\nRun number:\t{runNumber}')

                    # run forest building
                    clsResults = growClassifier(ESTNUM, startDEPTH, X, y)

                    column_names = clsResults.columns.tolist()

                    # Join column names with tab separators
                    header = '\t'.join(column_names)

                    # Set file name system for raw data
                    saveRawDataHere = os.path.join(rawDataPath, dataset, f'_{ESTNUM}_{startDEPTH}_{dataset}_{model}_{task}_')

                    # add header to raw and agg file
                    with open(saveRawDataHere, 'a') as raw_file:
                        if isEmpty(saveRawDataHere):
                            raw_file.write(f"{header}\n") 
                    
                    # Set file name system for agg data
                    saveAggDataHere = os.path.join(aggDataPath, f'_{dataset}_{model}_{task}_')
                    
                    # add header to agg data file 
                    with open(saveAggDataHere, 'a') as agg_file:
                        if isEmpty(saveAggDataHere):
                            agg_file.write(f"{header}\n")

                    
                    # write data to file
                    print(f'saving data in {saveRawDataHere}')
                    clsResults.to_csv(saveRawDataHere, mode='a', index=False, header=False, sep='\t')
                    # increment counter    
                    runNumber += 1
                startDEPTH +=1


def growRegressor(NUMTREES: int, DEPTH: int, X: pd.DataFrame , y: np.ndarray):

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, shuffle=True)

    print(f'\nBuilding regression forest with {NUMTREES} trees each {DEPTH} deep\n')

    base_estimator = DecisionTreeClassifier(max_depth=DEPTH)

    start_time = time.time()

    ada = AdaBoostClassifier(estimator=base_estimator, n_estimators=NUMTREES)
    ada.fit(X_train, y_train)

    elapsed_time = time.time() - start_time

    # Use staged_predict to get staged predictions
    staged_test_predictions = ada.staged_predict(X_test)
    
    r2s, rmses, mses, maes, buildtime = [], [], [], [], []
    # Iterate over staged predictions and evaluate performance at each stage
    for i, y_pred in enumerate(staged_test_predictions, start=1):
        r2s.append(r2_score(y_test, y_pred))
        mses.append(mean_squared_error(y_test, y_pred))
        rmses.append(math.sqrt(mean_squared_error(y_test, y_pred)))
        maes.append(mean_absolute_error(y_test, y_pred))

    adaRegResults = pd.DataFrame()

    numTrees, treeDepth = [], [] 
    for x in range(NUMTREES):
        numTrees.append(x+1) 
        treeDepth.append(DEPTH)
        buildtime.append(elapsed_time)
    
    adaRegResults['numTrees'] = numTrees
    adaRegResults['treeDepth'] = treeDepth
    adaRegResults['r2'] = r2s
    adaRegResults['rmse'] = rmses
    adaRegResults['mse'] = mses
    adaRegResults['mae'] = maes
    adaRegResults['buildTime'] = buildtime

    
    # print(adaClsResults)
    return adaRegResults



def growClassifier(NUMTREES: int, DEPTH: int, X: pd.DataFrame , y: np.ndarray):
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, shuffle=True)

    print(f'\nBuilding classification forest with {NUMTREES} trees each {DEPTH} deep\n')

    # Initialize the week classifier 
    base_estimator = DecisionTreeClassifier(max_depth=DEPTH)

    start_time = time.time()

    ada = AdaBoostClassifier(estimator=base_estimator, n_estimators=NUMTREES)
    ada.fit(X_train, y_train)

    elapsed_time = time.time() - start_time

    # Use staged_predict to get staged predictions
    staged_test_predictions = ada.staged_predict(X_test)
    

    f1_test, accuracy_test, precision_test, recall_test, buildtime_test = [], [], [], [], []
    # Iterate over staged predictions and evaluate performance at each stage
    for i, y_pred in enumerate(staged_test_predictions, start=1):
        # print(f'i:{i}\ny_pred:{y_pred}')
        accuracy = accuracy_score(y_test, y_pred)
        accuracy_test.append(accuracy)
        
        precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        precision_test.append(precision)

        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        recall_test.append(recall)

        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        f1_test.append(f1)

        # print(f'accuracy:{accuracy}')

    adaClsResults = pd.DataFrame()

    numTrees, treeDepth = [], [] 
    for x in range(1, NUMTREES+1, 1):
        # print(i, x)
        numTrees.append(x) 
        treeDepth.append(DEPTH)
        buildtime_test.append(elapsed_time)

    if (i>=40): 
        print(f'\n\nOnly boosted {i} times. Did not have  {NUMTREES} stages. Adding 0 for remaining stages \n\n')
        while (i < NUMTREES):
            accuracy_test.append(0)
            precision_test.append(0)
            recall_test.append(0)
            f1_test.append(0)
            buildtime_test[i] = 0
            i += 1
        
        # print(f'stages:{i}\nlen(NumTrees):{len(numTrees)}\nlen(f1):{len(f1_test)}\nlen(accuracy_test):{len(accuracy_test)}\nlen(precision_test):{len(precision_test)}\nlen(recall_test):{len(recall_test)}\nlen(buildtime_test):{len(buildtime_test)}')

        adaClsResults['numTrees'] = numTrees
        adaClsResults['treeDepth'] = treeDepth
        adaClsResults['f1'] = f1_test
        adaClsResults['accuracy'] = accuracy_test
        adaClsResults['precision'] = precision_test
        adaClsResults['recall'] = recall_test
        adaClsResults['buildTime'] = buildtime_test
        return adaClsResults

    elif(i<40): 
        # print(f'stages:{i}\nlen(NumTrees):{len(numTrees)}\nlen(f1):{len(f1_test)}\nlen(accuracy_test):{len(accuracy_test)}\nlen(precision_test):{len(precision_test)}\nlen(recall_test):{len(recall_test)}\nlen(buildtime_test):{len(buildtime_test)}')
        # print(f'f1:{f1}\t:{f1_test}')
        print(f'\n\nfailed. Only boosted {i} times. Did not have  {NUMTREES} stages. running again \n\n')
        growClassifier(NUMTREES, DEPTH, X, y)

    return adaClsResults
    

    # # print(f'should never get here if len(numTrees) dne (len(treeDepth) and len(f1_test) and len(accuracy_test) and len(precision_test) and len(recall_test) and len(buildtime_test))')
        
    # # print(adaClsResults)
