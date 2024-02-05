from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import os, time, math
import inputDataParser as parse
import pandas as pd, numpy as np
import xgboost as xgb

timeToBuild = time.time()
iteration = 0

def cls_eval(preds, dtrain):
    # Record the start time at the beginning of the evaluation
    start_time = time.time()

    labels = dtrain.get_label()  # Get the true labels from the training dataset
    preds_classes = np.asarray(preds)
    f1 = f1_score(labels, preds_classes, average='weighted' , zero_division=0)
    accuracy = accuracy_score(labels, preds_classes)
    precision = precision_score(labels, preds_classes, average="weighted", zero_division=0)
    recall = recall_score(labels, preds_classes, average='weighted', zero_division=0)

    # Record the finish time at the end of the evaluation

    global iteration, timeToBuild
    iteration +=1
    
    buildtime = start_time - timeToBuild

    if iteration % 2 == 0:
        timeToBuild = start_time

    # Return the metrics as a list of tuples
    return [('f1', f1), ('accuracy', accuracy), ('precision', precision), ('recall', recall), ('buildtime', buildtime)]


def isEmpty(file_path):
    return os.stat(file_path).st_size == 0


def classificationRuns(model: str, task: str, allDatasets: list, clsDatasets: list, ESTNUM: int, DEPTH: int, MAX_RUNS: int, rawDataPath: str, aggDataPath: str ):
    for dataset in allDatasets:
        if dataset in clsDatasets: 

            X,y = parse.getClsData(dataset)
            for depth in range(DEPTH+1):

                runNumber = 1
                depth +=1
                while (runNumber < MAX_RUNS + 1):
                    print(f'\n{dataset} Run number:\t{runNumber}')

                    # Set file name system for raw data
                    saveRawDataHere = os.path.join(rawDataPath, dataset, f'_{ESTNUM}_{depth}_{dataset}_{model}_{task}_')

                    # add header to raw file
                    with open(saveRawDataHere, 'a') as raw_file:
                        if isEmpty(saveRawDataHere):
                            raw_file.write(f"numTrees\ttreeDepth\tmlogloss_train\tmlogloss_test\tf1_train\tf1_test\taccuracy_train\taccuracy_test\tprecision_train\tprecision_test\trecall_train\trecall_test\tbuildTime_train\tbuildTime_test\n")
                    
                    # Set file name system for agg data
                    saveAggDataHere = os.path.join(aggDataPath, f'_{dataset}_{model}_{task}_')
                    
                    # add header to agg data file 
                    with open(saveAggDataHere, 'a') as agg_file:
                        if isEmpty(saveAggDataHere):
                            agg_file.write(f"numTrees\ttreeDepth\tmlogloss_train\tmlogloss_test\tf1_train\tf1_test\taccuracy_train\taccuracy_test\tprecision_train\tprecision_test\trecall_train\trecall_test\tbuildTime_train\tbuildTime_test\n")

                    # run forest building
                    clsResults = growClassifier(ESTNUM, depth, X, y)

                    # write data to file
                    print(f'saving data in {saveRawDataHere}')
                    clsResults.to_csv(saveRawDataHere, mode='a', index=False, header=False, sep='\t')
                    
                    # increment counter    
                    runNumber += 1

def growClassifier(NUMTREES: int, DEPTH: int, X: pd.DataFrame , y: np.ndarray):
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, shuffle=True)
    # print(f"np.unique(y_train):\t{np.unique(y_train)}")
    # print(f"np.unique(y_test):\t{np.unique(y_test)}")

    print(f'\nBuilding XGBoosted classification forest with {NUMTREES} trees each {DEPTH} deep\n')

    # Initialize the random forest
    xgCls = xgb.XGBClassifier(n_estimators=NUMTREES, max_depth=DEPTH, tree_method='hist', objective='multi:softmax', num_class=len(np.unique(y)))

    # Train the model
    global timeToBuild
    timeToBuild = time.time()
    xgCls.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)], eval_metric=cls_eval, verbose=True)

    # # Make predictions on the test set
    # # y_pred = rf.predict(X_test)

    eval_results = xgCls.evals_result()

    f1_train = eval_results['validation_0']['f1']
    accuracy_train = eval_results['validation_0']['accuracy']
    precision_train = eval_results['validation_0']['precision']
    recall_train = eval_results['validation_0']['recall']
    mlogloss_train = eval_results['validation_0']['mlogloss']
    buildTime_train = eval_results['validation_0']['buildtime']

    f1_test = eval_results['validation_1']['f1']
    accuracy_test = eval_results['validation_1']['accuracy']
    precision_test = eval_results['validation_1']['precision']
    recall_test = eval_results['validation_1']['recall']
    mlogloss_test = eval_results['validation_1']['mlogloss']
    buildTime_test = eval_results['validation_1']['buildtime']

    xgClsResults = pd.DataFrame()

    numTrees, treeDepth = [], [] 
    for x in range(NUMTREES):
        numTrees.append(x+1) 
        treeDepth.append(DEPTH)
    
    xgClsResults['numTrees'] = numTrees
    xgClsResults['treeDepth'] = treeDepth

    xgClsResults['mlogloss_train'] = mlogloss_train
    xgClsResults['mlogloss_test'] = mlogloss_test

    xgClsResults['f1_train'] = f1_train
    xgClsResults['f1_test'] = f1_test

    xgClsResults['accuracy_train'] = accuracy_train
    xgClsResults['accuracy_test'] = accuracy_test

    xgClsResults['precision_train'] = precision_train
    xgClsResults['precision_test'] = precision_test

    xgClsResults['recall_train'] = recall_train
    xgClsResults['recall_test'] = recall_test

    xgClsResults['buildTime_train'] = buildTime_train
    xgClsResults['buildTime_test'] = buildTime_test


    # print(eval_results.items())

    return xgClsResults


def reg_eval(preds, dtrain):
    # Record the start time at the beginning of the evaluation
    start_time = time.time()

    labels = dtrain.get_label()  # Get the true labels from the training dataset
    r2 = r2_score(labels, preds)  # Calculate R-squared
    mse = mean_squared_error(labels, preds)  # Calculate mean squared error
    mae = mean_absolute_error(labels, preds)  # Calculate mean absolute error

    global iteration, timeToBuild
    iteration +=1
    
    buildtime = start_time - timeToBuild

    if iteration % 2 == 0:
        timeToBuild = start_time
        
    # Return the metrics as a list of tuples
    return [('r2', r2), ('mse', mse), ('mae', mae), ('buildtime', buildtime)]

def growRegressor(NUMTREES: int, DEPTH: int, X: pd.DataFrame , y: np.ndarray):
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, shuffle=True)

    print(f'\nBuilding XGBoosted regression forest with {NUMTREES} trees each {DEPTH} deep\n')

    # Initialize the random forest
    xgReg = xgb.XGBRegressor(n_estimators=NUMTREES, max_depth=DEPTH, tree_method='hist', objective='reg:squarederror')

    # Time the model
    global timeToBuild
    timeToBuild = time.time()

    # Train the model
    xgReg.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)], eval_metric=reg_eval, verbose=True)
    
    # Make predictions on the test set
    # y_pred = rf.predict(X_test)

    eval_results = xgReg.evals_result()
    
    xgRegResults = pd.DataFrame()

    numTrees, treeDepth = [], [] 
    for x in range(NUMTREES):
        numTrees.append(x+1) 
        treeDepth.append(DEPTH)
    
    xgRegResults['numTrees'] = numTrees
    xgRegResults['treeDepth'] = treeDepth

    r2_train = eval_results['validation_0']['r2']
    rmse_train = eval_results['validation_0']['rmse']
    mse_train = eval_results['validation_0']['mse']
    mae_train = eval_results['validation_0']['mae']
    buildTime_train = eval_results['validation_0']['buildtime']

    r2_test = eval_results['validation_1']['r2']
    rmse_test = eval_results['validation_1']['rmse']
    mse_test = eval_results['validation_1']['mse']
    mae_test = eval_results['validation_1']['mae']
    buildTime_test = eval_results['validation_1']['buildtime']



    xgRegResults['r2_train'] = r2_train
    xgRegResults['r2_test'] = r2_test

    xgRegResults['rmse_train'] = rmse_train
    xgRegResults['rmse_test'] = rmse_test

    xgRegResults['mse_train'] = mse_train
    xgRegResults['mse_test'] = mse_test

    xgRegResults['mae_train'] = mae_train
    xgRegResults['mae_test'] = mae_test


    xgRegResults['buildTime_train'] = buildTime_train
    xgRegResults['buildTime_test'] = buildTime_test

    # print(xgRegResults)

    return xgRegResults


def regressionRuns(model: str, task: str, allDatasets: list, clsDatasets: list, ESTNUM: int, DEPTH: int, MAX_RUNS: int, rawDataPath: str, aggDataPath: str ):
    for dataset in allDatasets:
        if dataset in clsDatasets: 

            X,y = parse.getRegData(dataset)
            for depth in range(DEPTH+1):

                runNumber = 1
                depth +=1
                while (runNumber < MAX_RUNS + 1):
                    print(f'\n{dataset} Run number:\t{runNumber}')

                    # Set file name system for raw data
                    saveRawDataHere = os.path.join(rawDataPath, dataset, f'_{ESTNUM}_{depth}_{dataset}_{model}_{task}_')

                    # add header to raw file
                    with open(saveRawDataHere, 'a') as raw_file:
                        if isEmpty(saveRawDataHere):
                            raw_file.write(f"numTrees\ttreeDepth\tr2_train\tr2_test\trmse_train\trmse_test\tmse_train\tmse_test\tmae_train\tmae_test\tbuildTime_train\tbuildTime_test\n") 

                    
                    # Set file name system for agg data
                    saveAggDataHere = os.path.join(aggDataPath, f'_{dataset}_{model}_{task}_')
                    
                    # add header to agg data file 
                    with open(saveAggDataHere, 'a') as agg_file:
                        if isEmpty(saveAggDataHere):
                            agg_file.write(f"numTrees\ttreeDepth\tr2_train\tr2_test\trmse_train\trmse_test\tmse_train\tmse_test\tmae_train\tmae_test\tbuildTime_train\tbuildTime_test\n") 

                    # run forest building
                    regResults = growRegressor(ESTNUM, depth, X, y)

                    # write data to file
                    print(f'saving data in {saveRawDataHere}')
                    regResults.to_csv(saveRawDataHere, mode='a', index=False, header=False, sep='\t')
                    
                    # increment counter    
                    runNumber += 1