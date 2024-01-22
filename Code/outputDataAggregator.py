import os
import pandas as pd


def aggMyData(subdirectory_path, output_path):
    """
    The aggMyData function takes a directory of data files and aggregates them into one file.
    
    :param subdirectory_path: Specify the path to the directory containing all of your initial data files
    :param output_path: Specify where the aggregated data will be saved
    :return: None
    :doc-author: Trelent
    """
    for entry in os.listdir(subdirectory_path):
        entry_path = os.path.join(subdirectory_path, entry)

        if os.path.isdir(entry_path):
            # If it's a directory, ignore it
            print(f"agg is processing directory: {entry} ...")
            aggMyData(entry_path, output_path)

        if entry == '.DS_Store':
            print("agg is skipping .DS_Store file")
            continue

        elif os.path.isfile(entry_path):
            print(f"\nagg is processing file: {entry} ...")
            numTrees, treeDepth, modelType, taskType, fromDataset = splitSingleDataFileName(entry)
            saveHere = os.path.join(output_path, f'_{fromDataset}_{modelType}_{taskType}_')
            
            if taskType == 'reg': 
                myAggData = aggRegFile(entry_path)
                with open(saveHere, 'a') as myOut:
                    myOut.write(f"{myAggData['numTrees'].loc[0]}\t{myAggData['treeDepth'].loc[0]}\t{myAggData['r2'].loc[0]}\t{myAggData['rmse'].loc[0]}\t{myAggData['mse'].loc[0]}\t{myAggData['mae'].loc[0]}\t{myAggData['buildTime'].loc[0]}\n")

            elif taskType =='cls':
                myAggData = aggClsFile(entry_path)
                with open(saveHere, 'a') as myOut:
                    myOut.write(f"{myAggData['numTrees'].loc[0]}\t{myAggData['treeDepth'].loc[0]}\t{myAggData['f1'].loc[0]}\t{myAggData['accuracy'].loc[0]}\t{myAggData['precision'].loc[0]}\t{myAggData['recall'].loc[0]}\t{myAggData['buildTime'].loc[0]}\n")   

          

def splitSingleDataFileName(entry_name): 
    """
    The splitSingleDataFileName function takes in a single file name and splits it into its constituent parts.
    The function returns the number of trees, tree depth, model type (e.g., RandomForestClassifier), task type (e.g., classification), 
    and fromDataset (the dataset used to train the model). The function is called by splitDataFileNames.
    
    :param entry_name: The name of the file that will be processed
    :return: numTrees, treeDepth, modelType, taskType, fromDataset
    :doc-author: Trelent
    """
    print('splitting file name...')

    # split file name by '_'
    fName = entry_name.split('_')

    # Assign last 4 values of the split, should be numtrees, treedepth, datasetname, modeltype
    attrName = fName[1:-1]
    numTrees = int(attrName[0])
    treeDepth = int(attrName[1])
    fromDataset = attrName[2]
    modelType = attrName[3]
    taskType = attrName[4]
    # print('my Fname\t',attrName,'\n')
    return numTrees, treeDepth, modelType, taskType, fromDataset



def aggRegFile(entry_path):
    """
    The aggRegFile function takes a single file path as an argument and returns a pandas DataFrame with the following columns:
        - numTrees: mean number of trees in the forests in the file
        - treeDepth: mean depth of each tree in the forests in the file
        - r2, rmse, mse, mae : mean regression metrics for these runs (see https://scikit-learn.org/stable/modules/model_evaluation.html#regression-metrics) 
    
    :param entry_path: Specify the path to the file that is being aggregated
    :return: The aggregated dataframe
    :doc-author: Trelent
    """
    print(f'aggregating {entry_path}...')

    data = pd.read_csv(entry_path, sep='\t', header=0)
    aggData = pd.DataFrame()
    
    aggData['numTrees'] = [data['numTrees'].mean()]
    aggData['treeDepth'] = [data['treeDepth'].mean()]
    aggData['r2'] = [data['r2'].mean()]
    aggData['rmse'] = [data['rmse'].mean()]
    aggData['mse'] = [data['mse'].mean()]
    aggData['mae'] = [data['mae'].mean()]
    aggData['buildTime'] = [data['buildTime'].mean()]
    return aggData



def aggClsFile(entry_path):
    """
    The aggClsFile function takes in a path to a file containing the results of
    a classification experiment and returns an aggregated dataframe with the mean
    values for each metric. 
    
    The output dataframe will have one row and 
    six columns: numTrees, treeDepth, f_one_score (f-measure), accuracy score, 
    precision score and recall score.
    
    :param entry_path: Specify the path of the file to be aggregated
    :return: The aggregated dataframe with the average values of each column
    :doc-author: Trelent
    """
    print(f'aggregating {entry_path}...')

    data = pd.read_csv(entry_path, sep='\t', header=0)
    aggData = pd.DataFrame()

    aggData['numTrees'] = [data['numTrees'].mean()]
    aggData['treeDepth'] = [data['treeDepth'].mean()]
    aggData['f1'] = [data['f1'].mean()]
    aggData['accuracy'] = [data['accuracy'].mean()]
    aggData['precision'] = [data['precision'].mean()]
    aggData['recall'] = [data['recall'].mean()]
    aggData['buildTime'] = [data['buildTime'].mean()]
    return aggData



def sortAggData(entry_path): 
    """
    The sortAggData function takes in a path to an aggregated data file and returns the sorted data.
    The function sorts the data by numTrees and treeDepth, ascending.
    
    :param entry_path: Specify the path of the file to be sorted
    :return: A sorted dataframe
    :doc-author: Trelent
    """
    print(f"Sorting {entry_path}...")    
    data = pd.read_csv(entry_path, sep='\t', header=0)

    sortedData = data.sort_values(by=['numTrees', 'treeDepth'], ascending=[True, True], ignore_index=True)
    return sortedData