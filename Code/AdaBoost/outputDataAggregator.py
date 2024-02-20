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
            print(f"\naggregator is processing directory: {entry} ...")
            aggMyData(entry_path, output_path)

        if entry == '.DS_Store':
            print("\naggregator is skipping .DS_Store file")
            continue

        elif os.path.isfile(entry_path):
            print(f"\naggregator is processing file: {entry} ...")
            numTrees, treeDepth, modelType, taskType, fromDataset = splitSingleDataFileName(entry)
            saveHere = os.path.join(output_path, f'_{fromDataset}_{modelType}_{taskType}_')
            
            myAggData = aggFile(entry_path)
            myAggData.to_csv(saveHere, mode='a', index=False, header=False, sep='\t')

          

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


def aggFile(entry_path):
   
    print(f'aggregating {entry_path}...')

    data = pd.read_csv(entry_path, sep='\t', header=0)
    aggData = pd.DataFrame(columns=data.columns)

    unique_values = data['numTrees'].unique()

    for nT in unique_values:
        filtered_rows = data[data['numTrees'] == nT]
        meanValues = filtered_rows.mean()

        # Append a new row to the DataFrame
        new_row = {'numTrees': nT, **meanValues.to_dict()}

        new_rows_df = pd.DataFrame([new_row])  # Wrap the new row in a list

        aggData = pd.concat([aggData, new_rows_df], ignore_index=True)
    
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