import os
import pandas as pd, numpy as np

def aggMyData(subdirectory_path, output_path):
    for entry in os.listdir(subdirectory_path):
        entry_path = os.path.join(subdirectory_path, entry)

        if os.path.isdir(entry_path):
            # If it's a directory, ignore it
            print(f"agg is Processing directory: {entry} ...")
            aggMyData(entry_path, output_path)

        if entry == '.DS_Store':
            print("agg is Skipping .DS_Store file")
            continue

        elif os.path.isfile(entry_path):
            print(f"agg is Processing file: {entry} ...")
            numTrees, treeDepth, modelType, taskType, fromDataset = splitSingleDataFileName(entry)
            saveHere = os.path.join(output_path, f'_{fromDataset}_{modelType}_{taskType}_')
            
            if taskType == 'reg': 
                myAggData = aggRegFile(entry_path)
                with open(saveHere, 'a') as myOut:
                    myOut.write(f"{numTrees}\t{treeDepth}\t{myAggData['oob'].loc[0]}\t{myAggData['r2'].loc[0]}\t{myAggData['rmse'].loc[0]}\t{myAggData['mse'].loc[0]}\t{myAggData['mae'].loc[0]}\t{myAggData['buildtime'].loc[0]}\n")

            elif taskType =='cls':
                myAggData = aggClsFile(entry_path)
                with open(saveHere, 'a') as myOut:
                    myOut.write(f"{numTrees}\t{treeDepth}\t{myAggData['oob'].loc[0]}\t{myAggData['f1'].loc[0]}\t{myAggData['accuracy'].loc[0]}\t{myAggData['precision'].loc[0]}\t{myAggData['recall'].loc[0]}\t{myAggData['buildtime'].loc[0]}\n")   

          

def splitSingleDataFileName(entry_name): 
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
    data = pd.read_csv(entry_path, sep='\t', header=0)
    aggData = pd.DataFrame()

    aggData['oob'] = [data['oob'].mean()]
    aggData['r2'] = [data['r2'].mean()]
    aggData['rmse'] = [data['rmse'].mean()]
    aggData['mse'] = [data['mse'].mean()]
    aggData['mae'] = [data['mae'].mean()]
    aggData['buildtime'] = [data['buildtime'].mean()]
    return aggData



def aggClsFile(entry_path):
    data = pd.read_csv(entry_path, sep='\t', header=0)
    aggData = pd.DataFrame()

    aggData['oob'] = [data['oob'].mean()]
    aggData['f1'] = [data['f1'].mean()]
    aggData['accuracy'] = [data['accuracy'].mean()]
    aggData['precision'] = [data['precision'].mean()]
    aggData['recall'] = [data['recall'].mean()]
    aggData['buildtime'] = [data['buildtime'].mean()]
    return aggData


