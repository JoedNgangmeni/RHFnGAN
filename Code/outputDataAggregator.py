import os
import pandas as pd, numpy as np

def aggMyData(subdirectory_path, output_path):
    for entry in os.listdir(subdirectory_path):
        entry_path = os.path.join(subdirectory_path, entry)

        if os.path.isdir(entry_path):
            # If it's a directory, ignore it
            print(f"Processing directory: {entry} ...")
            aggMyData(entry_path, output_path)
            # continue

        if entry == '.DS_Store':
            # print("Skipping .DS_Store file")
            continue

        elif os.path.isfile(entry_path):
            print(f"Processing file: {entry} ...")
            numTrees, treeDepth, whichModel, whichTask, whichData = parseFileName(entry_path)
            myAggData = processFile(entry_path)
            
            saveHere = os.path.join(output_path, f'_agg_{whichData}_{whichModel}_{whichTask}_')
            
            if whichTask == 'reg': 
                with open(saveHere, 'a') as myOut:
                    myOut.write(f"{numTrees}\t{treeDepth}\t{myAggData['oob'].loc[0]}\t{myAggData['r2'].loc[0]}\t{myAggData['rmse'].loc[0]}\t{myAggData['mse'].loc[0]}\t{myAggData['mae'].loc[0]}\n")

            elif whichTask =='cls':
                with open(saveHere, 'a') as myOut:
                    myOut.write(f"{numTrees}\t{treeDepth}\t{myAggData['oob'].loc[0]}\t{myAggData['f1'].loc[0]}\t{myAggData['accuracy'].loc[0]}\t{myAggData['precision'].loc[0]}\t{myAggData['recall'].loc[0]}\n")        

          

def parseFileName(entry_path): 
    # split file name by '_'
    fName = entry_path.split('_')

    # Assign last 4 values of the split, should be numtrees, treedepth, datasetname, modeltype
    attrName = fName[2:-1]
    numTrees = attrName[0]
    treeDepth = attrName[1]
    whichData = attrName[2]
    whichModel = attrName[3]
    whichTask = attrName[4]
    print('my Fname\t',attrName,'\n')

    return numTrees, treeDepth, whichModel, whichTask, whichData



def processFile(entry_path):
    data = pd.read_csv(entry_path, sep='\t')
    aggData = pd.DataFrame()

    if data.shape[1] == 5: # regression data
        aggData['oob'] = [data['oob'].mean()]
        aggData['r2'] = [data['r2'].mean()]
        aggData['rmse'] = [data['rmse'].mean()]
        aggData['mse'] = [data['mse'].mean()]
        aggData['mae'] = [data['mae'].mean()]
        
    else: # Classification data
        aggData['oob'] = [data['oob'].mean()]
        aggData['f1'] = [data['f1'].mean()]
        aggData['accuracy'] = [data['accuracy'].mean()]
        aggData['precision'] = [data['precision'].mean()]
        aggData['recall'] = [data['recall'].mean()]

    return aggData



