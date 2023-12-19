import math, os
import pandas as pd, numpy as np, matplotlib.pyplot as plt

extDataNames =['California Housing', 'Italy Air Quality', 'Facebook Comment Volume', 'Abalone', 'Pima Native American Diabetes' , 'Wisconsin Breast Cancer Diagnostic', 'Portugal Wine Quality' , 'Human Activity Recognition', 'Adult Income']
extFolderNames = ['cali', 'air', 'fb' , 'aba', 'diabetes', 'cancer', 'wine', 'HAR', 'income']

risingMetric = ['r2', 'accuracy', 'precision', 'recall', 'f1']
fallingMetric = ['rmse', 'mse', 'mae', 'oob']

def graphsNTables(subdirectory_path, graphs_path, tables_path):
    for entry in os.listdir(subdirectory_path):
        entry_path = os.path.join(subdirectory_path, entry)

        if os.path.isdir(entry_path):
            # If it's a directory, ignore it
            print(f"Processing directory: {entry} ...")
            graphsNTables(entry_path, graphs_path, tables_path)

        if entry == '.DS_Store':
            print("Skipping .DS_Store file...")
            continue

        elif os.path.isfile(entry_path):
            print(f"Processing file: {entry} ...")
            fromDataset, modelType, taskType = splitAggDataFileName(entry)
            if taskType == 'reg': # regression task 
                myHeader = ['trees', 'depth', 'oob', 'r2', 'rmse', 'mse', 'mae']

            elif taskType == 'cls': # classification task 
                myHeader = ['trees', 'depth', 'oob', 'f1', 'accuracy', 'precision', 'recall']
            
            myAggData = sortAggData(entry_path, myHeader)
            topHowMany = 10
            for errorMetric in myHeader[2:]:
                myTableFrame = makeTableFrame(topHowMany, myAggData, errorMetric)
                # myGraph = makeGraph(myAggData, fromDataset , errorMetric)
                # storeGraph(myGraph, fromDataset, errorMetric, modelType, taskType, graphs_path)
                storeTable(myTableFrame, fromDataset, errorMetric, modelType, taskType, tables_path)





def splitAggDataFileName(entry_name):
    print(f"Splitting file name: {entry_name} ...")
    fName = entry_name.split('_')

    # Assign last 4 values of the split, should be numtrees, treedepth, datasetname, modeltype
    attrName = fName[1:-1]
    fromDataset = attrName[0]
    modelType = attrName[1]
    taskType = attrName[2]
    # # print('my Fname\t',attrName,'\n')
    return fromDataset, modelType, taskType



def sortAggData(entry_path, DATAHEADERLIST: list[str]): 
    print(f"Sorting aggregated data...")    
    data = pd.read_csv(entry_path, sep='\t', header=None)

    data.columns = DATAHEADERLIST
    sortedData = data.sort_values(by=['trees', 'depth'], ascending=[True, True], ignore_index=True)
    return sortedData



def makeGraph(myData: pd.DataFrame, fromDataset, errorMetric):
    print(f"Graphing {errorMetric} data...")    
    X, Y = myData['trees'], myData['depth']
   
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the 3D surface
    ax.plot_trisurf(X, Y, myData[errorMetric], cmap='inferno', edgecolor='k', linewidth=0.03)

    # Set labels
    ax.set_xlabel('Trees')
    ax.set_ylabel('Depth')
    ax.set_zlabel(errorMetric.upper())

    if fromDataset == 'cali':
        myTitle = extDataNames[0]
    elif fromDataset == 'air':
        myTitle = extDataNames[1]
    elif fromDataset == 'fb':
        myTitle = extDataNames[2]
    elif fromDataset == 'aba':
        myTitle = extDataNames[3]
    elif fromDataset == 'diabetes':
        myTitle = extDataNames[4]
    elif fromDataset == 'cancer':
        myTitle = extDataNames[5]
    elif fromDataset == 'wine':
        myTitle = extDataNames[6]
    elif fromDataset == 'HAR':
        myTitle = extDataNames[7]
    elif fromDataset == 'income':
        myTitle = extDataNames[8]
    ax.set_title(f'Aggregate {errorMetric.upper()} Scores\n{myTitle} Dataset')

    ax.view_init(elev=27, azim=-142)

    plt.show()
    return plt



def storeGraph(myFig: plt.Figure, fromDataset, errorMetric, modelType, taskType, graphs_path):
    print(f"Storing the graph in {graphs_path}")
    gPath = os.path.join(graphs_path, fromDataset)

    # myFig.tight_layout()
    myFig.savefig(os.path.join(gPath, f'_{fromDataset}_{errorMetric}_{modelType}_{taskType}_'), dpi=600)



def makeTableFrame(topHowMany: int, myAggData: pd.DataFrame, errorMetric):
    print(f"Putting the top {topHowMany} {errorMetric.upper()} into a frame...")

    topErrs = pd.DataFrame(columns=myAggData.columns)        
    if errorMetric in risingMetric:
        for num in range(topHowMany):
            max_point = myAggData.loc[myAggData[errorMetric].idxmax()]
            # Transpose max_point Series to create a row DataFrame
            max_point_df = pd.DataFrame([max_point.values], columns=myAggData.columns)

            max_point_df = round(max_point_df , 4)

            # Concatenate max_point_df to topErrs
            topErrs = pd.concat([topErrs, max_point_df], axis=0, ignore_index=True)

            # topErrs = pd.concat([topErrs, max_point], axis=0, ignore_index=True)
            myAggData = myAggData.drop(myAggData[myAggData[errorMetric] == max_point[errorMetric]].index)
    
    elif errorMetric in fallingMetric:
        for num in range(topHowMany):
            max_point = myAggData.loc[myAggData[errorMetric].idxmin()]
            # Transpose max_point Series to create a row DataFrame
            max_point_df = pd.DataFrame([max_point.values], columns=myAggData.columns)

            max_point_df = round(max_point_df , 4)

            # Concatenate max_point_df to topErrs
            topErrs = pd.concat([topErrs, max_point_df], axis=0, ignore_index=True)

            # topErrs = pd.concat([topErrs, max_point], axis=0, ignore_index=True)
            myAggData = myAggData.drop(myAggData[myAggData[errorMetric] == max_point[errorMetric]].index)
    
    return topErrs


        
def storeTable(myTableFrame: pd.DataFrame, fromDataset, errorMetric, modelType, taskType, tables_path):
    # Plot the DataFrame as a table
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axis('off')  # Hide axes

    table = ax.table(cellText=myTableFrame.values, colLabels=myTableFrame.columns, cellLoc='center', loc='center')

    if fromDataset == 'cali':
        myTitle = extDataNames[0]
    elif fromDataset == 'air':
        myTitle = extDataNames[1]
    elif fromDataset == 'fb':
        myTitle = extDataNames[2]
    elif fromDataset == 'aba':
        myTitle = extDataNames[3]
    elif fromDataset == 'diabetes':
        myTitle = extDataNames[4]
    elif fromDataset == 'cancer':
        myTitle = extDataNames[5]
    elif fromDataset == 'wine':
        myTitle = extDataNames[6]
    elif fromDataset == 'HAR':
        myTitle = extDataNames[7]
    elif fromDataset == 'income':
        myTitle = extDataNames[8]
    ax.set_title(f'Best {errorMetric.upper()} Scores, {myTitle} Dataset', y=.8)

    tPath = os.path.join(tables_path, fromDataset)

    # plt.show()


    plt.savefig(os.path.join(tPath, f'_{fromDataset}_{errorMetric}_{modelType}_{taskType}_'), dpi=600)




