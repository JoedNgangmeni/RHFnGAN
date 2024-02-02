import sys, os
sys.path.append("..")

import pandas as pd, matplotlib.pyplot as plt
import outputDataAggregator as agg
import myStructure as my
import globalStuff as glbl

def graphsNTables(subdirectory_path:str, graphs_path:str, tables_path:str, topNUM:int):
    """
    The graphsNTables function takes in a subdirectory path, the 'graphs' directory path, and 'tables' directory path.
    It then iterates through the files in the subdirectory and creates a table and graph for each error metric. 
    It stores these tables and graphs into their respective paths.
    
    :param subdirectory_path: Specify the path to the directory where all of our data is stored
    :param graphs_path: Specify the directory where the graphs will be stored
    :param tables_path: Specify the directory where the tables will be stored
    :return: None 
    :doc-author: Trelent
    """
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
            print(f"\nProcessing file: {entry} ...")
            fromDataset, modelType, taskType = splitAggDataFileName(entry)
            if taskType == 'reg': # regression task 
                myHeader = ['numTrees', 'treeDepth', 'r2_train', 'r2_test',
                             'rmse_train', 'rmse_test', 'mse_train', 'mse_test',
                               'mae_train', 'mae_test', 'buildTime_train', 'buildTime_test']

            elif taskType == 'cls': # classification task 
                myHeader = ['numTrees', 'treeDepth','mlogloss_train', 'mlogloss_test',
                             'f1_train', 'f1_test', 'accuracy_train', 'accuracy_test',
                               'precision_train', 'precision_test', 'recall_train',
                                 'recall_test', 'buildTime_train', 'buildTime_test']
            
            myAggData = agg.sortAggData(entry_path)
            for errorMetric in myHeader[2:]:
                bestTableFrame = makeTableFrame(myAggData, errorMetric, topNUM)
                storeTable(bestTableFrame, fromDataset, errorMetric, modelType, taskType, tables_path)

                my3DGraph = make3DGraph(myAggData, fromDataset , errorMetric)
                storeGraph(my3DGraph, fromDataset, errorMetric, modelType, taskType, '3D', graphs_path)

                if 'train' in errorMetric:
                    my2DGraph = make2DGraph(myAggData , errorMetric)
                    storeGraph(my2DGraph, fromDataset, errorMetric, modelType, taskType,'2D', graphs_path)



def splitAggDataFileName(entry_name):
    """
    The splitAggDataFileName function takes in a string, which is the name of an aggregated data file.
    It splits the string into its component parts and returns them as a tuple. 
    
    The first element of this tuple is the dataset from which this aggregated data was generated (e.g., 'mnist', 'cifar10', etc.). 
    The second element is the type of model used to generate these results (e.g., 'mlp' or 'cnn'). 
    The third element is the task type that was performed on these datasets (i.e., either classification or regression). 
    
    :param entry_name: Get the name of the file
    :return: The dataset name, model type and task type
    :doc-author: Trelent
    """
    print(f"Splitting {entry_name} ...")
    fName = entry_name.split('_')

    # Assign last 4 values of the split, should be numtrees, treedepth, datasetname, modeltype
    attrName = fName[1:-1]
    fromDataset = attrName[0]
    modelType = attrName[1]
    taskType = attrName[2]
    # # print('my Fname\t',attrName,'\n')
    return fromDataset, modelType, taskType



def make3DGraph(myData: pd.DataFrame, fromDataset, errorMetric):
    """
    The makeGraph function takes in a dataframe, the name of the dataset it came from, and an error metric.
    It then plots a 3D surface graph with trees on the x-axis, depth on the y-axis, and error metric values on z-axis. 
    It also plots two 2D scatterplots showing how tree count affects build time (blue) and how 
    tree depth affects build time (red). The function returns a matplotlib object (the graph).
    
    :param myData: pd.DataFrame: Pass in the dataframe that we want to graph
    :param fromDataset: Determine which dataset the data is coming from
    :param errorMetric: Determine which error metric to graph
    :return: A plot object (the graph) that we want to save 
    :doc-author: Trelent
    """
    print(f"Making a 3D Graphing of {errorMetric} data...")  

    myTitle = glbl.setTitle(fromDataset, glbl.extDataNames) 
    
    X, Y = myData['numTrees'], myData['treeDepth']
   
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the 3D surface
    ax.plot_trisurf(X, Y, myData[errorMetric], cmap='inferno', edgecolor='k', linewidth=0.03)

    # Set labels
    ax.set_xlabel('Trees')
    ax.set_ylabel('Depth')
    ax.set_zlabel(errorMetric.upper())
    ax.view_init(elev=27, azim=-142)
    ax.set_title(f'Aggregate {errorMetric.upper()} Scores\n{myTitle} Dataset')

    # plt.show()
    return plt

def make2DGraph(myData: pd.DataFrame, errorMetric):
    em1Split = errorMetric.split('_')
    errorMetric2 = em1Split[0]+ '_test'
    em2Split = errorMetric2.split('_')


    print(f"Making a 2D Graph of {errorMetric.upper()} and {errorMetric2.upper()} data...")  
    
    X, Y = myData['numTrees'], myData['treeDepth']
   
    fig, ax = plt.subplots(1, 2, figsize=(10, 4), tight_layout=True)

    # Scatter plot for the first subplot (left side)
    ax[0].scatter(X, myData[errorMetric], color='blue', label=errorMetric)
    ax[0].scatter(X, myData[errorMetric2], color='green', label=errorMetric2)
    ax[0].set_title(f'{em1Split[0].upper()} {em1Split[1]} and {em2Split[1]} vs. {X.name}')
    ax[0].set_xlabel(f'{X.name}')
    ax[0].set_ylabel('Error Metrics')
    ax[0].legend()

    # Scatter plot for the second subplot (right side)
    ax[1].scatter(Y, myData[errorMetric], color='red', label=errorMetric)
    ax[1].scatter(Y, myData[errorMetric2], color='orange', label=errorMetric2)
    ax[1].set_title(f'{em1Split[0].upper()} {em1Split[1]} and {em2Split[1]} vs. {Y.name}')
    ax[1].set_xlabel(f'{Y.name}')
    ax[1].set_ylabel('Error Metrics')
    ax[1].legend()

    # Returning the plot object
    return plt



def storeGraph(myFig: plt.Figure, fromDataset, errorMetric, modelType, taskType, graphtype:str, graphs_path):
    """
    The storeGraph function takes in a figure object, the dataset it was generated from,
    the error metric used to generate it (e.g. RMSE), the model type (e.g. RF), the task type
    (e.g., regression or classification). 
    
    It then stores that graph in a folder named after its dataset.
    
    :param myFig: plt.Figure: Pass the figure object (the graph object) to the function
    :param fromDataset: Determine the dataset it was generated from
    :param errorMetric: Determine the type of error metric used
    :param modelType: Determine the type of model that was used to generate the graph
    :param taskType: Distinguish between the different types of tasks
    :param graphs_path: Store the graphs in a specific folder
    :return: None
    :doc-author: Trelent
    """
    print(f"Storing the graph in {graphs_path}/{fromDataset}")
    gPath = os.path.join(graphs_path, fromDataset)

    # myFig.tight_layout()
    myFig.savefig(os.path.join(gPath, f'_{graphtype}_{fromDataset}_{errorMetric}_{modelType}_{taskType}_'), dpi=600)



def makeTableFrame(myAggData: pd.DataFrame, errorMetric, topNUM:int):
    """
    The makeTableFrame function takes in the following parameters:
        myAggData - The dataframe that contains all of the aggregated error metrics.
        errorMetric - A string the name of the an error metric (e.g. mse, mae, rmse, r2).
    
    :param myAggData: pd.DataFrame: Pass in the aggregated data frame from which you will select rows
    :param errorMetric: Determine which error metric to use for the analysis
    :return: A dataframe with the topNUM rows of myAggData
    :doc-author: Trelent
    """
    print(f"Putting the top {topNUM} {errorMetric.upper()} into a frame...")

    topErrs = pd.DataFrame(columns=myAggData.columns)   

    if errorMetric in my.risingMetric:
        for num in range(topNUM):
            max_point = myAggData.loc[myAggData[errorMetric].idxmax()]

            # Convert the values to numeric
            numeric_values = pd.to_numeric(max_point, errors='coerce')

            # Transpose max_point Series to create a row DataFrame
            max_point_df = pd.DataFrame([numeric_values.values], columns=myAggData.columns)

            max_point_df = round(max_point_df , 3)

            # Concatenate max_point_df to topErrs
            topErrs = pd.concat([topErrs, max_point_df], axis=0, ignore_index=True)

            # topErrs = pd.concat([topErrs, max_point], axis=0, ignore_index=True)
            myAggData = myAggData.drop(myAggData[myAggData[errorMetric] == max_point[errorMetric]].index)
    
    elif errorMetric in my.fallingMetric:
        for num in range(topNUM):
            max_point = myAggData.loc[myAggData[errorMetric].idxmin()]

            # Convert the values to numeric
            numeric_values = pd.to_numeric(max_point, errors='coerce')
            
            # Transpose max_point Series to create a row DataFrame
            max_point_df = pd.DataFrame([numeric_values.values], columns=myAggData.columns)

            max_point_df = round(max_point_df , 3)

            # Concatenate max_point_df to topErrs
            topErrs = pd.concat([topErrs, max_point_df], axis=0, ignore_index=True)

            # topErrs = pd.concat([topErrs, max_point], axis=0, ignore_index=True)
            myAggData = myAggData.drop(myAggData[myAggData[errorMetric] == max_point[errorMetric]].index)    


    return topErrs



def storeTable(myTableFrame: pd.DataFrame, fromDataset, errorMetric, modelType, taskType, tables_path):
    """
    The storeTable function takes in a DataFrame, the dataset name, error metric type (e.g. MSE, MAE, etc...), model type (regression or classification), 
    task type (classificarion or regression) and the path to store the table.
    It then plots this DataFrame as a table with appropriate column labels and saves it to the folder (tables_path/fromDataset).
    
    :param myTableFrame: pd.DataFrame: The dataframe that is to be plotted as a table
    :param fromDataset: Determine which dataset the table was created from
    :param errorMetric: Specify the error metric used
    :param modelType: Specify the type of model used to generate the results (e.g. RF, etc..)
    :param taskType: Determine if the task is a classification or regression problem
    :param tables_path: Specify the path to store the table
    :return: None
    :doc-author: Trelent
    """
    print(f"Storing the table in {tables_path}/{fromDataset}")

    # Plot the DataFrame as a table
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axis('off')  # Hide axes

    ax.table(cellText=myTableFrame.values, colLabels=myTableFrame.columns, cellLoc='center', loc='center')

    myTitle = glbl.setTitle(fromDataset, glbl.extDataNames) 

    ax.set_title(f'Best {errorMetric.upper()} Scores, {myTitle} Dataset', y=.8)

    tPath = os.path.join(tables_path, fromDataset)

    # plt.show()
    plt.savefig(os.path.join(tPath, f'_{fromDataset}_{errorMetric}_{modelType}_{taskType}_'), dpi=600)




