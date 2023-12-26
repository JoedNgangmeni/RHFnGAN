import os
import pandas as pd, matplotlib.pyplot as plt
import outputDataAggregator as agg

extDataNames =['California Housing', 'Italy Air Quality', 'Facebook Comment Volume', 'Abalone', 'Pima Native American Diabetes' , 'Wisconsin Breast Cancer Diagnostic', 'Portugal Wine Quality' , 'Human Activity Recognition', 'Adult Income']
extFolderNames = ['cali', 'air', 'fb' , 'aba', 'diabetes', 'cancer', 'wine', 'HAR', 'income']

risingMetric = ['r2', 'accuracy', 'precision', 'recall', 'f1','buildTime']
fallingMetric = ['rmse', 'mse', 'mae', 'oob']

def graphsNTables(subdirectory_path, graphs_path, tables_path):
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
                myHeader = ['numTrees', 'treeDepth', 'oob', 'r2', 'rmse', 'mse', 'mae', 'buildTime']

            elif taskType == 'cls': # classification task 
                myHeader = ['numTrees', 'treeDepth', 'oob', 'f1', 'accuracy', 'precision', 'recall', 'buildTime']
            
            myAggData = agg.sortAggData(entry_path)
            topHowMany = 10
            for errorMetric in myHeader[2:]:
                bestTableFrame = makeTableFrame(topHowMany, myAggData, errorMetric)
                storeTable(bestTableFrame, fromDataset, errorMetric, modelType, taskType, tables_path)

                myGraph = makeGraph(myAggData, fromDataset , errorMetric)
                storeGraph(myGraph, fromDataset, errorMetric, modelType, taskType, graphs_path)



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



def makeGraph(myData: pd.DataFrame, fromDataset, errorMetric):
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
    print(f"Graphing {errorMetric} data...")   
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

    X, Y = myData['numTrees'], myData['treeDepth']
   
    if errorMetric == 'buildTime':
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))

        ax[0].scatter(myData[errorMetric],X, color='blue')
        ax[0].set_title(f'{X.name} vs. {errorMetric}')
        ax[0].set_xlabel(f'{errorMetric}')
        ax[0].set_ylabel(f'{X.name}')


        ax[1].scatter(myData[errorMetric],Y, color='red')
        ax[1].set_title(f'{Y.name} vs. {errorMetric}')
        ax[1].set_xlabel(f'{errorMetric}')
        ax[1].set_ylabel(f'{Y.name}')



    elif errorMetric != 'buildTime': 
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



def storeGraph(myFig: plt.Figure, fromDataset, errorMetric, modelType, taskType, graphs_path):
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
    myFig.savefig(os.path.join(gPath, f'_{fromDataset}_{errorMetric}_{modelType}_{taskType}_'), dpi=600)



def makeTableFrame(topHowMany: int, myAggData: pd.DataFrame, errorMetric):
    """
    The makeTableFrame function takes in the following parameters:
        topHowMany - The number of rows to return from the dataframe.
        myAggData - The dataframe that contains all of the aggregated error metrics.
        errorMetric - A string the name of the an error metric (e.g. mse, mae, rmse, r2).
    
    :param topHowMany: int: Specify how many rows of data to return
    :param myAggData: pd.DataFrame: Pass in the aggregated data frame from which you will select rows
    :param errorMetric: Determine which error metric to use for the analysis
    :return: A dataframe with the topHowMany rows of myAggData
    :doc-author: Trelent
    """
    print(f"Putting the top {topHowMany} {errorMetric.upper()} into a frame...")

    topErrs = pd.DataFrame(columns=myAggData.columns)   

    if errorMetric in risingMetric:
        for num in range(topHowMany):
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
    
    elif errorMetric in fallingMetric:
        for num in range(topHowMany):
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




