import sys, os
sys.path.append("..")

import pandas as pd, matplotlib.pyplot as plt
import outputDataAggregator as agg
import myStructure as my
import globalStuff as glbl
import pandas as pd, matplotlib.pyplot as plt, seaborn as sns, numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as mcolors

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
                myHeader = my.regHeader

            elif taskType == 'cls': # classification task 
                myHeader = my.clsHeader
            
            myAggData = agg.sortAggData(entry_path)
            for errorMetric in myHeader[2:]:
                # bestTableFrame = makeTableFrame(myAggData, errorMetric, topNUM)
                # storeTable(bestTableFrame, fromDataset, errorMetric, modelType, taskType, tables_path)

                my3DGraph = make3DGraph(myAggData, fromDataset , errorMetric)
                storeGraph(my3DGraph, ' ', ' ', fromDataset, errorMetric, modelType, taskType, '3D', graphs_path, isNorm=False)
                my3DGraph.close()


                if 'test' in errorMetric:
                    # non normalize 2D graph with varying treeDepth
                    nonNormtD = my2DGraph(myAggData, fromDataset, 'treeDepth', 'numTrees' ,errorMetric)  
                    storeGraph(nonNormtD, 'treeDepth', 'numTrees', fromDataset, errorMetric, modelType, taskType, '2D', graphs_path, isNorm=False)
                    nonNormtD.close()

                    # non normalize 2D graph with varying numTrees
                    nonNormnT = my2DGraph(myAggData, fromDataset, 'numTrees', 'treeDepth' , errorMetric)
                    storeGraph(nonNormnT, 'numTrees', 'treeDepth' , fromDataset, errorMetric, modelType, taskType, '2D', graphs_path, isNorm=False)
                    nonNormnT.close()



                    # storeGraph(my2DGraph, fromDataset, errorMetric, modelType, taskType,'2D', graphs_path)



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



# def make3DGraph(myData: pd.DataFrame, fromDataset, errorMetric):
#     """
#     The makeGraph function takes in a dataframe, the name of the dataset it came from, and an error metric.
#     It then plots a 3D surface graph with trees on the x-axis, depth on the y-axis, and error metric values on z-axis. 
#     It also plots two 2D scatterplots showing how tree count affects build time (blue) and how 
#     tree depth affects build time (red). The function returns a matplotlib object (the graph).
    
#     :param myData: pd.DataFrame: Pass in the dataframe that we want to graph
#     :param fromDataset: Determine which dataset the data is coming from
#     :param errorMetric: Determine which error metric to graph
#     :return: A plot object (the graph) that we want to save 
#     :doc-author: Trelent
#     """
#     print(f"Making a 3D Graphing of {errorMetric} data...\n")  

#     #scale data
#     # print(f"Scaling the data...\n")  
#     # min_max_scaler = MinMaxScaler()
#     # myData = pd.DataFrame(min_max_scaler.fit_transform(myData), columns=myData.columns)


#     myTitle = glbl.setTitle(fromDataset, glbl.extDataNames) 
    
#     X, Y = myData['numTrees'], myData['treeDepth']
   
#     fig = plt.figure(figsize=(7, 5))
#     ax = fig.add_subplot(111, projection='3d')

#     # Plot the 3D surface
#     ax.plot_trisurf(X, Y, myData[errorMetric], cmap='inferno', edgecolor='k', linewidth=0.03)

#     # Set labels
#     errorMetric = errorMetric.split('_')[0]

#     ax.set_xlabel('Trees')
#     ax.set_ylabel('Depth')
#     ax.set_zlabel(errorMetric)
#     ax.view_init(elev=27, azim=-142)
#     ax.set_title(f'Aggregate {errorMetric} Scores\n{myTitle} Dataset')

#     # plt.show()
#     return plt

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
    print(f"Making a 3D Graphing of {errorMetric} data...\n")  

    myTitle = glbl.setTitle(fromDataset, glbl.extDataNames) 
    
    X, Y = myData['numTrees'], myData['treeDepth']
   
    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111, projection='3d')

    # Generate HUSL color palette
    num_colors = len(myData[errorMetric].unique())
    colors = sns.color_palette('husl', n_colors=num_colors)

    # Convert the list of colors to a colormap object
    cmap = mcolors.ListedColormap(colors)

    # Plot the 3D surface with HUSL color palette
    ax.plot_trisurf(X, Y, myData[errorMetric], cmap=cmap, edgecolor='k', linewidth=0.03)

    # Set labels
    errorMetric = errorMetric.split('_')[0]
    if errorMetric == 'buildTime':
        errorMetric = 'buildTime (s)'

    ax.set_xlabel('Trees')
    ax.set_ylabel('Depth')
    ax.set_zlabel(errorMetric)
    ax.view_init(elev=27, azim=-142)
    ax.set_title(f'Aggregate {errorMetric} Scores\n{myTitle} Dataset')

    # plt.show()
    return plt

def my2DGraph(myData: pd.DataFrame, fromDataset: str, staticVariable: str, changingVar: str, errorMetric: str):
    """
    The my2DGraph function compares the performance of a model based on a static variable (number of trees or tree depth)
    while varying another variable within a specified range.
    
    
    :param myData: pd.DataFrame: Pass in the data to be used for the graph
    :param fromDataset: str: Specify the name of the dataset being used
    :param staticVariable: str: Specify which variable is static
    :param changingVar: str: Specify the variable to be varied
    :param errorMetric: str: Specify the performance metric to plot (e
    :return: A matplotlib plot object
    :doc-author: Trelent
    """
    # # Filter the data based on the static variable
    # filtered_results = myData[(myData[changingVar] >= 1) & (myData[changingVar] <= myData[changingVar].max())]
    
    # Create a color palette with enough distinct colors

    print(f'\nMaking a 2D plot of {fromDataset} data y={errorMetric}, x={changingVar}, over {changingVar}...\n')
    # num_colors = len(myData[staticVariable].unique())
    # colors = sns.color_palette('husl', n_colors=num_colors)

    t_num= len(my.graphTheseTrees)
    tcol = sns.color_palette('husl', n_colors=t_num)

    
    # Create a plot
    fig, ax = plt.subplots(figsize=(7, 5))
    
    line_number = 0
    for label, group in myData.groupby(staticVariable):
        x_values = group[changingVar]
        y_values = group[errorMetric]

        # print(f'label: {label}\n group:{group}\n staticVariable: {staticVariable}\n')

        if staticVariable == 'numTrees':
            if label in my.graphTheseTrees:
                # print (myData.loc[myData[staticVariable]== x])
                ax.plot(x_values, y_values, marker='o', linestyle='-', label=f'{staticVariable}={label}', color=tcol[my.graphTheseTrees.index(label)], markersize=1)    
        
        elif staticVariable == 'treeDepth':
            uniqueDepths = myData[staticVariable].unique()
            d_num= len(uniqueDepths)
            dcol = sns.color_palette('husl', n_colors=d_num)

            depth_index = np.where(uniqueDepths == myData[staticVariable][label])[0]


            if (label == 1) or (label % 2 == 0):
                ax.plot(x_values, y_values, marker='o', linestyle='-', label=f'{staticVariable}={label}', color=dcol[depth_index[0]], markersize=1)    
        line_number +=1        

    errorMetric = errorMetric.split('_')[0]

    if errorMetric == 'buildTime':
        errorMetric = 'buildTime (s)'
    
    ax.set_xlabel(changingVar)
    ax.set_ylabel(errorMetric)
    ax.set_title(f'{errorMetric} vs. {changingVar} over {staticVariable} ranges\n(Dataset: {fromDataset})')
    plt.subplots_adjust(left=.1, right=.74)  # Adjust figure size
    ax.legend(title=staticVariable, bbox_to_anchor=(1.005, 1.1), loc='upper left')  # Put legend on the right
    ax.grid(True)
    # plt.show()
    return plt



def storeGraph(myFig: plt.Figure, staticVar:str, changingVar:str, fromDataset, errorMetric, modelType, taskType, graphtype:str, graphs_path, isNorm: bool):
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

    errorMetric = errorMetric.split('_')[0]

    # myFig.tight_layout()
    if graphtype =='2D':
        if isNorm == True:
            myFig.savefig(os.path.join(gPath, f'_{graphtype}_Norm_{fromDataset}_{errorMetric}_{changingVar}_{staticVar}_{modelType}_{taskType}_'), dpi=600)
        else: 
            myFig.savefig(os.path.join(gPath, f'_{graphtype}_NonNorm_{fromDataset}_{errorMetric}_{changingVar}_{staticVar}_{modelType}_{taskType}_'), dpi=600)
    
    elif graphtype =='3D':
        if isNorm == True:
            myFig.savefig(os.path.join(gPath, f'_{graphtype}_Norm_{fromDataset}_{errorMetric}_{modelType}_{taskType}_'), dpi=600)
        else: 
            myFig.savefig(os.path.join(gPath, f'_{graphtype}_NonNorm_{fromDataset}_{errorMetric}_{modelType}_{taskType}_'), dpi=600)
    


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

    errorMetric = errorMetric.split('_')[0]

    myTitle = glbl.setTitle(fromDataset, glbl.extDataNames) 

    ax.set_title(f'Best {errorMetric} Scores, {myTitle} Dataset', y=.8)

    tPath = os.path.join(tables_path, fromDataset)

    # plt.show()
    plt.savefig(os.path.join(tPath, f'_{fromDataset}_{errorMetric}_{modelType}_{taskType}_'), dpi=600)




