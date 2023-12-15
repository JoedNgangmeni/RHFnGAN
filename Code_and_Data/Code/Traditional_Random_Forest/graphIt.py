import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

base_dir = os.path.dirname(os.path.abspath(__file__))  # Gets the directory where your script is

outputDir = "Output"  # Name of the outputDir where you want to save the file
outputPath = os.path.join(base_dir, outputDir)

figureDir = "Figures"
figurePath = os.path.join(base_dir, figureDir)

regressionDatasets = ['cali', 'air', 'fb' , 'aba']
classificationDatasets = ['income', 'diabetes', 'cancer', 'wine', 'HAR']

allDatasets = regressionDatasets + classificationDatasets

regErrCol = ['r2', 'rmse', 'mse', 'mae', 'oob']
clsErrCol = ['accuracy', 'precision', 'recall', 'f1', 'oob']


def process_data_file(file_path):
    # Process a single data file
    # Return the processed data as DataFrame
    data = pd.read_csv(file_path, sep='\t', header=0)
    avgErrs = pd.DataFrame()

    if data.shape[1] == 5: # regression data
        # print(file_path)
        avgErrs['r2'] = [data['r2'].mean()]
        avgErrs['rmse'] = [data['rmse'].mean()]
        avgErrs['mse'] = [data['mse'].mean()]
        avgErrs['mae'] = [data['mae'].mean()]
        avgErrs['oob'] = [data['oob'].mean()]
        
    else: # Classification data
        avgErrs['accuracy'] = [data['accuracy'].mean()]
        avgErrs['precision'] = [data['precision'].mean()]
        avgErrs['recall'] = [data['recall'].mean()]
        avgErrs['f1'] = [data['f1'].mean()]
        avgErrs['oob'] = [data['oob'].mean()]

    return avgErrs
    
def process_subdirectory(subdirectory_path):
    # Iterate over all stored data, Ouput to files ----> numTrees in the run \t treeDepth for that run \t avg err for that numTrees and treeDepth \n
    # Each average error metric hs its own file
    # Each dataset has a group of avg error metric files 

    # Iterate over all entries in the subdirectory
    for entry in os.listdir(subdirectory_path):
        entry_path = os.path.join(subdirectory_path, entry)

        ''' THESE WERE TO GET A HEADER IN THE FIGURE FILE '''

        # data = pd.read_csv(entry_path , sep='\t', header=0)

        # if 'r2' in data.columns: # Regression Task
        #     with open(saveHere, 'a') as myFigData:
        #         myFigData.write(f"numTrees\ttreeDepth\tr2\trmse\tmse\mae\oob\n")

        # if 'f1' in data.columns: # Classification Task
        #     with open(saveHere, 'a') as myFigData:
        #         myFigData.write(f"accuracy\tprecision\trecall\tf1\toob\n")

        ''' THESE WERE TO GET A HEADER IN THE FIGURE FILE '''
        
        # Check if the entry is a directory
        if os.path.isdir(entry_path):
            # If it's a directory, ignore it or handle it as needed
            continue
        elif os.path.isfile(entry_path):

            # split file name by '_'
            fName = entry.split('_')

            # Assign last 4 values of the split, should be numtrees, treedepth, datasetname, modeltype
            attrName = fName[1:-1]
            numTrees = int(attrName[0].removesuffix('est'))
            treeDepth = int(attrName[1].removesuffix('deep'))
            whichData = attrName[2]
            whichModel = attrName[3]
            whichTask = attrName[4]

            figSubPath = os.path.join(figurePath, f'{whichData}Output')

            avgErrs = process_data_file(entry_path)

        '''YOU CAN ONLY HAVE ONE OF THE FOLLOWING FOR LOOPS AT A TIME  '''

        # # CLEAR EXISTING DATA
        # for col in avgErrs.columns:
        #     saveHere = os.path.join(figSubPath, f'_avgErr_{whichData}_{whichModel}_{whichTask}')
        #     with open(saveHere, 'w'):
        #         pass 
        
        # # GET NEW DATA
        saveHere = os.path.join(figSubPath, f'_avgErr_{whichData}_{whichModel}_{whichTask}')
        if 'r2' in avgErrs.columns: # Regression Task
            with open(saveHere, 'a') as myFigData:
                myFigData.write(f"{numTrees}\t{treeDepth}\t{avgErrs['r2'].loc[0]}\t{avgErrs['rmse'].loc[0]}\t{avgErrs['mse'].loc[0]}\t{avgErrs['mae'].loc[0]}\t{avgErrs['oob'].loc[0]}\n")

        if 'f1' in avgErrs.columns: # Classification Task
            with open(saveHere, 'a') as myFigData:
                myFigData.write(f"{numTrees}\t{treeDepth}\t{avgErrs['accuracy'].loc[0]}\t{avgErrs['precision'].loc[0]}\t{avgErrs['recall'].loc[0]}\t{avgErrs['f1'].loc[0]}\t{avgErrs['oob'].loc[0]}\n")        
        

        '''YOU CAN ONLY HAVE ONE OF THE ABOVE FOR LOOPS AT A TIME  '''
        


def process_directory(directory_path):
    # Iterate over all entries in the directory
    for entry in os.listdir(directory_path):
        entry_path = os.path.join(directory_path, entry)

        # Check if the entry is a subdirectory
        if os.path.isdir(entry_path):
            # If it's a subdirectory, process it
            process_subdirectory(entry_path)


def makeGraph(directory_path):
    # Iterate over all entries in the directory
    for entry in os.listdir(directory_path):
        entry_path = os.path.join(directory_path, entry)

        # Check if the entry is a directory
        if os.path.isdir(entry_path):
            # If it's a directory, ignore it or handle it as needed
            makeGraph(entry_path)
            
        if os.path.isfile(entry_path):
            if entry_path == '.DS_Store':
                pass
            # split file name by '_'
            fName = entry.split('_')
            print(entry_path)

            # Assign last 4 values of the split, should be numtrees, treedepth, datasetname, modeltype
            attrName = fName[1:]
            whichData = attrName[1]
            whichModel = attrName[2]
            whichTask = attrName[3]

            data = pd.read_csv(entry_path, sep='\t', header=None)

            if whichTask == 'reg': # regression task 
                data.columns = ['trees', 'depth', 'r2', 'rmse', 'mse', 'mae', 'oob']

            elif whichTask == 'cls': # classification task 
                data.columns = ['trees', 'depth', 'accuracy', 'precision', 'recall', 'f1', 'oob']

            useCol = data.columns[2:]
            
            sortedData = data.sort_values(by=['trees', 'depth'], ascending=[True, True])

            X, Y = sortedData['trees'], sortedData['depth']

            for error in useCol:
                Z = sortedData[error]
                # fig = plt.figure()
                # ax = fig.add_subplot(111, projection='3d')

                # # Plot the 3D surface
                # ax.plot_trisurf(X, Y, sortedData[error], cmap='inferno', edgecolor='k', linewidth=0.03)

                # # colors = ['red', 'orangered', 'darkorange', 'orange', 'gold', 'khaki', 'navajowhite', 'white' ]
                # colors = ['royalblue', 'dodgerblue', 'steelblue', 'deepskyblue', 'lightskyblue', 'skyblue', 'lightsteelblue', 'powderblue']

                mySortedData = sortedData
                # graphPeak = 1 
                # for mycolor in colors:
                #     # Find the highest point
                    
                   

                #     # Plot a circle at the x highest point
                #     label_info = f'{graphPeak} Highest Point - Error: {max_point[error]:.5f}, Trees: {max_point["trees"]}, Depth: {max_point["depth"]}'
                #     ax.scatter(max_point['trees'], max_point['depth'], max_point[error], color=mycolor, s=100, label=label_info)

                #     # Add an arrow pointing to the highest point
                #     arrow_length = .2
                #     ax.quiver(max_point['trees'], max_point['depth'], max_point[error], 0, 0, arrow_length, color=mycolor, arrow_length_ratio=0.1, alpha=0.8)

                #     # Remove the highest data point 
                #     # sortedData = sortedData.drop(sortedData[error] == max_point[error].index)

                    
                #     graphPeak +=1



                # # Set labels
                # ax.set_xlabel('Trees')
                # ax.set_ylabel('Depth')
                # ax.set_zlabel(error)

                topBestErr = 10
                topErrs = pd.DataFrame(columns=mySortedData.columns)        
                if whichTask == 'cls':
                    for _ in range(topBestErr):
                        max_point = mySortedData.loc[mySortedData[error].idxmax()]
                        # Transpose max_point Series to create a row DataFrame
                        max_point_df = pd.DataFrame([max_point.values], columns=mySortedData.columns)

                        max_point_df = round(max_point_df , 4)

                        # Concatenate max_point_df to topErrs
                        topErrs = pd.concat([topErrs, max_point_df], axis=0, ignore_index=True)

                        # topErrs = pd.concat([topErrs, max_point], axis=0, ignore_index=True)
                        mySortedData = mySortedData.drop(mySortedData[mySortedData[error] == max_point[error]].index)
                
                elif whichTask == 'reg':
                    for _ in range(topBestErr):
                        max_point = mySortedData.loc[mySortedData[error].idxmin()]
                        # Transpose max_point Series to create a row DataFrame
                        max_point_df = pd.DataFrame([max_point.values], columns=mySortedData.columns)

                        max_point_df = round(max_point_df , 4)

                        # Concatenate max_point_df to topErrs
                        topErrs = pd.concat([topErrs, max_point_df], axis=0, ignore_index=True)

                        # topErrs = pd.concat([topErrs, max_point], axis=0, ignore_index=True)
                        mySortedData = mySortedData.drop(mySortedData[mySortedData[error] == max_point[error]].index)
                

                topErrs = topErrs.sort_values(by=['trees', 'depth'], ascending=[True, True], ignore_index=True)


                # Plot the DataFrame as a table
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.axis('off')  # Hide axes for better visualization
                table = ax.table(cellText=topErrs.values, colLabels=topErrs.columns, cellLoc='center', loc='center')
                table.auto_set_font_size(False)
                table.set_fontsize(10)  


                extDataNames =['California Housing', 'Italy Air Quality', 'Facebook Comment Volume', 'Abalone', 'Pima Native American Diabetes' , 'Wisconsin Breast Cancer Diagnostic', 'Portugal Wine Quality' , 'Human Activity Recognition', 'Adult Income']

                if whichData == 'cali':
                    myTitle = extDataNames[0]
                elif whichData == 'air':
                    myTitle = extDataNames[1]
                elif whichData == 'fb':
                    myTitle = extDataNames[2]
                elif whichData == 'aba':
                    myTitle = extDataNames[3]
                elif whichData == 'diabetes':
                    myTitle = extDataNames[4]
                elif whichData == 'cancer':
                    myTitle = extDataNames[5]
                elif whichData == 'wine':
                    myTitle = extDataNames[6]
                elif whichData == 'HAR':
                    myTitle = extDataNames[7]
                elif whichData == 'income':
                    myTitle = extDataNames[8]

                # ax.set_xlabel('X-axis Label', fontsize=12)  # Adjust the font size of the X-axis label
                # ax.set_ylabel('Y-axis Label', fontsize=12)  # Adjust the font size of the Y-axis label
                # ax.set_title('Title', fontsize=14)  # Adjust the font size of the title

                
                title = ax.set_title(f'Top {topBestErr} Best Scores on {myTitle} Data')
                title.set_fontsize(14)
                # plt.subplots_adjust(top=0.30)
                # ax.view_init(elev=27, azim=-139)

                

                # Save the graph
                # fig.savefig(os.path.join(directory_path, f'_avg_{whichModel}_{whichTask}_{error}_{whichData}.png'), dpi=600)
                fig.savefig(os.path.join(directory_path, f'_avg_{whichModel}_{whichTask}_{error}_{whichData}.png'), dpi=600 ,bbox_inches='tight')
                # plt.legend()
                # plt.show()
                plt.close(fig)

                # Show the plot
                # print(ax.elev, ax.azim)




                
            # print(whichData, whichModel, whichTask)




def main():
    # Process the entire directory
    # process_directory(outputPath)
    makeGraph(figurePath)

if __name__ == "__main__":
    main()
