import os
import pandas as pd
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
    # Your code to process a single data file goes here
    # Return the processed data (e.g., a DataFrame)
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
    # Initialize an empty list to store processed data from each file
    procRegData = pd.DataFrame(columns=regErrCol)
    procClsData = pd.DataFrame(columns=clsErrCol)

    # Iterate over all entries in the subdirectory
    for entry in os.listdir(subdirectory_path):
        entry_path = os.path.join(subdirectory_path, entry)
        
        # Check if the entry is a directory
        if os.path.isdir(entry_path):
            # If it's a directory, ignore it or handle it as needed
            continue
        elif os.path.isfile(entry_path):
            # If it's a file, process it
            # data = pd.read_csv(entry_path, sep='\t', header=0)
            # if data.shape[1] == 5: # regression data
            #     new_entry_path = os.path.join(subdirectory_path, f'{entry}_')
            #     os.rename(entry_path, new_entry_path)
            # else:
            #     new_entry_path = os.path.join(subdirectory_path, f'{entry}_')
            #     os.rename(entry_path, new_entry_path)

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
            saveHere = os.path.join(figSubPath, f'_{whichModel}_{whichTask}_avg_Errs_{whichData}')

            avgErrs = process_data_file(entry_path)

            # if 'r2' in avgErrs.columns: # Regression
            #     procRegData.loc[treeDepth] = avgErrs

            # elif 'f1' in avgErrs.columns: # classification
            #     procClsData.loc[treeDepth] = avgErrs
        
        # print(f'avgErr: {avgErrs.flatten()}\n')
        print(f'procRegData: {procClsData.columns}\n')


    if 'r2' in avgErrs.columns: # Regression
        with open(saveHere, 'a') as myFigData:
            myFigData.write(f"r2\trmse\tmse\tmae\toob\n")   

    elif 'f1' in avgErrs.columns: # classification
        with open(saveHere, 'a') as myFigData:
            myFigData.write(f"accuracy\tprecision\trecall\tf1\toob\n")   
        

    

    # myGraph = plt.axes(projection='3d')

    # # Combine all processed data into a single DataFrame
    # combined_data = pd.concat(processed_data, ignore_index=True)

    # # Create a graph using the combined data
    # plt.figure()
    # # Your code to create a graph goes here
    # plt.title(f"Graph for {os.path.basename(subdirectory_path)}")
    # plt.show()

def process_directory(directory_path):
    # Iterate over all entries in the directory
    for entry in os.listdir(directory_path):
        entry_path = os.path.join(directory_path, entry)

        # Check if the entry is a subdirectory
        if os.path.isdir(entry_path):
            # If it's a subdirectory, process it
            process_subdirectory(entry_path)


def main():
    # Process the entire directory
    process_directory(outputPath)

if __name__ == "__main__":
    main()
