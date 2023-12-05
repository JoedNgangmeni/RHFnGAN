import os
import pandas as pd
import matplotlib.pyplot as plt

base_dir = os.path.dirname(os.path.abspath(__file__))  # Gets the directory where your script is

outputDir = "Output"  # Name of the outputDir where you want to save the file
outputPath = os.path.join(base_dir, outputDir)

figureDir = "Figures"
figurePath = os.path.join(base_dir, figureDir)

def isForRegression( myFrame = pd.DataFrame()):
    if myFrame.shape[1] == 5:
        return True
    else:
        return False

def regProcess(myFrame = pd.DataFrame()):
    myColumns = ['r2', 'oob', 'mae']
    # print(f'Regression Head {myFrame.columns}')
    for metric in myColumns:
        best_row = round(myFrame.loc[myFrame[metric].idxmax()], 3)
        print(best_row)


def classProcess(myFrame = pd.DataFrame()):
    myColumns = ['accuracy', 'precision', 'recall', 'f1', 'oob', 'confMatrxVars']

    # print(f'Classification Head {myFrame.columns}')

def process_data_file(file_path):
    # Your code to process a single data file goes here
    # Return the processed data (e.g., a DataFrame)
    data = pd.read_csv(file_path, sep='\t', header=0)
    if isForRegression(data):
        print(file_path)
        regProcess(data)
    # else:
    #     classProcess(data)
    
def process_subdirectory(subdirectory_path):
    # Initialize an empty list to store processed data from each file
    processed_data = []

    # Iterate over all entries in the subdirectory
    for entry in os.listdir(subdirectory_path):
        entry_path = os.path.join(subdirectory_path, entry)

        # Check if the entry is a directory
        if os.path.isdir(entry_path):
            # If it's a directory, ignore it or handle it as needed
            continue
        elif os.path.isfile(entry_path):
            # If it's a file, process it
            # print(entry_path)
            process_data_file(entry_path)

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
