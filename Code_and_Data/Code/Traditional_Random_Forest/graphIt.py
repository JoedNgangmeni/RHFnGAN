import os
import pandas as pd
import matplotlib.pyplot as plt

base_dir = os.path.dirname(os.path.abspath(__file__))  # Gets the directory where your script is

outputDir = "Output"  # Name of the outputDir where you want to save the file
outputPath = os.path.join(base_dir, outputDir)

figureDir = "Figures"
figurePath = os.path.join(base_dir, figureDir)



def process_data_file(file_path):
    # Your code to process a single data file goes here
    # Return the processed data (e.g., a DataFrame)
    data = pd.read_csv(file_path, sep='\t', header=0)
    if data.shape[1] == 5: # regression data
        # print(file_path)
        myColumns = ['r2', 'oob', 'mae']
        allErrs = pd.DataFrame(columns=myColumns)

    #   print(f'Regression Head {myFrame.columns}')
        for metric in myColumns:
            best_row = round(data.loc[data[metric].idxmax()], 5)
            # allErrs = allErrs.append(best_row[myColumns], ignore_index=True)

            allErrs = pd.concat([allErrs, best_row[myColumns].to_frame().T], ignore_index=True)
            # print(f'allErrs , {allErrs}\n\n\n\n')


            # allErrs = allErrs[f'{metric}'].append(best_row[metric], ignore_index=True)
            # allErrs = allErrs.append(best_row[myColumns], ignore_index=True)
        
        return allErrs
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
            myErr = process_data_file(entry_path)
            fName = entry_path.split('_')
            print(f'fName , {fName}\n\n\n\n')


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
