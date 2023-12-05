import numpy as np
import pandas as pd
from tabulate import tabulate
import matplotlib.pyplot as plt
import os

base_dir = os.path.dirname(os.path.abspath(__file__))  # Gets the directory where your script is
outputDir = "Output"  # Name of the outputDir where you want to save the file
outputPath = os.path.join(base_dir, outputDir)
figureDir = "Figures"
figurePath = os.path.join(base_dir, figureDir)

N_ESTIMATORS = [1, 10, 50, 100, 150, 200]
TARGET_COLUMN = 'r2'

best_row, avg_row, worst_row = pd.Series(), pd.Series(), pd.Series()
bestFrame, avgFrame, worstFrame = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()


def regressionAggregator():
    print("hello")

def process_files_in_directory(directory):
    # Iterate over all entries in the directory
    for entry in os.listdir(directory):
        entry_path = os.path.join(directory, entry)

        # Check if the entry is a directory
        if os.path.isdir(entry_path):
            # If it's a directory, recursively call the function
            process_files_in_directory(entry_path)
        else:
            # If it's a file, perform your desired operation
            # print(f"Processing file: {entry_path}")
            data = pd.read_csv(entry_path, delimiter='\t', header=0)
            print(entry_path,'\n')
            print(data.head())
            # Your code to process the file goes here

process_files_in_directory(outputPath)

# for numEstimators in N_ESTIMATORS:
#     fileName = f'{numEstimators}_est'
#     output_path = os.path.join(base_dir, outputDir, fileName)
#     df = pd.read_csv(output_path, sep='\t', header=None)
#     df.columns = ['r2', 'rmse', 'mse', 'oob', 'mae']
    
#     avg_row = round(df.mean(), 3)
#     avgFrame[f'{numEstimators}'] = avg_row
#     # Choose the column for which you want to find the best row (adjust TARGET_COLUMN)

#     if TARGET_COLUMN == 'r2':
#         # Find the row with the minimum value in the chosen column and save it to a new DataFrame
#         best_row = round(df.loc[df[TARGET_COLUMN].idxmax()], 3)
#         bestFrame[f'{numEstimators}'] = best_row

#         worst_row = round(df.loc[df[TARGET_COLUMN].idxmin()] , 3)
#         worstFrame[f'{numEstimators}'] = worst_row
    
#     if TARGET_COLUMN != 'r2':
#         best_row = round(df.loc[df[TARGET_COLUMN].idxmin()], 3)
#         bestFrame[f'{numEstimators}'] = best_row
        
#         worst_row = round(df.loc[df[TARGET_COLUMN].idxmax()], 3)
#         worstFrame[f'{numEstimators}'] = worst_row

    
#     # print(f'{fileName}_by_{TARGET_COLUMN}')
#     # print(f'best_row: {best_row.values}')
#     # print(f'avg_row: {avg_row.values}')
#     # print(f'worst_row: {worst_row.values}')
#     # print(f'outputDF: {avgFrame.values}\n')


# whichFrame = ['bestFrame', 'avgFrame', 'worstFrame']

# for i in range(len(whichFrame)):
#     myFrame = globals()[whichFrame[i]]
#     # Add row names to the DataFrame
#     myFrame.insert(0, ' ', ['r2', 'rmse', 'mse', 'oob', 'mae'])
#     table_str = tabulate(myFrame.set_index(' '), headers='keys', tablefmt='fancy_grid')
#     print(f'{whichFrame[i]}:')
#     print(table_str)



#     # Save the table as a PNG image
#     fig, ax = plt.subplots(figsize=(7, 2))  # Adjust figsize as needed
#     ax.axis('off')  # Hide axes

#     # TODO ADD TITLE FOR PLOT
#     if whichFrame[i] == 'bestFrame':
#         plt.title(f'Best Error Scores by {TARGET_COLUMN} Vs. Number of Estimators', y=.88) 
#     elif whichFrame[i] == 'avgFrame':
#         plt.title(f'Average Error Scores Vs. Number of Estimators', y=.88) 
#     elif whichFrame[i] == 'worstFrame':
#         plt.title(f'Worst Error Scores by {TARGET_COLUMN} Vs. Number of Estimators', y=.88)    
#     table = ax.table(cellText=myFrame.values, colLabels=myFrame.columns, cellLoc='center', loc='center')
    
#     # Save the table to a file
#     figure_path = os.path.join(figurePath, f'{TARGET_COLUMN}_{whichFrame[i]}.png')
#     plt.savefig(figure_path)
#     # plt.show()