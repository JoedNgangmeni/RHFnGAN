import numpy as np
import pandas as pd
from tabulate import tabulate
import matplotlib.pyplot as plt
import os

base_dir = os.path.dirname(os.path.abspath(__file__))  # Gets the directory where your script is
subdirectory = "Output"  # Name of the subdirectory where you want to save the file
imageSubDir = "Figures"
figure_base_path = os.path.join(base_dir, imageSubDir)

N_ESTIMATORS = [1, 10, 50, 100, 150, 200]
TARGET_COLUMN = 'r2'

best_row, avg_row, worst_row = pd.Series(), pd.Series(), pd.Series()
bestFrame, avgFrame, worstFrame = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()


for numEstimators in N_ESTIMATORS:
    fileName = f'{numEstimators}_est'
    output_path = os.path.join(base_dir, subdirectory, fileName)
    df = pd.read_csv(output_path, sep='\t', header=None)
    df.columns = ['r2', 'rmse', 'mse', 'oob', 'mae']
    
    avg_row = round(df.mean(), 3)
    avgFrame[f'{numEstimators}'] = avg_row
    # Choose the column for which you want to find the best row (adjust TARGET_COLUMN)

    if TARGET_COLUMN == 'r2':
        # Find the row with the minimum value in the chosen column and save it to a new DataFrame
        best_row = round(df.loc[df[TARGET_COLUMN].idxmax()], 3)
        bestFrame[f'{numEstimators}'] = best_row

        worst_row = round(df.loc[df[TARGET_COLUMN].idxmin()] , 3)
        worstFrame[f'{numEstimators}'] = worst_row
    
    if TARGET_COLUMN != 'r2':
        best_row = round(df.loc[df[TARGET_COLUMN].idxmin()], 3)
        bestFrame[f'{numEstimators}'] = best_row
        
        worst_row = round(df.loc[df[TARGET_COLUMN].idxmax()], 3)
        worstFrame[f'{numEstimators}'] = worst_row

    
    # print(f'{fileName}_by_{TARGET_COLUMN}')
    # print(f'best_row: {best_row.values}')
    # print(f'avg_row: {avg_row.values}')
    # print(f'worst_row: {worst_row.values}')
    # print(f'outputDF: {avgFrame.values}\n')


whichFrame = ['bestFrame', 'avgFrame', 'worstFrame']

for i in range(len(whichFrame)):
    myFrame = globals()[whichFrame[i]]
    # Add row names to the DataFrame
    myFrame.insert(0, ' ', ['r2', 'rmse', 'mse', 'oob', 'mae'])
    table_str = tabulate(myFrame.set_index(' '), headers='keys', tablefmt='fancy_grid')
    print(f'{whichFrame[i]}:')
    print(table_str)



    # Save the table as a PNG image
    fig, ax = plt.subplots(figsize=(7, 2))  # Adjust figsize as needed
    ax.axis('off')  # Hide axes

    # TODO ADD TITLE FOR PLOT
    if whichFrame[i] == 'bestFrame':
        plt.title(f'Best Error Scores by {TARGET_COLUMN} Vs. Number of Estimators', y=.88) 
    elif whichFrame[i] == 'avgFrame':
        plt.title(f'Average Error Scores Vs. Number of Estimators', y=.88) 
    elif whichFrame[i] == 'worstFrame':
        plt.title(f'Worst Error Scores by {TARGET_COLUMN} Vs. Number of Estimators', y=.88)    
    table = ax.table(cellText=myFrame.values, colLabels=myFrame.columns, cellLoc='center', loc='center')
    
    # Save the table to a file
    figure_path = os.path.join(figure_base_path, f'{TARGET_COLUMN}_{whichFrame[i]}.png')
    plt.savefig(figure_path)
    # plt.show()