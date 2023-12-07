import pandas as pd
import os
import numpy as np

# myDF = pd.DataFrame()
# s1, s2, s3 = pd.Series(), pd.Series(), pd.Series()
# arr1 = [1, 2, 3]
# arr2 = [4, 5, 6]
# myDF['f1'] = s1
# myDF['oob'] = s2
# myDF['testGroup'] = s3

# # Concatenate two lists to the 'testGroup' column
# myDF['testGroup'] = [arr1] + [arr2]
# myDF['f1'] = [7, 8]

# f1Mean = myDF['f1'].mean()
# test_mean = myDF['testGroup'].apply(lambda x: pd.Series(x)).mean(axis=0)

# print("Shape of myDF:", myDF.shape)
# print("Headers of myDF:", myDF.columns)
# print("myDF:")
# print(test_mean)  # Access the second element of the 'testGroup' column


base_dir = os.path.dirname(os.path.abspath(__file__))  # Gets the directory where your script is
outputDir = "Output"  # Name of the outputDir where you want to save the file
outputPath = os.path.join(base_dir, outputDir)

def process_data_file(file_path):
    # Your code to process a single data file goes here
    # Return the processed data (e.g., a DataFrame)
    data = pd.read_csv(file_path, sep='\t', header=0)
    
    # Initialize a list to store lists created from 'confMatrxVars'
    s6 = []
    
    # Initialize the DataFrame for averages
    avgErrs = pd.DataFrame()

    # Calculate average values for specific columns
    avgErrs['accuracy'] = [data['accuracy'].mean()]
    avgErrs['precision'] = [data['precision'].mean()]
    avgErrs['recall'] = [data['recall'].mean()]
    avgErrs['f1'] = [data['f1'].mean()]
    avgErrs['oob'] = [data['oob'].mean()]

    # Process 'confMatrxVars' column
    for row in range(data.shape[0]):
        conf_matrix = list(map(int, data['confMatrxVars'][row].split(',')))
            # Reshape the list into a square matrix
        conf_matrix = np.array(conf_matrix).reshape(int(np.sqrt(len(conf_matrix))), -1)
        # Append the matrix to the list
        s6.append(conf_matrix)

    # Create a DataFrame from the list of confusion matrices
    conf_df = pd.DataFrame(s6)

    # Calculate the mean across matrices (axis=0)
    avg_matrix = conf_df.mean(axis=0)

    # Flatten the matrix to store it in the 'avgErrs' DataFrame
    avgErrs['confMatrxVars'] = avg_matrix.values.flatten()
    
    
    # print(type(s_6[0][0]))
    print(avgErrs)

    return avgErrs
 

myFile = 'Output/wineOutput/_200est_16deep_wine_RF_cls_'


process_data_file(myFile)

# print(PF.shape)
