import sys, os
sys.path.append("..")

import dataParser as parse
import randomForester as forest

# base_dir = os.path.dirname(os.path.abspath(__file__))  # Gets the directory where your script is
# saveHere = os.path.join(OUTPUTDATAPATH, f'{NUMTREES}_{DEPTH}_RF_reg')
#     print(f'{saveHere}\n')
# with open(saveHere, 'a') as output_file:
#         output_file.write(f"{r2}\t{oob}\t{rmse}\t{mse}\t{mae}\n")

regressionDatasets = ['cali', 'air', 'fb' , 'aba']
classificationDatasets = ['income', 'diabetes', 'cancer', 'wine', 'HAR']
allDatasets = regressionDatasets + classificationDatasets

NUMTREES = 30
DEPTH = 5
X, y = parse.getWineData()
'''
oob, r2, rmse, mse, mae = forest.buildRegressor(NUMTREES, DEPTH, X, y)
print(oob, r2, rmse, mse, mae)
'''

oob, f1, accuracy, precision, recall, conf_matrix = forest.buildClassifier(NUMTREES, DEPTH, X, y)
print(oob, f1, accuracy, precision, recall, conf_matrix)