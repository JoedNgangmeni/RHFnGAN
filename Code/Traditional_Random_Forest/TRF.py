import sys, os
sys.path.append("..")

import inputDataParser as parse
import randomForester as forest
import outputDataAggregator as agg

base_dir = os.path.dirname(os.path.abspath(__file__))  # Gets the directory where your script is
rawTRFDataPath = os.path.abspath(os.path.join(base_dir, '..', '..', 'Results', 'RawData','TRF'))
aggTRFDataPath = os.path.abspath(os.path.join(base_dir, '..', '..', 'Results', 'AggData','TRF'))


# saveHere = os.path.join(OUTPUTDATAPATH, f'{NUMTREES}_{DEPTH}_RF_reg')
#     print(f'{saveHere}\n')
# with open(saveHere, 'a') as output_file:
#         output_file.write(f"{r2}\t{oob}\t{rmse}\t{mse}\t{mae}\n")


regressionDatasets = ['cali', 'air', 'fb' , 'aba']
classificationDatasets = ['income', 'diabetes', 'cancer', 'wine', 'HAR']
allDatasets = regressionDatasets + classificationDatasets

NUMTREES = 30
DEPTH = 5
whichTask  = 'cls'
'''
if whichTask  == 'reg':
    X, y = parse.getCaliData()
    for _ in range(5):
        oob, r2, rmse, mse, mae = forest.growRegressor(NUMTREES, DEPTH, X, y)
        print(oob, '\n', r2, '\n', rmse,  '\n',mse, '\n', mae,  '\n')

if whichTask  == 'cls':
    X, y = parse.getWineData()
    for _ in range(5):
        oob, f1, accuracy, precision, recall, conf_matrix = forest.growClassifier(NUMTREES, DEPTH, X, y)
        print( f1,'\n', accuracy,'\n', precision,'\n', recall,'\n')
'''

# print(rawTRFDataPath)
agg.aggMyData(rawTRFDataPath, aggTRFDataPath)