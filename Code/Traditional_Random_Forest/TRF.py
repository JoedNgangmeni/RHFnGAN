import sys, os
sys.path.append("..")

import inputDataParser as parse
import randomForester as forest
import outputDataAggregator as agg
import dataVisualizer as vis

base_dir = os.path.dirname(os.path.abspath(__file__))  # Gets the directory where your script is
rawTRFDataPath = os.path.abspath(os.path.join(base_dir, '..', '..', 'Results', 'RawData','TRF'))
aggTRFDataPath = os.path.abspath(os.path.join(base_dir, '..', '..', 'Results', 'AggData','RHF'))
TRFGraphsPath = os.path.abspath(os.path.join(base_dir, '..', '..', 'Results', 'Graphs','TRF'))
TRFTablesPath = os.path.abspath(os.path.join(base_dir, '..', '..', 'Results', 'Tables','TRF'))



'''
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

if whichTask  == 'reg':
    X, y = parse.getCaliData()
    for _ in range(5):
        oob, r2, rmse, mse, mae = forest.growRegressor(NUMTREES, DEPTH, X, y)
        print(oob, '\n', r2, '\n', rmse,  '\n',mse, '\n', mae,  '\n')

if whichTask  == 'cls':
    X, y = parse.getIncomeData()
    for _ in range(5):
        oob, f1, accuracy, precision, recall, conf_matrix = forest.growClassifier(NUMTREES, DEPTH, X, y)
        print( conf_matrix)
        print('acc:\t', accuracy)
        print('prec:\t', precision)
        print('rec:\t', recall)
        print('f1:\t', f1, '\n')
        # print(forest.growClassifier(NUMTREES, DEPTH, X, y))


# print(rawTRFDataPath)
# agg.aggMyData(rawTRFDataPath, aggTRFDataPath)
'''

# agg.aggMyData(rawTRFDataPath, aggTRFDataPath)

# print(aggTRFDataPath)
vis.graphsNTables(aggTRFDataPath, TRFGraphsPath, TRFTablesPath)