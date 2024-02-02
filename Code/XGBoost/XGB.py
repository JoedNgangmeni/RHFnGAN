import sys, os
sys.path.append("..")

import XGBooster as myForest
import outputDataAggregator as agg
import dataVisualizer as vis
import myStructure as my
import globalStuff as glbl

base_dir = os.path.dirname(os.path.abspath(__file__))  # Gets the directory where your script is

# import inputDataParser as idp 
# import numpy as np
# X, y = idp.getYearData()
# # # testdata = os.path.abspath(os.path.join(base_dir, '..', '..', 'TEST', 'RawData', 'XGB', 'MNIST','_10_1_MNIST_XGB_cls_'))
# # # agg.aggFile(testdata)
# myForest.growRegressor(10,5,X,y)
# duj
                                           
focusParentDir, focusDataDir, MAX_RUNS, ESTNUM, DEPTH, topNUM = my.paramDecider('XGB')


rawDataPath = os.path.abspath(os.path.join(base_dir, '..', '..', f'{focusParentDir}', 'RawData', f'{focusDataDir}'))
aggDataPath = os.path.abspath(os.path.join(base_dir, '..', '..', f'{focusParentDir}', 'AggData',f'{focusDataDir}'))
graphsPath = os.path.abspath(os.path.join(base_dir, '..', '..', f'{focusParentDir}', 'Graphs',f'{focusDataDir}'))
tablesPath = os.path.abspath(os.path.join(base_dir, '..', '..', f'{focusParentDir}', 'Tables',f'{focusDataDir}'))


# # Get storage ready to store data 
if focusParentDir == 'TEST':
    glbl.deleteAllDirs(f'{focusParentDir}', f'{focusDataDir}')
#     glbl.makeAllDirs(f'{focusParentDir}', f'{focusDataDir}', my.allDatasets)

# elif focusParentDir == 'RESULTS':
#     glbl.makeAllDirs(f'{focusParentDir}', f'{focusDataDir}')


# print('\nstarting regression runs...\n')
# myForest.regressionRuns(f'{focusDataDir}', 'reg', my.allDatasets, my.regDatasets, ESTNUM, DEPTH, MAX_RUNS, rawDataPath, aggDataPath)
# print('\nregression runs complete...\n')

# print('\nstarting classification runs...\n')
# myForest.classificationRuns(f'{focusDataDir}', 'cls', my.allDatasets, my.clsDatasets, ESTNUM, DEPTH, MAX_RUNS, rawDataPath, aggDataPath)
# print('\nclassification runs complete...\n')

# print('\nstarting data aggregation...\n')
# agg.aggMyData(rawDataPath,aggDataPath)
# print('\ndata aggregation complete...\n')

# print('\nstarting data tabling and graphing...\n')
# vis.graphsNTables(aggDataPath, graphsPath, tablesPath, topNUM)
# print('\ndata tabling and graphing complete...\n')




'''
import inputDataParser as parse
X,y = parse.getRegData('air')

r2, rmse, mse, mae  = myForest.growRegressor(25, 10, X, y)
print(f"r2, rmse, mse, mae : {r2}, {rmse}, {mse}, {mae} ")

X,y = ID.getClsData("MNIST")

# print(f'year x: {X[1]}\n')
# print(y.shape)
print(X.shape)
'''