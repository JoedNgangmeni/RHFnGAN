import sys, os
sys.path.append("..")

import randomForester as myForest
import outputDataAggregator as agg
import dataVisualizer as vis
import myStructure as my

base_dir = os.path.dirname(os.path.abspath(__file__))  # Gets the directory where your script is

focusParentDir, focusDataDir, MAX_RUNS, ESTNUM, DEPTH, topNUM = my.paramDecider('TRF')


rawDataPath = os.path.abspath(os.path.join(base_dir, '..', '..', f'{focusParentDir}', 'RawData', f'{focusDataDir}'))
aggDataPath = os.path.abspath(os.path.join(base_dir, '..', '..', f'{focusParentDir}', 'AggData',f'{focusDataDir}'))
graphsPath = os.path.abspath(os.path.join(base_dir, '..', '..', f'{focusParentDir}', 'Graphs',f'{focusDataDir}'))
tablesPath = os.path.abspath(os.path.join(base_dir, '..', '..', f'{focusParentDir}', 'Tables',f'{focusDataDir}'))

# # Get storage ready to store data 
if focusParentDir == 'TEST':
    my.deleteAllDirs(f'{focusParentDir}', f'{focusDataDir}')
    my.makeAllDirs(f'{focusParentDir}', f'{focusDataDir}')

elif focusParentDir == 'RESULTS':
    my.makeAllDirs(f'{focusParentDir}', f'{focusDataDir}')

# print('\nstarting regression runs...\n')
# myForest.regressionRuns(f'{focusDataDir}', 'reg', my.allDatasets, my.regDatasets, ESTNUM, DEPTH, MAX_RUNS, rawDataPath, aggDataPath)
# print('\nregression runs complete...\n')

# print('\nstarting classification runs...\n')
# myForest.classificationRuns(f'{focusDataDir}', 'cls', my.allDatasets, my.clsDatasets, ESTNUM, DEPTH, MAX_RUNS, rawDataPath, aggDataPath)
# print('\nclassification runs complete...\n')

# print('\nstarting data aggregation...\n')
# agg.aggMyData(rawDataPath,aggDataPath)
# print('\ndata aggregation complete...\n')

print('\nstarting data tabling and graphing...\n')
vis.graphsNTables(aggDataPath, graphsPath, tablesPath, topNUM)
print('\ndata tabling and graphing complete...\n')

# '''
'''
import inputDataParser as ID

X,y = ID.getClsData("MNIST")

# print(f'year x: {X[1]}\n')
# print(y.shape)
print(X.shape)
'''