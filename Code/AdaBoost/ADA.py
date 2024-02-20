import sys, os
sys.path.append("..")

import outputDataAggregator as agg
import dataVisualizer as vis
import myStructure as my
import adabooster as myBoost
import globalStuff as glbl



base_dir = os.path.dirname(os.path.abspath(__file__))  # Gets the directory where your script is

focusParentDir, focusDataDir, MAX_RUNS, ESTNUM, DEPTH, topNUM = my.paramDecider('ADA')
# focusParentDir, focusDataDir, MAX_RUNS, ESTNUM, DEPTH, topNUM = my.testRun('ADA')


rawDataPath = os.path.abspath(os.path.join(base_dir, '..', '..', f'{focusParentDir}', 'RawData', f'{focusDataDir}'))
aggDataPath = os.path.abspath(os.path.join(base_dir, '..', '..', f'{focusParentDir}', 'AggData',f'{focusDataDir}'))
graphsPath = os.path.abspath(os.path.join(base_dir, '..', '..', f'{focusParentDir}', 'Graphs',f'{focusDataDir}'))
tablesPath = os.path.abspath(os.path.join(base_dir, '..', '..', f'{focusParentDir}', 'Tables',f'{focusDataDir}'))

# import inputDataParser as parse
# X, y = parse.getMNISTData()
# DF = myBoost.growClassifier(3, 2, X, y)

# Get storage ready to store data 
if focusParentDir == 'TEST':
    glbl.deleteAllDirs(f'{focusParentDir}', f'{focusDataDir}')
    glbl.makeAllDirs(f'{focusParentDir}', f'{focusDataDir}', my.allDatasets)

elif focusParentDir == 'RESULTS':
    # glbl.deleteAllDirs(f'{focusParentDir}', f'{focusDataDir}')
    glbl.makeAllDirs(f'{focusParentDir}', f'{focusDataDir}', my.allDatasets)

print('\nstarting regression runs...\n')
myBoost.regressionRuns(f'{focusDataDir}', 'reg', my.allDatasets, my.regDatasets, ESTNUM, DEPTH, MAX_RUNS, rawDataPath, aggDataPath)
print('\nregression runs complete...\n')

print('\nstarting classification runs...\n')
myBoost.classificationRuns(f'{focusDataDir}', 'cls', my.allDatasets, my.clsDatasets, ESTNUM, DEPTH, MAX_RUNS, rawDataPath, aggDataPath)
print('\nclassification runs complete...\n')

print('\nstarting data aggregation...\n')
agg.aggMyData(rawDataPath,aggDataPath)
print('\ndata aggregation complete...\n')

print('\nstarting data tabling and graphing...\n')
vis.graphsNTables(aggDataPath, graphsPath, tablesPath, topNUM)
print('\ndata tabling and graphing complete...\n')

