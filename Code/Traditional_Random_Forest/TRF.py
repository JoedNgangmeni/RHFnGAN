import sys, os
sys.path.append("..")

import randomForester as myForest
import outputDataAggregator as agg
import dataVisualizer as vis
import myStructure as my
import inputDataParser as parse

base_dir = os.path.dirname(os.path.abspath(__file__))  # Gets the directory where your script is

focusParentDir = input("In which parent directory do you want to save data ('RESULTS', 'TEST', etc...): ")
focusDataDir = input("What kind of algorithm are you running ('TRF', 'ADA', etc...): ")

rawTRFDataPath = os.path.abspath(os.path.join(base_dir, '..', '..', f'{focusParentDir}', 'RawData', f'{focusDataDir}'))
aggTRFDataPath = os.path.abspath(os.path.join(base_dir, '..', '..', f'{focusParentDir}', 'AggData',f'{focusDataDir}'))
TRFGraphsPath = os.path.abspath(os.path.join(base_dir, '..', '..', f'{focusParentDir}', 'Graphs',f'{focusDataDir}'))
TRFTablesPath = os.path.abspath(os.path.join(base_dir, '..', '..', f'{focusParentDir}', 'Tables',f'{focusDataDir}'))

# # Get storage ready to store data 
my.deleteAllDirs(f'{focusParentDir}', f'{focusDataDir}')
# my.makeAllDirs(f'{focusParentDir}', f'{focusDataDir}')
# # my.resetStorage(f'{focusParentDir}', f'{focusDataDir}')

# # # Define the runs parameters
# MAX_RUNS = int(input("Set the number of runs per permutation: "))

# start_est = int(input("Set min number of trees: "))
# final_est = int(input("Set max number of trees: "))
# est_step = int(input("Set tree number step size: ")) 
# ESTNUM = list(range(start_est, final_est + 1, est_step))

# start_depth = int(input("Set starting tree depth: "))
# final_depth = int(input("Set max tree depth: "))
# depth_step = int(input("Set tree depth step size: "))
# DEPTH = list(range(start_depth, final_depth + 1, depth_step))

# topNUM = my.topHowMany()

# print('\nstarting regression runs...\n')
# myForest.regressionRuns('RF', 'reg', my.allDatasets, my.regDatasets, ESTNUM, DEPTH, MAX_RUNS, rawTRFDataPath, aggTRFDataPath)
# print('\nregression runs complete...\n')

# print('\nstarting classification runs...\n')
# myForest.classificationRuns('RF', 'cls', my.allDatasets, my.clsDatasets, ESTNUM, DEPTH, MAX_RUNS, rawTRFDataPath, aggTRFDataPath)
# print('\nclassification runs complete...\n')

# print('\nstarting data aggregation...\n')
# agg.aggMyData(rawTRFDataPath,aggTRFDataPath)
# print('\ndata aggregation complete...\n')

# print('\nstarting data tabling and graphing...\n')
# vis.graphsNTables(aggTRFDataPath, TRFGraphsPath, TRFTablesPath, topNUM)
# print('\ndata tabling and graphing complete...\n')