import sys, os
sys.path.append("..")

import randomForester as myForest
import outputDataAggregator as agg
import dataVisualizer as vis
import myStructure
import myStructure as my



base_dir = os.path.dirname(os.path.abspath(__file__))  # Gets the directory where your script is
rawTRFDataPath = os.path.abspath(os.path.join(base_dir, '..', '..', 'RESULTS', 'RawData','TRF'))
aggTRFDataPath = os.path.abspath(os.path.join(base_dir, '..', '..', 'RESULTS', 'AggData','TRF'))
TRFGraphsPath = os.path.abspath(os.path.join(base_dir, '..', '..', 'RESULTS', 'Graphs','TRF'))
TRFTablesPath = os.path.abspath(os.path.join(base_dir, '..', '..', 'RESULTS', 'Tables','TRF'))

# Define the runs parameters
MAX_RUNS = 30

start_est = 1
final_est = 200
est_step = 1 
ESTNUM = list(range(start_est, final_est + 1, est_step))

start_depth = 1 
final_depth = 20
depth_step = 1
DEPTH = list(range(start_depth, final_depth + 1, depth_step))

myStructure.resetStorage()

print('\nstarting regression runs...\n')
myForest.regressionRuns('RF', 'reg', my.allDatasets, my.regDatasets, ESTNUM, DEPTH, MAX_RUNS, rawTRFDataPath, aggTRFDataPath)
print('\nregression runs complete...\n')

print('\nstarting classification runs...\n')
myForest.classificationRuns('RF', 'cls', my.allDatasets, my.clsDatasets, ESTNUM, DEPTH, MAX_RUNS, rawTRFDataPath, aggTRFDataPath)
print('\nclassification runs complete...\n')

print('\nstarting data aggregation...\n')
agg.aggMyData(rawTRFDataPath,aggTRFDataPath)
print('\ndata aggregation complete...\n')

print('\nstarting data tabling and graphing...\n')
vis.graphsNTables(aggTRFDataPath, TRFGraphsPath, TRFTablesPath)
print('\ndata tabling and graphing complete...\n')
