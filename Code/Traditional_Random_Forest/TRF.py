import sys, os
sys.path.append("..")

import time 
import inputDataParser as parse
import randomForester as forest
import outputDataAggregator as agg
import dataVisualizer as vis
from sklearn.model_selection import train_test_split

base_dir = os.path.dirname(os.path.abspath(__file__))  # Gets the directory where your script is
rawTRFDataPath = os.path.abspath(os.path.join(base_dir, '..', '..', 'Results', 'RawData','TRF'))
aggTRFDataPath = os.path.abspath(os.path.join(base_dir, '..', '..', 'Results', 'AggData','TRF'))
TRFGraphsPath = os.path.abspath(os.path.join(base_dir, '..', '..', 'Results', 'Graphs','TRF'))
TRFTablesPath = os.path.abspath(os.path.join(base_dir, '..', '..', 'Results', 'Tables','TRF'))

regDatasets = ['cali', 'air', 'fb' , 'aba']
clsDatasets = ['income', 'diabetes', 'cancer', 'wine', 'HAR']
allDatasets = regDatasets + clsDatasets

# Define the runs parameters
MAX_RUNS = 2

start_est = 1
final_est = 10
est_step = 1 
ESTNUM = list(range(start_est, final_est + 1, est_step))

start_depth = 1 
final_depth = 5
depth_step = 1
DEPTH = list(range(start_depth, final_depth + 1, depth_step))

def isEmpty(file_path):
    return os.stat(file_path).st_size == 0

def regressionRuns():
    for dataset in allDatasets:
        if dataset in regDatasets: 

            # Get the data 
            if dataset == 'cali':
                X,y = parse.getCaliData()
            elif dataset == 'air':
                X,y = parse.getAirData()
            elif dataset == 'fb':
                X,y = parse.getFbData()
            elif dataset == 'aba':
                X,y = parse.getAbaData()
                
            for numEstimators in ESTNUM:
                for depth in DEPTH:
                    runNumber = 1

                    while (runNumber < MAX_RUNS + 1):


                        # Set file name system for raw data
                        saveRawDataHere = os.path.join(rawTRFDataPath, dataset, f'_{numEstimators}_{depth}_{dataset}_RF_reg_')

                        # add header to raw and agg file
                        with open(saveRawDataHere, 'a') as raw_file:
                            if isEmpty(saveRawDataHere):
                                raw_file.write(f"numTrees\ttreeDepth\toob\tr2\trmse\tmse\tmae\tbuildTime\n") 
                        
                        # Set file name system for agg data
                        saveAggDataHere = os.path.join(aggTRFDataPath, f'_{dataset}_RF_reg_')
                        
                        # add header to agg data file 
                        with open(saveAggDataHere, 'a') as agg_file:
                            if isEmpty(saveAggDataHere):
                                agg_file.write(f"numTrees\ttreeDepth\toob\tr2\trmse\tmse\tmae\tbuildTime\n")

                        # run and time forest building
                        start_time = time.time()
                        oob, r2, rmse, mse, mae = forest.growRegressor(numEstimators, depth, X, y)
                        finish_time = time.time()
                        buildtime = finish_time - start_time
                        with open(saveRawDataHere, 'a') as raw_file:
                            raw_file.write(f"{numEstimators}\t{depth}\t{oob}\t{r2}\t{rmse}\t{mse}\t{mae}\t{buildtime}\n")

                        # increment counter    
                        runNumber += 1



def classificationRuns():
    for dataset in allDatasets:
        if dataset in clsDatasets: 

            # Get the data 
            if dataset == 'income':
                X,y = parse.getIncomeData()
            elif dataset == 'diabetes':
                X,y = parse.getDiabetesData()
            elif dataset == 'cancer':
                X,y = parse.getCancerData()
            elif dataset == 'wine':
                X,y = parse.getWineData()
            elif dataset == 'HAR':
                X,y = parse.getHARData()
                
            for numEstimators in ESTNUM:
                for depth in DEPTH:
                    runNumber = 1

                    while (runNumber < MAX_RUNS + 1):


                        # Set file name system for raw data
                        saveRawDataHere = os.path.join(rawTRFDataPath, dataset, f'_{numEstimators}_{depth}_{dataset}_RF_cls_')

                        # add header to raw and agg file
                        with open(saveRawDataHere, 'a') as raw_file:
                            if isEmpty(saveRawDataHere):
                                raw_file.write(f"numTrees\ttreeDepth\toob\tf1\taccuracy\tprecision\trecall\tbuildTime\n") 
                        
                        # Set file name system for agg data
                        saveAggDataHere = os.path.join(aggTRFDataPath, f'_{dataset}_RF_cls_')
                        
                        # add header to agg data file 
                        with open(saveAggDataHere, 'a') as agg_file:
                            if isEmpty(saveAggDataHere):
                                agg_file.write(f"numTrees\ttreeDepth\toob\tf1\taccuracy\tprecision\trecall\tbuildTime\n")

                        # run and time forest building
                        start_time = time.time()
                        oob, f1, accuracy, precision, recall, conf_matrix = forest.growClassifier(numEstimators, depth, X, y)
                        finish_time = time.time()
                        buildtime = finish_time - start_time
                        with open(saveRawDataHere, 'a') as raw_file:
                            raw_file.write(f"{numEstimators}\t{depth}\t{oob}\t{f1}\t{accuracy}\t{precision}\t{recall}\t{buildtime}\n")

                        # increment counter    
                        runNumber += 1


print('\nstarting regression runs...\n')
regressionRuns()
print('\nregression runs complete...\n')

print('\nstarting classification runs...\n')
classificationRuns()
print('\nclassification runs complete...\n')

print('\nstarting data aggregation...\n')
agg.aggMyData(rawTRFDataPath,aggTRFDataPath)
print('\ndata aggregation complete...\n')

print('\nstarting data tabling and graphing...\n')
vis.graphsNTables(aggTRFDataPath, TRFGraphsPath, TRFTablesPath)
print('\ndata tabling and graphing complete...\n')
