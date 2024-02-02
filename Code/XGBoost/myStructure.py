regDatasets = ['year', 'air']
clsDatasets = ['MNIST', 'HAR']
allDatasets =  regDatasets + clsDatasets

regHeader = ['numTrees', 'treeDepth', 'r2_train', 'r2_test',
                             'rmse_train', 'rmse_test', 'mse_train', 'mse_test',
                               'mae_train', 'mae_test', 'buildTime_train', 'buildTime_test']

clsHeader = ['numTrees', 'treeDepth','mlogloss_train', 'mlogloss_test',
                             'f1_train', 'f1_test', 'accuracy_train', 'accuracy_test',
                               'precision_train', 'precision_test', 'recall_train',
                                 'recall_test', 'buildTime_train', 'buildTime_test']

risingMetric = ['r2_train', 'accuracy_train', 'precision_train', 'recall_train', 'f1_train','buildTime_train'
                , 'r2_test', 'accuracy_test', 'precision_test', 'recall_test', 'f1_test','buildTime_test']

fallingMetric = ['rmse_train', 'mse_train', 'mae_train', 'mlogloss_train'
                 ,'rmse_test', 'mse_test', 'mae_test', 'mlogloss_test']

def testRun(focusDataDir:str):
    focusParentDir = 'TEST'
    MAX_RUNS = 2
    ESTNUM = 10
    DEPTH = 3
    bestRuns = 7

    return focusParentDir, focusDataDir, MAX_RUNS, ESTNUM, DEPTH, bestRuns

def prodRun(focusDataDir:str):
    focusParentDir = 'RESULTS'
    
    # # Define the runs parameters
    MAX_RUNS = int(input("Set the number of runs per permutation: "))
    ESTNUM = int(input("Set max number of trees: "))
    DEPTH = int(input("Set max tree depth: "))
    bestRuns = int(input("Input the number of samples you want shown in the tables: "))
    return focusParentDir, focusDataDir, MAX_RUNS, ESTNUM, DEPTH, bestRuns

def paramDecider(focusDataDir:str):
    while True:
        user_input = input("Is this a test run? Please enter 'yes' or 'no': ").lower()  # Convert input to lowercase for case-insensitivity

        if user_input == 'yes':
            print("You entered 'yes'. Beginning test run sequence...\n")
            focusParentDir, focusDataDir, MAX_RUNS, ESTNUM, DEPTH, topNUM = testRun(focusDataDir)
            break  # Exit the loop if the input is valid
        elif user_input == 'no':
            print("You entered 'no'. Beginning prod run sequence...\n")
            focusParentDir, focusDataDir, MAX_RUNS, ESTNUM, DEPTH, topNUM = prodRun(focusDataDir)
            break  # Exit the loop if the input is valid
        else:
            print("Invalid input. Please enter either 'yes' or 'no'.")
            
    return focusParentDir, focusDataDir, MAX_RUNS, ESTNUM, DEPTH, topNUM
        