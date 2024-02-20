# regDatasets = ['cali', 'air', 'fb' , 'aba']
# clsDatasets = ['income', 'diabetes', 'cancer', 'wine', 'HAR', 'heart']
# allDatasets = regDatasets + clsDatasets

regDatasets = ['year', 'air']
clsDatasets = ['MNIST', 'HAR']
allDatasets =  regDatasets + clsDatasets

risingMetric = ['r2', 'accuracy', 'precision', 'recall', 'f1','buildTime']
fallingMetric = ['rmse', 'mse', 'mae', 'mlogloss']

regHeader = ['numTrees', 'treeDepth', 'r2', 'rmse', 'mse', 'mae', 'buildTime']
clsHeader = ['numTrees', 'treeDepth', 'f1', 'accuracy', 'precision', 'recall', 'buildTime']

graphTheseTrees = [1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 21., 51., 101., 201., 301., 401., 500.]


def testRun(focusDataDir:str):
    focusParentDir = 'TEST'
    MAX_RUNS = 2

    start_est = 1
    final_est = 10
    est_step = 1 
    ESTNUM = list(range(start_est, final_est + 1, est_step))

    start_depth = 1
    final_depth = 3
    depth_step = 1
    DEPTH = list(range(start_depth, final_depth + 1, depth_step))

    bestRuns = 7

    return focusParentDir, focusDataDir, MAX_RUNS, ESTNUM, DEPTH, bestRuns

def prodRun(focusDataDir:str):
    focusParentDir = 'RESULTS'
    
    # # Define the runs parameters
    MAX_RUNS = int(input("Set the number of runs per permutation: "))

    start_est = int(input("Set min number of trees: "))
    final_est = int(input("Set max number of trees: "))
    est_step = int(input("Set tree number step size: ")) 
    ESTNUM = list(range(start_est, final_est + 1, est_step))

    start_depth = int(input("Set starting tree depth: "))
    final_depth = int(input("Set max tree depth: "))
    depth_step = int(input("Set tree depth step size: "))
    DEPTH = list(range(start_depth, final_depth + 1, depth_step))

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
        