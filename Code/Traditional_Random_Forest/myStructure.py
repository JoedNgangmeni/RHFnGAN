import os, shutil

myDir = os.path.dirname(os.path.abspath(__file__))  # Gets the directory where your script is
resultDirs = ['RawData', 'AggData', 'Graphs', 'Tables']

# regDatasets = ['cali', 'air', 'fb' , 'aba']
# clsDatasets = ['income', 'diabetes', 'cancer', 'wine', 'HAR', 'heart']
# allDatasets = regDatasets + clsDatasets

regDatasets = ['year', 'air']
clsDatasets = ['MNIST', 'HAR']
allDatasets =  regDatasets + clsDatasets


extDataNames =['California Housing', 'Italy Air Quality', 'Facebook Comment Volume', 'Abalone', 'Pima Native American Diabetes' , 'Wisconsin Breast Cancer Diagnostic', 'Portugal Wine Quality' , 'Human Activity Recognition', 'Adult Income', 'C.H.S.LB Heart Disease', 'Year Prediction MSD']
extFolderNames = ['cali', 'air', 'fb' , 'aba', 'diabetes', 'cancer', 'wine', 'HAR', 'income', 'heart', 'MNIST', 'year']

risingMetric = ['r2', 'accuracy', 'precision', 'recall', 'f1','buildTime']
fallingMetric = ['rmse', 'mse', 'mae', 'mlogloss']


def resetStorage(parPath:str, curModel:str):
    print(f'\nResetting storage...\n')
    deleteAllDirs(parPath, curModel)
    makeAllDirs(parPath, curModel)
    print(f'\nStorage has been reset...\n')

def makeAllDirs(parPath:str, curModel:str):
    print(f'\nEnsuring all necessary folders for data storage exist...\n')
    for curDir in resultDirs:
        parentDir = os.path.abspath(os.path.join(myDir, '..', parPath, curDir, curModel))
        makeParentDir(parentDir)
        if curDir !='AggData':
            makeSubDirs(parentDir, allDatasets)
    print(f'\nAll necessary folders for data storage exist...\n')
    

def deleteAllDirs(parPath:str, curModel:str):
    print(f'Deleting all previously existing storage files...\n')
    for curDir in resultDirs:
        parentDir = os.path.abspath(os.path.join(myDir, '..', parPath, curDir, curModel))
        print(f'Deleting:{parentDir}')
        if os.path.exists(parentDir):
            shutil.rmtree(parentDir)
    
    print(f'All previously existing storage has been deleted...')

def makeParentDir(TESTPATHPath):
    print(f'Checking if {TESTPATHPath} exists...')
    if not os.path.exists(TESTPATHPath):
        print(f'{TESTPATHPath} does not exists, making one...')
        os.makedirs(TESTPATHPath)
    print(f'{TESTPATHPath} has been created...\n')


def makeSubDirs(TESTPATHDir, subDirList:list):
    for name in subDirList:
        subDir = os.path.abspath(os.path.join(TESTPATHDir, name))
        print(f'Checking if {subDir} exists...')
        if not os.path.exists(subDir):
            print(f'{subDir} does not exists, making one...')
            os.makedirs(subDir)
        print(f'{subDir} has been created...\n')


def setTitle(fromDataset: str, extDataNames:list):
    """
    The setTitle function takes in a string and a list of strings.
    The string is the abbreviated name of the dataset that we are working with, 
    and the list contains the full names for each dataset. 
    
    :param fromDataset: str: abbreviated name of the dataset that we are working with
    :param extDataNames:list: list containing the full names for each dataset
    :return: The full name of the dataset
    :doc-author: Trelent
    """
    if fromDataset == 'cali':
        myTitle = extDataNames[0]
    elif fromDataset == 'air':
        myTitle = extDataNames[1]
    elif fromDataset == 'fb':
        myTitle = extDataNames[2]
    elif fromDataset == 'aba':
        myTitle = extDataNames[3]
    elif fromDataset == 'diabetes':
        myTitle = extDataNames[4]
    elif fromDataset == 'cancer':
        myTitle = extDataNames[5]
    elif fromDataset == 'wine':
        myTitle = extDataNames[6]
    elif fromDataset == 'HAR':
        myTitle = extDataNames[7]
    elif fromDataset == 'income':
        myTitle = extDataNames[8] 
    elif fromDataset == 'heart':
        myTitle = extDataNames[9] 
    elif fromDataset == 'year':
        myTitle = extDataNames[10] 
    elif fromDataset == 'MNIST':
        myTitle = 'MNIST' 

    return myTitle


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
        