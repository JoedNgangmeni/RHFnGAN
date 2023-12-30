import os, shutil

myDir = os.path.dirname(os.path.abspath(__file__))  # Gets the directory where your script is

myPaths = ['TEST', 'RESULTS']

workingModels = ['TRF', 'RHF']

regDatasets = ['cali', 'air', 'fb' , 'aba']
clsDatasets = ['income', 'diabetes', 'cancer', 'wine', 'HAR']
allDatasets = regDatasets + clsDatasets

extDataNames =['California Housing', 'Italy Air Quality', 'Facebook Comment Volume', 'Abalone', 'Pima Native American Diabetes' , 'Wisconsin Breast Cancer Diagnostic', 'Portugal Wine Quality' , 'Human Activity Recognition', 'Adult Income']
extFolderNames = ['cali', 'air', 'fb' , 'aba', 'diabetes', 'cancer', 'wine', 'HAR', 'income']

risingMetric = ['r2', 'accuracy', 'precision', 'recall', 'f1','buildTime']
fallingMetric = ['rmse', 'mse', 'mae', 'oob']


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

    return myTitle

def resetStorage():
    print(f'\nResetting storage...\n')
    deleteAllDirs()
    makeAllDirs()
    print(f'\nStorage has been reset...\n')

def makeAllDirs():
    print(f'\nEnsuring all necessary folders for data storage exist...\n')

    for ending in myPaths:
        parentDir = os.path.abspath(os.path.join(myDir, '..', ending))

        RawDataDirName = os.path.abspath(os.path.join(parentDir,'RawData'))
        aggDirName = os.path.abspath(os.path.join(parentDir, 'AggData'))
        graphDirName = os.path.abspath(os.path.join(parentDir,'Graphs'))
        tablesDirName = os.path.abspath(os.path.join(parentDir,'Tables'))

        trfRaw = os.path.abspath(os.path.join(RawDataDirName, 'TRF'))
        rhfRaw = os.path.abspath(os.path.join(RawDataDirName, 'RHF'))

        trfGraphs = os.path.abspath(os.path.join(graphDirName, 'TRF'))
        rhfGraphs = os.path.abspath(os.path.join(graphDirName, 'RHF'))

        trfTables = os.path.abspath(os.path.join(tablesDirName,'TRF'))
        rhfTables = os.path.abspath(os.path.join(tablesDirName,'RHF'))

        makeTESTPATHDir(parentDir)
        makeAggDir(aggDirName)
        makeGraphsDir(graphDirName)
        makeRawDataDir(RawDataDirName)
        makeTablesDir(tablesDirName)

        # make subdirs for raw data
        makeSubDirs(RawDataDirName, workingModels)
        makeSubDirs(trfRaw, allDatasets)
        makeSubDirs(rhfRaw, allDatasets)

        # make subdirs for agg data
        makeSubDirs(aggDirName, workingModels)

        # make subdirs for graphs 
        makeSubDirs(graphDirName, workingModels)
        makeSubDirs(trfGraphs, allDatasets)
        makeSubDirs(rhfGraphs, allDatasets)

        # make subdirs for tables 
        makeSubDirs(tablesDirName, workingModels)
        makeSubDirs(trfTables, allDatasets)
        makeSubDirs(rhfTables, allDatasets)
    print(f'\nAll necessary folders for data storage exist...\n')
    

def deleteAllDirs():
    print(f'Deleting all previously existing storage files...\n')

    for ending in myPaths:
        parentDir = os.path.abspath(os.path.join(myDir, '..', ending))
        if os.path.exists(parentDir):
            shutil.rmtree(parentDir)
    
    print(f'All previously existing storage has been deleted...')

def makeTESTPATHDir(TESTPATHPath):
    print(f'Checking if {TESTPATHPath} exists...')
    if not os.path.exists(TESTPATHPath):
        print(f'{TESTPATHPath} does not exists, making one...')
        os.makedirs(TESTPATHPath)
    print(f'{TESTPATHPath} has been created...\n')

def makeAggDir(aggDirName):
    print(f'Checking if {aggDirName} exists...')
    if not os.path.exists(aggDirName):
        print(f'{aggDirName} does not exists, making one...')
        os.makedirs(aggDirName)
    print(f'{aggDirName} has been created...\n')

def makeGraphsDir(graphDirName):
    print(f'Checking if {graphDirName} exists...')
    if not os.path.exists(graphDirName):
        print(f'{graphDirName} does not exists, making one...')
        os.makedirs(graphDirName)
    print(f'{graphDirName} has been created...\n')

def makeRawDataDir(RawDataDir):
    print(f'Checking if {RawDataDir} exists...')
    if not os.path.exists(RawDataDir):
        print(f'{RawDataDir} does not exists, making one...')
        os.makedirs(RawDataDir)
    print(f'{RawDataDir} has been created...\n')

def makeTablesDir(tablesDir):
    print(f'Checking if {tablesDir} exists...')
    if not os.path.exists(tablesDir):
        print(f'{tablesDir} does not exists, making one...')
        os.makedirs(tablesDir)
    print(f'{tablesDir} has been created...\n')

def makeSubDirs(TESTPATHDir, subDirList:list):
    for name in subDirList:
        subDir = os.path.abspath(os.path.join(TESTPATHDir, name))
        print(f'Checking if {subDir} exists...')
        if not os.path.exists(subDir):
            print(f'{subDir} does not exists, making one...')
            os.makedirs(subDir)
        print(f'{subDir} has been created...\n')