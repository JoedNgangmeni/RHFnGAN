import sys, os
sys.path.append("..")

import outputDataAggregator as agg
import dataVisualizer as vis
import myStructure as my
import adabooster as myBoost


base_dir = os.path.dirname(os.path.abspath(__file__))  # Gets the directory where your script is

focusParentDir = 'RESULTS'
focusDataDir = 'ADA'

rawTRFDataPath = os.path.abspath(os.path.join(base_dir, '..', '..', f'{focusParentDir}', 'RawData', f'{focusDataDir}'))
aggTRFDataPath = os.path.abspath(os.path.join(base_dir, '..', '..', f'{focusParentDir}', 'AggData',f'{focusDataDir}'))
TRFGraphsPath = os.path.abspath(os.path.join(base_dir, '..', '..', f'{focusParentDir}', 'Graphs',f'{focusDataDir}'))
TRFTablesPath = os.path.abspath(os.path.join(base_dir, '..', '..', f'{focusParentDir}', 'Tables',f'{focusDataDir}'))

# Get storage ready to store data 
my.resetStorage('RESULTS', 'ADA')

MAX_RUNS = int(input("Set the number of runs per permutation: "))

start_est = int(input("Set min number of trees: "))
final_est = int(input("Set max number of trees: "))
est_step = int(input("Set num tree step size: ")) 
ESTNUM = list(range(start_est, final_est + 1, est_step))

start_learn = int(input("Set starting learning rate: "))
final_learn = int(input("Set max learning rate: "))
learn_step = int(input("Set learning rate step size: "))
LEARNING_RATE = list(range(start_learn, final_learn + 1, learn_step))

topNUM = my.topHowMany()