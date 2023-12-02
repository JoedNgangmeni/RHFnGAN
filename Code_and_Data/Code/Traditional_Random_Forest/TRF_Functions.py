import numpy as np
import pandas as pd
import math, sklearn, os, random
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc, roc_auc_score

from sklearn.datasets import fetch_california_housing
from ucimlrepo import fetch_ucirepo
from scipy.stats import pearsonr


# Obtain current date for naming convention
from datetime import datetime
current_date = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")

# Setting up file locations
base_dir = os.path.dirname(os.path.abspath(__file__))  # Gets the directory where your script is
outpDir = "Output"
specOut =""


def setupData(myDatasets):
    for dataset in myDatasets:
        if dataset == 'cali':
            specOut = "caliOutput"
            # https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html
            data = sklearn.datasets.fetch_california_housing(download_if_missing=True, as_frame=True)
            X = data.data
            y = data.target

        elif dataset == 'air':
            specOut = "airOutput"
            # https://archive.ics.uci.edu/dataset/360/air+quality
            data = fetch_ucirepo(id=360) 
            temp = data.data.features.drop('PT08.S3(NOx)', axis =1)
            temp = temp.drop('Time', axis =1)
            temp = temp.drop('Date', axis =1)
            temp = temp.drop('NO2(GT)', axis =1)
            temp = temp.drop('PT08.S4(NO2)', axis =1)
            y = data.data.features['NOx(GT)'] 
            X = temp.drop('NOx(GT)', axis =1)

        elif dataset == 'fb':
            specOut = "fbOutput"
            # https://archive.ics.uci.edu/dataset/363/facebook+comment+volume+dataset
            pathToFile = os.path.join(base_dir, '../../Datasets/fb_git_repo/dataset/Training/')
            trainingVariants =['Features_Variant_1.csv', 'Features_Variant_2.csv', 'Features_Variant_3.csv', 'Features_Variant_4.csv', 'Features_Variant_5.csv']
            fileName = random.choice(trainingVariants)
            data = pd.read_csv(pathToFile+ fileName)
            X = data.iloc[:,:-1]
            y = data.iloc[:,-1]
            # y = y.values.ravel()
        
        elif dataset =='aba':
            specOut = "abaOutput"
            # https://archive.ics.uci.edu/dataset/1/abalone
            data = fetch_ucirepo(id=1) 
            data['data']['features'].loc[:, 'Sex'] = data['data']['features']['Sex'].astype('category')

            # Create one-hot encoding
            one_hot_encoded = pd.get_dummies(data['data']['features']['Sex'], prefix='Sex')

            # Concatenate the one-hot encoded columns with the original DataFrame
            data['data']['features'] = pd.concat([data['data']['features'], one_hot_encoded], axis=1)

            # Drop the original 'Sex' column
            data['data']['features'] = data['data']['features'].drop('Sex', axis=1)

            # # data (as pandas dataframes) 
            X = data.data.features 
            y = data.data.targets 
            y = y.values.ravel()
        
        if dataset == 'income':
            specOut = "incomeOutput"
            # https://archive.ics.uci.edu/dataset/2/adult
            # fetch dataset 
            data = fetch_ucirepo(id=2) 
            
            for nonNumFeat in data.data.features.columns:
                if (data.data.features.dtypes[nonNumFeat] != np.int64) and (data.data.features.dtypes[nonNumFeat] != np.float64):
                    # Create one-hot encoding
                    one_hot_encoded = pd.get_dummies(data.data.features[nonNumFeat], prefix=nonNumFeat)

                    # Concatenate the one-hot encoded columns with the original DataFrame
                    data.data.features = pd.concat([data.data.features, one_hot_encoded], axis=1)

                    # Drop the original column
                    data.data.features = data.data.features.drop(nonNumFeat, axis=1)

            # data (as pandas dataframes) 
            X = data.data.features 
            y = data.data.targets 
            y = y.values.ravel()  

            y[y == '<=50K.'] = '<=50K' 
            y[y == '>50K.'] = '>50K' 
            y[y == '<=50K'] = 'NO' 
            y[y == '>50K'] = "YES" 

        elif dataset == 'diabetes': 
            specOut = "diabetesOutput"
            # https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database/?select=diabetes.csv
            pathToFile = os.path.join(base_dir, '../../Datasets/diabetes.csv')
            data = pd.read_csv(pathToFile)
            X = data.iloc[:,:-1]
            y = data.iloc[:,-1]
            # y = y.values.ravel()


        elif dataset == 'cancer':
            specOut = "cancerOutput"
            # https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic
            # fetch dataset 
            data = fetch_ucirepo(id=17) 
            
            # data (as pandas dataframes) 
            X = data.data.features 
            y = data.data.targets 
            y = y.values.ravel()

        elif dataset == 'wine':
            specOut = "wineOutput"
            # https://archive.ics.uci.edu/dataset/186/wine+quality
  
            # fetch dataset     
            data = fetch_ucirepo(id=186) 
            
            # data (as pandas dataframes) 
            X = data.data.features 
            y = data.data.targets 
            y = y.values.ravel()
            
        elif dataset == 'HAR':
            specOut = "harOutput"
            # https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones
            data = pd.read_csv(os.path.join(base_dir, '../../Datasets/hartrain.csv') , delimiter = ',')
            X = data.iloc[:,:-1]
            y = data.iloc[:,-1]




for numEstimators in N_ESTIMATORS:
    for depth in DEPTH:
        runNumber = 1
        while (runNumber < MAX_RUNS + 1):
            # print(depth)
            # print(f'{numEstimators}est_{depth}deep_RF')

            output_path = os.path.join(base_dir, outpDir, specOut)
            # print(output_path)
            saveHere = os.path.join(output_path, f'{numEstimators}est_{depth}deep_{dataset}_RF')


            # add header to data file
            if runNumber == 1:
                with open(saveHere, 'a') as output_file:
                    output_file.write(f"r2\trmse\tmse\toob\tmae\n")    


            # Split the data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

            # Initialize the random forest
            rf = RandomForestRegressor(n_estimators=numEstimators, max_depth=depth, max_features='sqrt', bootstrap=True, oob_score=True)

            # Train the model
            rf.fit(X_train, y_train)

            # Make predictions on the test set
            y_pred = rf.predict(X_test)

            # Calculate R^2 score
            r2 = r2_score(y_test, y_pred)

            # Calculate MSE
            mse = mean_squared_error(y_test, y_pred)

            # Calculate RMSE
            rmse = math.sqrt(mse)

            # Calculate MAE
            mae = mean_absolute_error(y_test, y_pred)

            # Calculate OOB
            oob = 1 - rf.oob_score_

            with open(saveHere, 'a') as output_file:
                output_file.write(f"{r2}\t{rmse}\t{mse}\t{oob}\t{mae}\n")

            runNumber +=1

elif dataset in classificationDatasets:


for numEstimators in N_ESTIMATORS:
    for depth in DEPTH:
        runNumber = 1
        while (runNumber < MAX_RUNS + 1):
            # print(depth)
            # print(f'{numEstimators}est_{depth}deep_RF')

            output_path = os.path.join(base_dir, outpDir, specOut)
            # print(output_path)
            saveHere = os.path.join(output_path, f'{numEstimators}est_{depth}deep_{dataset}_RF')

            # add header to data file
            if runNumber == 1:
                with open(saveHere, 'a') as output_file:
                    # output_file.write(f"accuracy\tprecision\trecall\tf1\tfpr\ttpr\tthresholds\n")    
                    output_file.write(f"accuracy\tprecision\trecall\tf1\tconfMatrxVars\n")    


            # Split the data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

            # Initialize the random forest
            rf = RandomForestClassifier(n_estimators=numEstimators, max_depth=depth, max_features='sqrt', bootstrap=True, oob_score=True)

            # Train the model
            rf.fit(X_train, y_train)

            # Make predictions on the test set
            y_pred = rf.predict(X_test)

            # measure Accuracy 
            accuracy = accuracy_score(y_test, y_pred)

            # measure precision 
            precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)

            # measure recall 
            recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)

            # measure f1 
            f1 = f1_score(y_test, y_pred, average='weighted' , zero_division=0)

            # Get predicted probabilities for the positive class
            y_probs = rf.predict_proba(X_test)[:, 1]


            # Compute ROC curve and AUC
            # if dataset == "income":
            #     fpr, tpr, thresholds = roc_curve(y_test, y_probs, pos_label='YES')
            #     roc_auc = auc(fpr, tpr)

            # elif dataset == "cancer":
            #     fpr, tpr, thresholds = roc_curve(y_test, y_probs, pos_label='M')
            #     roc_auc = auc(fpr, tpr)

            # elif dataset == "wine":
            #     y_probs = rf.predict_proba(X_test)
            #     roc_auc = roc_auc_score(y_test, y_probs, multi_class='ovr')


            # else:
            #     fpr, tpr, thresholds = roc_curve(y_test, y_probs)
            #     roc_auc = auc(fpr, tpr)
                

            # thresholds_str = ",".join(map(str, thresholds))

            # create confusion matrix
            conf_matrix = confusion_matrix(y_test, y_pred)
            # print(conf_matrix.shape)
        

            # Save data
            with open(saveHere, 'a') as output_file:
                # output_file.write(f"{accuracy}\t{precision}\t{recall}\t{f1}\t{fpr}\t{tpr}\t{thresholds_str}\n")
                output_file.write(f"{accuracy}\t{precision}\t{recall}\t{f1}\t")
                for row in conf_matrix:
                    output_file.write(",".join(map(str, row)))
                output_file.write("\n")
            runNumber +=1



            


            



        




# print(f'R^2 Score: {r2}')
# print(f'rmse: {rmse}')
# print(f'mse: {mse}')
# print(f'OOB Score: {oob}')
# print(f'mae: {mae}')