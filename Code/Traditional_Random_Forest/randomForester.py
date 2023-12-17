import math, os
import pandas as pd, numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


def growRegressor(NUMTREES: int, DEPTH: int, X: pd.DataFrame , y: np.ndarray):
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)

    # Initialize the random forest
    rf = RandomForestRegressor(n_estimators=NUMTREES, max_depth=DEPTH, max_features='sqrt', bootstrap=True, oob_score=True)

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

    return oob, r2, rmse, mse, mae 


def growClassifier(NUMTREES: int, DEPTH: int, X: pd.DataFrame , y: np.ndarray):
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)

    # Initialize the random forest
    rf = RandomForestClassifier(n_estimators=NUMTREES, max_depth=DEPTH, max_features='sqrt', bootstrap=True, oob_score=True)

    # Train the model
    rf.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = rf.predict(X_test)

    # measure Accuracy 
    accuracy = accuracy_score(y_test, y_pred)

    # measure precision 
    precision = precision_score(y_test, y_pred, average="micro", zero_division=0)

    # measure recall 
    recall = recall_score(y_test, y_pred, average='micro', zero_division=0)

    # Calculate OOB
    oob = 1 - rf.oob_score_

    # measure f1 
    f1 = f1_score(y_test, y_pred, average='micro' , zero_division=0)

    # create confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    # flatConfMatrix = conf_matrix.ravel()

    return oob, f1, accuracy, precision, recall, conf_matrix
