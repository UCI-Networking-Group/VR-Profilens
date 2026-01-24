#model
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from scipy.stats import randint
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd
# Create a series containing feature importances from the model and feature names from the training data
from textwrap import wrap
from matplotlib.lines import Line2D
from sklearn.model_selection import RandomizedSearchCV, cross_val_score
from sklearn.svm import SVC
from scipy.stats import randint, uniform
from sklearn.model_selection import GridSearchCV
#lightGBM
import lightgbm as lgb
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, cross_val_score, KFold
from sklearn.model_selection import cross_val_predict
from collections import Counter


#functions for Producing final labels
#divide each index based on user
def index_dev(arr):
  arr=np.array(arr)
  ind = np.where(arr[:-1] != arr[1:])[0] + 1
  return ind
  
def divide_pred(arr,points):
  new_arr = []
  start = 0
  for point in points:
      new_arr.append(arr[start:point])
      start = point

  new_arr.append(np.array(arr[start:]))
  #new_arr=arr.tolist()
  return new_arr

#calculate final label via max voting
def final_label(new_arr,n):
  #n=number of index
  label=np.zeros(n+1)
  for i in range(n+1):
    counts = np.bincount(new_arr[i].astype(int))
    label[i] = np.argmax(counts)
    #print(counts,label[i])
  return label

def final_label_r(new_arr, n):
    label = np.zeros(n + 1)
    for i in range(n + 1):
        most_common = Counter(new_arr[i]).most_common(1)[0][0]
        label[i] = most_common
    return label

def final_Block(new_arr,n):
  #n=number of index
  label=np.zeros(n+1)
  for i in range(n+1):
    counts = np.bincount(new_arr[i])
    label[i] = np.argmax(counts)
    #print(counts,label[i])
  return label
  
#Final block
def final_Block(new_arr,n):
  #n=number of index
  label=np.zeros(n+1)
  for i in range(n+1):
    counts = np.bincount(new_arr[i])
    label[i] = np.argmax(counts)
    #print(counts,label[i])
  return label
#Choose best RF model
def RF_tuning(X_train,y_train,cv):
    param_dist = {'n_estimators': randint(50,200),
              'max_depth': randint(1,20)}

# Create a random forest classifier
    rf = RandomForestClassifier()

# Use random search to find the best hyperparameters
    rand_search = RandomizedSearchCV(rf,
                                 param_distributions = param_dist,
                                 n_iter=5,
                                 cv=cv)

# Fit the random search object to the data
    rand_search.fit(X_train, y_train)
    # Create a variable for the best model
    best_rf = rand_search.best_estimator_

    # Print the best hyperparameters
    print('Best hyperparameters:',  rand_search.best_params_)
    # Calculate cross-validation scores
    cv_results = cross_val_score(best_rf, X_train, y_train, cv=cv, scoring='accuracy')
    cv_results_f1 = cross_val_score(best_rf, X_train, y_train, cv=cv, scoring='f1_macro')

    # Calculate and print the average accuracy across all folds
    avg_accuracy = cv_results.mean()
    avg_F1 = cv_results_f1.mean()
    print("Average Accuracy:", avg_accuracy)
    print("Average F1:", avg_F1)
    return best_rf, avg_accuracy, avg_F1

# Choose XGB model
def XB_tuning(X_train, y_train, cv):
    # Define a grid of hyperparameters to search
    param = {
        'n_estimators': [50, 100, 150, 200],  # Number of trees
        'max_depth': [3, 5, 10, 15, 20],      # Maximum depth of trees
        'learning_rate': [0.1, 0.01],          # Learning rate
        # 'min_child_weight': [1, 2, 3],      # Minimum sum of instance weight needed in a child
    }

    xgb_model = xgb.XGBClassifier()

    # Create the GridSearchCV object
    grid_search = GridSearchCV(estimator=xgb_model, param_grid=param, scoring='accuracy', cv=cv)

    # Fit the grid search to the data
    grid_search.fit(X_train, y_train)

    # Get the best hyperparameters
    best_params = grid_search.best_params_
    print("Best Hyperparameters:", best_params)

    # Train the XGBoost model with the best hyperparameters
    best_xgb_model = xgb.XGBClassifier(**best_params)

    # Calculate cross-validation scores
    cv_results = cross_val_score(best_xgb_model, X_train, y_train, cv=cv, scoring='accuracy')
    cv_results_f1 = cross_val_score(best_xgb_model, X_train, y_train, cv=cv, scoring='f1_macro')

    # Calculate and print the average accuracy across all folds
    avg_accuracy = cv_results.mean()
    avg_F1 = cv_results_f1.mean()
    print("Average Accuracy:", avg_accuracy)
    print("Average F1:", avg_F1)

    return best_xgb_model, avg_accuracy, avg_F1


# Function to tune SVM model
def SVM_tuning(X_train, y_train, cv):
    # Define the parameter distribution
    param_dist = {
        'C': uniform(0.1, 10),        # Regularization parameter
        'kernel': ['linear', 'rbf', 'poly'],  # Kernel type
        'gamma': ['scale', 'auto'],   # Kernel coefficient for 'rbf' and 'poly'
        'degree': randint(2, 5)       # Degree for polynomial kernel (only used if kernel='poly')
    }

    # Create an SVM classifier
    svm = SVC()

    # RandomizedSearchCV to find the best hyperparameters
    rand_search = RandomizedSearchCV(svm,
                                     param_distributions=param_dist,
                                     n_iter=10,  # Number of parameter settings to sample
                                     cv=cv,      # Cross-validation folds
                                     verbose=1)

    # Fit the random search to the training data
    rand_search.fit(X_train, y_train)

    # Get the best model
    best_svm = rand_search.best_estimator_

    # Print the best hyperparameters
    print('Best hyperparameters:', rand_search.best_params_)

    # Cross-validation on the best model
    cv_results_accuracy = cross_val_score(best_svm, X_train, y_train, cv=cv, scoring='accuracy')
    cv_results_f1 = cross_val_score(best_svm, X_train, y_train, cv=cv, scoring='f1_macro')

    # Calculate and print the average scores across folds
    avg_accuracy = cv_results_accuracy.mean()
    avg_f1 = cv_results_f1.mean()
    print("Average Accuracy:", avg_accuracy)
    print("Average F1 Macro:", avg_f1)

    return best_svm, avg_accuracy, avg_f1
    


# LightGBM tuning for simpler datasets
def LGB_tuning(X_train, y_train, cv):
    # Simplified hyperparameter grid
    
    param = {
        'n_estimators': [50, 80, 100, 150],        # Fewer trees
        'max_depth': [3, 5, 8,10],              # Shallow trees to avoid overfitting
        'learning_rate': [0.1, .05],           # Default learning rate
        'num_leaves': [15, 31, 50],           # Reduced leaf nodes
        'min_child_samples': [20],        # Control leaf overfitting
    }
    

    lgb_model = lgb.LGBMClassifier(device='gpu',
        gpu_platform_id=0,      # Optional, change if multiple GPUs
        gpu_device_id=0,random_state=42)        # Optional)

    # Create the GridSearchCV object
    grid_search = GridSearchCV(estimator=lgb_model, param_grid=param, scoring='accuracy', cv=cv, verbose=1)

    # Fit the grid search to the data
    grid_search.fit(X_train, y_train)

    # Get the best hyperparameters
    best_params = grid_search.best_params_
    print("Best Hyperparameters:", best_params)

    # Train the LightGBM model with the best hyperparameters
    best_lgb_model = lgb.LGBMClassifier(**best_params, device='gpu', random_state=None)

    # Calculate cross-validation scores
    cv_results = cross_val_score(best_lgb_model, X_train, y_train, cv=cv, scoring='accuracy')
    cv_results_f1 = cross_val_score(best_lgb_model, X_train, y_train, cv=cv, scoring='f1_macro')

    # Calculate and print the average accuracy and F1 score across all folds
    avg_accuracy = cv_results.mean()
    avg_F1 = cv_results_f1.mean()
    print(f"Average Accuracy: {avg_accuracy:.4f}")
    print(f"Average F1 Score: {avg_F1:.4f}")

    return best_lgb_model, avg_accuracy, avg_F1


def RF_Reg(X_train, y_train, X_test, y_test, tol, sd, y_sd):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    print('random_forest_regressor')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # round predictions and true values
    y_pred = np.round(y_pred).astype(int)
    y_test = np.round(y_test).astype(int)

    # apply your custom labeling method
    new_label = final_label_r(np.array(divide_pred(y_pred, sd), dtype=object), len(sd))
    true_labels = final_label_r(np.array(divide_pred(y_test, sd), dtype=object), len(sd))
    print('true and new label', true_labels, new_label)

    correct = np.abs(new_label - true_labels) <= tol
    accuracy = correct.mean()

    return model, y_pred, accuracy

def LR_train(X_train, y_train,X_test,y_test,tol,sd,y_sd):
    model = LinearRegression()
    print('linear_regression')
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    #mae = mean_absolute_error(y_train, y_pred)
    #r2 = r2_score(y_train, y_pred)
    #print(abs(y_pred-y_test))
    # mark as correct if absolute error ≤ tol
    #correct = np.abs(y_pred - y_test) <= tol
    
    #for per sample accuracy calculation
    y_pred=np.round(y_pred).astype(int)
    y_test=np.round(y_test).astype(int)
    new_label=final_label_r(np.array(divide_pred(y_pred,sd),dtype=object),len(sd))
    true_labels=final_label_r(np.array(divide_pred(y_test,sd),dtype=object),len(sd))
    print('true and new label',true_labels,new_label)
    correct = np.abs(new_label - true_labels) <= tol

    #mean accuracy
    accuracy = correct.mean()
    #print('accuracy',accuracy)

    return model, y_pred, accuracy


