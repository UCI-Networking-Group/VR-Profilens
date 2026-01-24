##initialize the required libraries
import numpy as np
import pandas as pd
import os
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import argparse
from Input_data import Final_feature
import os
from util import call_da, data_clear, run_attribute, save_file_lgb,save_file
parser = argparse.ArgumentParser()

#Sensor data
parser.add_argument('--SG', type=str, help='The sensor group-BM/FE/EG/HJ/BM_FE/BM_FE_EG',default='BM')
#dtype
parser.add_argument('--dtype', type=str, help='hand or other',default='other')
#Call App ids
parser.add_argument('--app_id', type=int, nargs='+', help='List of app IDs', default=[])

#Call tolerance score
parser.add_argument('--tol', type=int, nargs='+', help='List of app IDs', default=[])

#call attribute targets
parser.add_argument('--target_final', type=str, nargs='+', help='List of target attributes',
                    default=[])

#call for mode (RF/XGB/SVM or others)
parser.add_argument('--mode', type=str, help='RF/XGB/SVM/LGB/Reg',default='RF')

#call for mode2 for Regression
parser.add_argument('--mode2', type=str, help='LR/RFR',default='RFR')

args = parser.parse_args()

#initialization
SG=args.SG #Define Sensor Group
dtype=args.dtype #Define Sensor Group
app_id=args.app_id
target_final=args.target_final
mode=args.mode
mode2=args.mode2
tol=args.tol
print("the attributes we are evaluating", target_final)

#initialize arrays/lists
multiple_round_results = []

#call data
dataD=Final_feature(SG)
data=dataD[1]
#print("data shape", data.shape)
print("data column", data.columns)

for k in range(len(target_final)):
    #data.drop(columns=target_final,inplace=True)
    drop_target = list(set(target_final) - set([target_final[k]]))
    print('target, drop:',target_final,drop_target)
    data_pass=data_clear(data,target_final)
    final_models,result_list_acc,result_list_f1, X_train_apps = run_attribute(mode,app_id,target_final[k],data_pass,mode2,tol[k],dtype,drop_target=None)
    print('F1 score',result_list_f1)
    #result_list_f1 = run_cross(mode,app_id,target_final[k],data_pass,drop_target=None)
    multiple_round_results.append(result_list_f1)
    #save_file(target_final[k], final_models, X_train_apps)  # Pass individual attribute instead of the entire list
    
    file_path1 = ''+mode+'/'+SG+'/Acc/accuracy_'+target_final[k]+'.txt'
    file_path2 = '/'+mode+'/'+SG+'/F1/F1_'+target_final[k]+'.txt'
    with open(file_path1, 'w') as file:
             file.write(str(result_list_acc))
    with open(file_path2, 'w') as file:
             file.write(str(result_list_f1))
    save_file_lgb(target_final[k], final_models, X_train_apps,SG,app_id,mode)



