import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
import re

#call models
from model import RF_tuning, XB_tuning, SVM_tuning, LGB_tuning, LR_train, RF_Reg
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import RandomizedSearchCV, cross_val_score, KFold

from textwrap import wrap
import pandas as pd
import matplotlib.pyplot as plt

#Function for Preprocessing Body Motion Data
def change_nameH(datar):
  datar.columns=datar.columns.str.replace('_1', ' Headset')
  datar.columns=datar.columns.str.replace('_2', ' Left Controller')
  datar.columns=datar.columns.str.replace('_3', ' Right Controller')
  datar.columns=datar.columns.str.replace('Pos0', 'Position.x')
  datar.columns=datar.columns.str.replace('Pos1', 'Position.y')
  datar.columns=datar.columns.str.replace('Pos2', 'Position.z')
  datar.columns=datar.columns.str.replace('_max', ' Max')
  datar.columns=datar.columns.str.replace('_min', ' Min')
  datar.columns=datar.columns.str.replace('_mean', ' Mean')
  datar.columns=datar.columns.str.replace('_median', ' Median')
  datar.columns=datar.columns.str.replace('_std', ' Std')
  datar['user_id'] = datar['user_id'].replace(21, 9)
  #datar['user_id'] = datar['user_id'].replace(17, 9)
  return datar
  
#Function for Processing Eye Gaze Data
def change_nameE(datar):
  datar.columns=datar.columns.str.replace('_x', '_L')
  datar.columns=datar.columns.str.replace('_y', '_R')
  datar['user_id'] = datar['user_id'].replace(21, 9)
  return datar

#Change name of final eyedata (after augmentation)
def change_nameEye(datar):
  datar.columns=datar.columns.str.replace('_LR', ' Left Right')
  datar.columns=datar.columns.str.replace('_L', ' Left')
  datar.columns=datar.columns.str.replace('_R', ' Right')
  #datar['user_id'] = datar['user_id'].replace(11.0, 7.0)
  #datar['user_id'] = datar['user_id'].replace(16, 7)
  #datar['user_id'] = datar['user_id'].replace(21, 9)
  datar.columns=datar.columns.str.replace('_max', ' Max')
  datar.columns=datar.columns.str.replace('_min', ' Min')
  datar.columns=datar.columns.str.replace('_mean', ' Mean')
  datar.columns=datar.columns.str.replace('_median', ' Median')
  datar.columns=datar.columns.str.replace('_std', ' Std')
  return datar
#Feature Augmentation of Eye data
def RLeft_Right(d, b=1):
    metrics = [
        'Quaty_mean', 'Quatx_mean', 'Quatw_mean',
        'Quaty_min', 'Quatx_min', 'Quatw_min',
        'Quaty_max', 'Quatx_max', 'Quatw_max',
        'Quaty_std', 'Quatx_std', 'Quatw_std',
        'Quaty_median', 'Quatx_median', 'Quatw_median'
    ]
    
    for metric in metrics:
        d[f'{metric}_LR'] = abs(d[f'{metric}_L']*b - d[f'{metric}_R']*b)
    
    return d
  
#Function for Preprocessing Facial Data
def change_nameFace(datar):
  for i in range(64,0,-1):
      s2='Element_'+str(i)+'_'
      s1='w'+str(i-1)+'_'
      #print(s1,s2)
      datar.columns=datar.columns.str.replace(s1,s2)
  #datar['user_id'] = datar['user_id'].replace(16, 7)
  datar['user_id'] = datar['user_id'].replace(21, 9)

  datar.columns=datar.columns.str.replace('_max', ' Max')
  datar.columns=datar.columns.str.replace('_min', ' Min')
  datar.columns=datar.columns.str.replace('_mean', ' Mean')
  datar.columns=datar.columns.str.replace('_med', ' Median')
  datar.columns=datar.columns.str.replace('_std', ' Std')
  return datar

def change_nameF(datar):
  for i in range(64,0,-1):
      s2='w'+str(i)+'_'
      s1='w'+str(i-1)+'_'
      #print(s1,s2)
      datar.columns=datar.columns.str.replace(s1,s2)
  #datar['user_id'] = datar['user_id'].replace(16, 7)
  datar['user_id'] = datar['user_id'].replace(21, 9)
  #datar['user_id'] = datar['user_id'].replace(11.0, 7.0)
  return datar
  
  
#Process Hand Data/Feature Engineering
def change_nameHand(datar):
  datar.columns=datar.columns.str.replace('_x', ' Left')
  datar.columns=datar.columns.str.replace('_y', ' Right')
  datar.columns=datar.columns.str.replace('BoneRotation', 'Rotation')
  datar.columns=datar.columns.str.replace('BonePosition', 'Position')
  # 1) bump the numeric suffixes
  for old, new in [(i-1, i) for i in range(26,0,-1)]:
      datar.columns = datar.columns.str.replace(
          rf"_([xyzw])_{old}\b",
          lambda m: f"_{m.group(1)}_{new}",
          regex=True
      )

  datar.columns=datar.columns.str.replace('_max', ' Max ')
  datar.columns=datar.columns.str.replace('_min', ' Min ')
  datar.columns=datar.columns.str.replace('_med', ' Med ')
  datar.columns=datar.columns.str.replace('_std', ' Std ')
  datar.columns=datar.columns.str.replace('_mean', ' Mean ')

  #datar['user_id'] = datar['user_id'].replace(16, 7)
  datar['user_id'] = datar['user_id'].replace(21, 9)
  datar=datar.sort_values(by='user_id')
  datar = datar.reset_index(drop=True)
  #datar['user_id'] = datar['user_id'].replace(11.0, 7.0)
  return datar

#eliminate specific types of features (here Headset Features)
def change_eliminate_Head(datar):
  datar.columns=datar.columns.str.replace('_1', '_H')
  datar.columns=datar.columns.str.replace('_2', '_CL')
  datar.columns=datar.columns.str.replace('_3', '_CR')
  datar.columns=datar.columns.str.replace('0', 'x')
  datar.columns=datar.columns.str.replace('1', 'y')
  datar.columns=datar.columns.str.replace('2', 'z')
  filtered_columns = datar.filter(like='_H', axis=1).columns
  datar = datar.drop(columns=filtered_columns)
  return datar

#eliminate specific types of features (here Right Hand Features)
def change_eliminate_RightHand(datar):
  filtered_columns = datar.filter(like='Right', axis=1).columns
  datar = datar.drop(columns=filtered_columns)
  return datar
  
#eliminate specific types of features (here Emotion Facial Expression Features)
def change_nameFacialEexpression(datar,f):
  for i in range(64,0,-1):
      s2='w'+str(i)+'_'
      s1='w'+str(i-1)+'_'
      #print(s1,s2)
      datar.columns=datar.columns.str.replace(s1,s2)
  datar=datar.drop(f,axis=1)
  return datar

#Pre-process the data
def data_preProcess(d_h,g_id,target):
  #identify the columns that contains all zero values
   d_h=d_h.fillna(0)
   drop_list=[]

   #for col in d_h.columns:
    #   if (d_h[col] == 0).all():
     #     drop_list.append(col)
     #  if (d_h[col] == 1).all():
      #    drop_list.append(col)

   d_h=d_h.drop(drop_list,axis=1)
   d_h=d_h.drop(['block_id'],axis=1)

   d_h=d_h.sort_values(by='user_id')
   d_h = d_h.reset_index(drop=True)

   if (target=='user_id'):         #select data according to game id
      d_h=d_h[d_h['game_id']== g_id]
      d_h=d_h.drop(['game_id'],axis=1)
   else:
      d_h=d_h[d_h['game_id']== g_id]
      d_h=d_h.drop(['game_id'],axis=1)
      #d_h=d_h.drop('user_id',axis=1)
      #d_h=d_h.drop('age',axis=1)
   return d_h
   
#divide the dataset into training and testing based on their round. For round-1: training data, round-2: test data

#Get d_train and d_test from a function
def dT(d_h,target,id1,id2):
  d_train=d_h[d_h['user_id'].isin(id1)]
  d_test=d_h[d_h['user_id'].isin(id2)]
  print(d_train.shape)

  #drop unnecessary columns
  d_train=d_train.drop(['user_id','round_id'],axis=1)
  d_test_sd=d_test.drop(['round_id',target],axis=1)
  d_test=d_test.drop(['user_id','round_id'],axis=1)
  return d_train, d_test, d_test_sd

def train_test(d_h,target,id1,id2):
  d_train,d_test,d_test_sd=dT(d_h,target,id1,id2)

  #train data
  y_train1 = np.array(d_train[target])
  X_train1= d_train.drop(target, axis = 1)

  #test data
  y_test1 = np.array(d_test[target])
  X_test1= d_test.drop(target, axis = 1)

  y_sd1=np.array(d_test_sd['user_id'])

  #calculate sd
  #sd=index_dev(y_sd)
  #label_sd=divide_pred(y_sd,sd)
  return X_train1, y_train1, X_test1, y_test1,y_sd1

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

def concatenate_arrays(new_preds, new_pred1):
    result = []
    for a1, a2 in zip(new_preds, new_pred1):
        if isinstance(a1, np.ndarray) and isinstance(a2, np.ndarray):
            new_array = np.concatenate((a1, a2), axis=None)
            result.append(new_array)
        elif isinstance(a1, np.ndarray):
            result.append(a1.astype(np.object))  # Convert to object dtype
        elif isinstance(a2, np.ndarray):
            result.append(a2.astype(np.object))  # Convert to object dtype
    return result

def Emotion_units():
        #Emotion Units
    #smile
    f_smile=['user_id', 'game_id', 'round_id', 'device_id', 'block_id','w33_max',
    'w33_min','w33_mean','w33_std','w33_med','w34_max',
    'w34_min','w34_mean','w34_std','w34_med','w6_max','w6_min','w6_mean','w6_std','w6_med','w5_max','w5_min','w5_mean','w5_std','w5_med']
    #surprise
    f_surprise=['user_id', 'game_id', 'round_id', 'device_id', 'block_id','w58_max','w58_min','w58_mean','w58_std','w58_med','w59_max','w59_min',
                'w59_mean','w59_std','w59_med','w60_max','w60_min','w60_mean','w60_std','w60_med','w61_max','w61_min','w61_mean','w61_std','w61_med',
    'w23_max','w23_min','w23_mean','w23_std','w23_med','w24_max','w24_min','w24_mean','w24_std','w24_med','w25_max','w25_min',
    'w25_mean','w25_std','w25_med']   #1 + 2 + 5 + 26. #(23,24)+ (58,59)+ (60,61)+ (25)
    #Anger=4 + 5 + 7 + 23  (1,2)+ (60,61)+ (29,30)+ (49,50)
    f_anger=['user_id', 'game_id', 'round_id', 'device_id', 'block_id','w1_min','w1_mean','w1_std','w1_med','w1_max',
    'w2_min','w2_mean','w2_std','w2_med','w60_max','w60_min','w60_mean','w60_std','w60_med','w61_max','w61_min','w61_mean','w61_std','w61_med',
    'w29_max','w29_min','w29_mean','w29_std','w29_med',
    'w30_max','w30_min','w30_mean','w30_std','w30_med','w49_max','w49_min','w49_mean','w49_std','w49_med','w50_max','w50_min','w50_mean','w50_std','w50_med']
    #Sadness 1 + 4 + 15; (23,24)+ (1,2)+ (31,32)
    f_sadness=['user_id', 'game_id', 'round_id', 'device_id', 'block_id','w1_max',
    'w1_min','w1_mean','w1_std','w1_med','w2_max','w2_min','w2_mean','w2_std','w2_med',
    'w23_max','w23_min','w23_mean','w23_std','w23_med','w24_max','w24_min','w24_mean','w24_std','w24_med',
    'w31_max','w31_min','w31_mean','w31_std','w31_med','w32_max','w32_min','w32_mean','w32_std','w32_med',
    ]
    f_fear=['user_id', 'game_id', 'round_id', 'device_id', 'block_id',
            'w23_max','w23_min','w23_mean','w23_std','w23_med','w24_max','w24_min','w24_mean','w24_std','w24_med',
            'w58_max','w58_min','w58_mean','w58_std','w58_med','w59_max','w59_min','w59_mean','w59_std','w59_med',
            'w1_min','w1_mean','w1_std','w1_med','w2_max','w2_min','w2_mean','w2_std','w2_med',
            'w60_max','w60_min','w60_mean','w60_std','w60_med','w61_max','w61_min','w61_mean','w61_std','w61_med',
            'w29_max','w29_min','w29_mean','w29_std','w29_med','w30_max','w30_min','w30_mean','w30_std','w30_med',
            'w43_max','w43_min','w43_mean','w43_std','w43_med','w44_max','w44_min','w44_mean','w44_std','w44_med',
            'w25_max','w25_min','w25_mean','w25_std','w25_med']

    f_disgust=['user_id', 'game_id', 'round_id', 'device_id', 'block_id','w31_max',
    'w31_min','w31_mean','w31_std','w31_med','w32_max','w32_min','w32_mean','w32_std','w32_med','w52_max','w52_min','w52_mean','w52_std','w52_med',
               'w53_max','w53_min','w53_mean','w53_std','w53_med', 'w56_max','w56_min','w56_mean','w56_std','w56_med',
               'w57_max','w57_min','w57_mean','w57_std','w57_med']
    f_contempt=['user_id', 'game_id', 'round_id', 'device_id', 'block_id','w33_max',
    'w33_min','w33_mean','w33_std','w33_med','w12_max','w12_min','w12_mean','w12_std','w12_med','w11_max','w11_min','w11_mean','w11_std','w11_med']

    f_imp=['w51_max',
    'w51_min','w51_mean','w51_std','w51_med','w28_max','w28_min','w28_mean','w28_std','w28_med','w13_max','w13_min','w13_mean','w13_std','w13_med',
           'w14_max','w14_min','w14_mean','w14_std','w14_med']


    #All
    common_words = set(f_smile).intersection(f_anger, f_sadness)
    all_words = set(f_smile + f_anger + f_sadness+f_surprise+f_fear+f_disgust)
    all_words1=set(f_smile + f_anger + f_sadness+f_surprise+f_fear+f_disgust+f_contempt+f_imp)
    words_to_keep=['user_id', 'game_id', 'round_id', 'device_id', 'block_id']
    f=list(all_words)
    f1=list(set(all_words1) - set(words_to_keep))
    emo=[f_smile,f_surprise,f_anger,f_disgust,f_fear,f_sadness, f]
    return emo, f1

#feature Engineering for face data/this will be more automated with more dataset
#take only the features based on emotion recognition/ smile=6+12
def Emotion(d,f):
  d=d[f]
  return d

#emotion process
def DataE(D,f):
  dE=[]
  for i in range(len(D)):
    d=D[i]
    dnew=Emotion(d,f)
    dE.append(dnew)
  D=dE
  return D

#app grouping
def app_groups_name():
     gname=['social', 'social','flight', 'flight', 'shoot', 'Arch', 'beat', 'IN', 'IN', 'Kwalk']
     return gname

def app_grouping(app_group):
    if (app_group=='social'):
        g=[12,15,18]
    elif (app_group=='flight'): #Flight Simulation
        g=[20,19,3]
    elif (app_group=='golf'): #Golfing
        g=[6]
    elif (app_group=='IN'): #Interactive navigation
        g=[2,9,10,16,17]
    elif (app_group=='KW'): #knuckle walking
        g=[7]
    elif (app_group=='Rhy'): #Rhythm
        g=[1]
    elif (app_group=='shoot'): #Shooting and archary
        g=[5,13,14]
    elif (app_group=='teleport'): #Teleportation
        g=[4,8]
    return g

#concatenate different apps data or user can just use one apps data, in that case len(g1) should be1

def f_data(g1,d1,target):
    M=len(g1)
    print(M)
    df_list = []
    if (M==1):
        d_h=data_preProcess(d1, g1[0],target)
    else:
        for i in range(M):
            df = data_preProcess(d1, g1[i],target)
            df = df.sample(frac=1/M)
            df_list.append(df)
        d_h = pd.concat(df_list, axis=0)
    d_h=d_h.sort_values('user_id')
    return d_h


#Functions for profiling
#Calculate BMI
def calculate_bmi(weight_kg, height_cm):
    """
    Calculate BMI (Body Mass Index) given weight in kilograms and height in meters.

    Parameters:
        weight_kg (float): Weight in kilograms.
        height_m (float): Height in meters.

    Returns:
        float: BMI value.
    """
    height_m = height_cm / 100
    bmi = weight_kg / (height_m ** 2)
    return bmi

#Call attributes
def call_da():
  da1 = pd.read_csv('../Survey/Collection_Result.csv',skiprows=1) #read attributes csv file
              
  return da1

def add_att(target,mode,da):
  #data.drop(columns=[target], inplace=True)
  da=call_da()
  id1=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,19]
  #based on classification task design and data point (example:age, BMI)
  
  if target=='Age':
    da['Age']=da['Age'].astype(float)
    da['Age'] = np.where(da['Age'] >= 30, 1, 0)

  elif target=='BMI':
    if mode!='Reg':
      da['BMI']=da['BMI'].astype(float)
      da['BMI'] = np.where(da['BMI'] > 24.9, 1, 0)

  id2=id1
  target_values = da[target].astype(float)
  return target_values,id1,id2

def add_targetV(target,df,mode):
    targetV,id1,id2=add_att(target,mode,df)
    df[target] = [targetV[i - 1] if i <= len(targetV) else None for i in df['user_id']]
    return df,targetV ,id1,id2


#Separate and map attribute labels
def map_label(target_array,id1,id2):
  train_labels = {i - 1: target_array[i-1] for i in id1}
  true_maps = {i - 1: target_array[i-1] for i in id2}

  train_labels = np.array(list(train_labels.values()))
  true_maps = np.array(list(true_maps.values()))

  print("Mapped values for id1:", train_labels)
  print("Mapped values for id2:", true_maps)
  return true_maps

def save_file_lgb(target1,final_models,X_train_app,SG,app_id,mode):
    # Define the list to store feature importance data
    feature_importance_final = []

    # Define group_id and app_id
    group_id = ['social', 'social','flight', 'flight', 'shoot', 'Arch', 'beat', 'IN', 'IN', 'Kwalk']

    # Loop over each group_id and app_id
    for i in range(len(group_id)):
        # Raw gain-based importances
        raw_gain = final_models[i].booster_.feature_importance(importance_type='gain')
        # Normalize to get values between 0 and 1
        gain_importance_normalized = raw_gain / raw_gain.sum()

        feature_importances = pd.Series(gain_importance_normalized
, index=X_train_app.columns).sort_values(ascending=False)
        feature_top = feature_importances.head(50)
        f2=feature_top.index
        f2=f2.str.replace('Pos0', 'Position.x')
        f2=f2.str.replace('Pos1', 'Position.y')
        f2=f2.str.replace('Pos2', 'Position.z')

        # Append group_id and app_id to feature_importance_final
        feature_importance_final.append("Group ID: " + group_id[i] + "\n")
        feature_importance_final.append("App ID: " + str(app_id[i]) + "\n")

        # Append feature names and their importance values to feature_importance_final
        for feature, importance in feature_top.items():
            feature_importance_final.append(f"{feature}: {importance}\n")

        # Plot feature importance
    # Specify the file path where you want to save the text file
    file_path = '/home/ijarin/VRprofile/results/'+mode+'/'+SG+'/FI/'+SG+'_' + target1 + '.txt'  # Modify the path as needed

     #Write the feature importance data to the text file
    with open(file_path, 'w') as file:
        for item in feature_importance_final:
           file.write(str(item))


           
def save_file(target1,final_models,X_train_app,SG,app_id,mode):
        # Define the list to store feature importance data
    feature_importance_final = []

    # Define group_id and app_id
    group_id = ['social', 'social','flight', 'flight', 'shoot', 'Arch', 'beat', 'IN', 'IN', 'Kwalk']

    # Loop over each group_id and app_id
    for i in range(len(group_id)):
        feature_importances = pd.Series(final_models[i].feature_importances_, index=X_train_app.columns).sort_values(ascending=False)
        feature_top = feature_importances.head(50)
        f2=feature_top.index
        f2=f2.str.replace('Pos0', 'Position.x')
        f2=f2.str.replace('Pos1', 'Position.y')
        f2=f2.str.replace('Pos2', 'Position.z')

        # Append group_id and app_id to feature_importance_final
        feature_importance_final.append("Group ID: " + group_id[i] + "\n")
        feature_importance_final.append("App ID: " + str(app_id[i]) + "\n")

        # Append feature names and their importance values to feature_importance_final
        for feature, importance in feature_top.items():
            feature_importance_final.append(f"{feature}: {importance}\n")
    # Specify the file path where you want to save the text file
    file_path = '../VRprofile/results/'+mode+'/'+SG+'/FI/'+SG+'_' + target1 + '.txt'  # Modify the path as needed

     #Write the feature importance data to the text file
    with open(file_path, 'w') as file:
        for item in feature_importance_final:
           file.write(str(item))

#Return final models
def run_attribute(mode,app_id,target,D,mode2,tol,dtype,drop_target=None):
    select='nemo'
    #dtype='hand'
    n_app=len(app_id)
    print(target)
    final_models=[]
    data1,att,id1,id2=add_targetV(target,D,mode)
    d1=len(id2)
    print('no of id',d1)
    final_labels=np.zeros((n_app,d1))
    label=np.zeros((n_app,d1))
    Acc_Gender=np.zeros(n_app)
    Acc_pass=np.zeros(n_app)
    F1_pass=np.zeros(n_app)
    #select data type:
    if select=='emo':
      fe=emo[6]
      fE=change_nameFeatures(fe)
      #print(fE)
      d=Emotion(data1,fE)
      d[target] = data1[target]
    else:
      d=data1
      
            #print(d)
   #calculate accuracy for each
    for i in range(n_app):
      print('The following app is',app_id[i])
      d_h=data_preProcess(d,app_id[i],target)
      #d_h.columns = [clean_column_name(col) for col in d_h.columns]
      X_train,y_train,X_test,y_test,y_sd=train_test(d_h,target,id1,id2)
      #X_train.columns = [clean_column_name(col) for col in X_train.columns]
      if drop_target:
        columns_to_drop = [col for col in drop_target if col in X_train.columns]
        if columns_to_drop:
            X_train1 = X_train.drop(columns=columns_to_drop, errors='ignore')
            X_test1 = X_test.drop(columns=columns_to_drop, errors='ignore')
            print(X_train1.columns)
      else:
           X_train1=X_train
           X_test1=X_test
      #print(X_train1.columns[0:20])

      #get data from each user pole
      sd=index_dev(y_sd)

      if (mode=='RF'):
          #model_RF = RandomForestClassifier()
          if dtype=='hand':
            print('handling Hand-data')
            model_RF = RandomForestClassifier()
            model_RF.fit(np.array(X_train1),y_train)
            feature_importances = model_RF.feature_importances_
            top_500_indices = feature_importances.argsort()[-400:][::-1]
            X_train1 = X_train1.iloc[:, top_500_indices]
            X_test1=X_test1.iloc[:, top_500_indices]
        #RF model Training Start
          print('train data size for RF', X_train1.shape)
          #model_RF = RandomForestClassifier()
          model_RF,Acc_pass[i],F1_pass[i]=RF_tuning(X_train1,y_train,5)
          model_RF.fit(np.array(X_train1),y_train)
          model=model_RF
      elif (mode=='XGB'):
          print('XGB')
          le = LabelEncoder()
          y_train = le.fit_transform(y_train)
          y_test=le.fit_transform(y_test)

          if dtype=='hand':
            print('handling Hand-data')
            model_XB = xgb.XGBClassifier()
            model_XB.fit(np.array(X_train1),y_train)
            feature_importances = model_XB.feature_importances_
            top_500_indices = feature_importances.argsort()[-500:][::-1]
            X_train1 = X_train1.iloc[:, top_500_indices]
            X_test1=X_test1.iloc[:, top_500_indices]
          print('train data size', X_train.shape)
          #model_XB = xgb.XGBClassifier()
          model_XB, Acc_pass[i],F1_pass[i]=XB_tuning(X_train,y_train,5)
          # Train the model on training data
          X_train1.columns = [clean_column_name(col) for col in X_train1.columns]
          model_XB.fit(np.array(X_train1),y_train)
          model=model_XB
          print(app_id[i])

      #for LGB
      elif (mode=='LGB'):
          print('LGB')
          if dtype=='hand':
            print('handling Hand-data')
            model_LGB = lgb.LGBMClassifier()
            model_LGB.fit(np.array(X_train1),y_train)
            feature_importances = model_LGB.feature_importances_
            top_500_indices = feature_importances.argsort()[-300:][::-1]
            X_train1 = X_train1.iloc[:, top_500_indices]
            X_test1=X_test1.iloc[:, top_500_indices]
          print('train data size', X_train1.shape)
          # Create an SVM classifier
          #model_LGB = lgb.LGBMClassifier()
          model_LGB,Acc_pass[i],F1_pass[i] = LGB_tuning(X_train1,y_train,5)
          # Train the classifier
          model_LGB.fit(X_train1, y_train)
          model=model_LGB
      #for SVM
      elif (mode=='Reg' and mode2=='LR'):
          n_splits=4
          count=0
          kf = KFold(n_splits, shuffle=False)
          id=np.array(id1)
          accuracy=np.zeros(n_splits)
          for fold, (train_idx, test_idx) in enumerate(kf.split(id), start=1):
            idt=id[train_idx]
            ids=id[test_idx]
            X_train_r,y_train_r,X_test_r,y_test_r,y_sd=train_test(d_h,target,idt,ids)
            sd=index_dev(y_sd)
            model_LR, y_pred,accuracy[count] = LR_train(X_train_r, y_train_r,X_test_r,y_test_r,tol,sd,y_sd)
            count=count+1
            print(idt,ids)
          #model_LR.fit(X_train1, y_train)
          model = model_LR
          Acc_Gender[i]=np.mean(accuracy)
          print('accuracy',Acc_Gender[i])

      elif (mode=='Reg' and mode2=='RFR'):
          n_splits=4
          count=0
          kf = KFold(n_splits, shuffle=False)
          id=np.array(id1)
          accuracy=np.zeros(n_splits)
          for fold, (train_idx, test_idx) in enumerate(kf.split(id), start=1):
            idt=id[train_idx]
            ids=id[test_idx]
            X_train_r,y_train_r,X_test_r,y_test_r,y_sd=train_test(d_h,target,idt,ids)
            sd=index_dev(y_sd)
            model_LR, y_pred,accuracy[count] = RF_Reg(X_train_r, y_train_r,X_test_r,y_test_r,tol,sd,y_sd)
            count=count+1
            print(idt,ids)
          #model_LR.fit(X_train1, y_train)
          model = model_LR
          Acc_Gender[i]=np.mean(accuracy)
          print('accuracy',Acc_Gender[i])

      #save the final models for later use, i.e feature analysis
      final_models.append(model)
      if(mode!='Reg'):
        #calculate prediction
        y_pred = model.predict(np.array(X_test1))
        #yr_pred = regressor.predict(X_test)
        #print((divide_pred(y_pred,sd)))
        #new_pred=np.array(divide_pred(y_pred,sd))
        new_pred=np.array(divide_pred(y_pred,sd),dtype=object)
        #print(new_pred)

      if (mode=='Reg'):
          #final_labels[i,:] = [np.mean(arr) for arr in new_pred]
          print(final_labels[i,:])
      elif (mode=='SVM'):
          y_pred[y_pred == -1] = 0
          final_labels[i,:]=final_label(new_pred,len(sd))
      else:
          final_labels[i,:]=final_label(new_pred,len(sd))


      if(mode!='Reg'):
        #calculate true and final labels wrt each user
        true_labels=final_label(np.array(divide_pred(y_sd,sd),dtype=object),len(sd))
        #print(true_labels)
        true_labels=map_label(att,id1,id2)
      #label[i,:]=(np.abs(true_labels - final_labels[i,:]) <= 5)
    #mean_squared_error(true_labels, final_labels[i,:])
      if (mode=='Reg'):
          Acc_pass=Acc_Gender
          F1_pass=Acc_Gender
      else:
          Acc_Gender[i]=accuracy_score(true_labels, final_labels[i,:])*100
      #print('true_labels',true_labels,app_id[i])
    result_list_acc = list(zip(app_id, [round(x, 2) for x in Acc_pass]))
    result_list_F1 = list(zip(app_id, [round(x, 2) for x in F1_pass]))
    return final_models ,result_list_acc, result_list_F1, X_train1

#Run Cross app evaluation
def run_cross(mode,app_id,target,D,drop_target=None):
    select='nemo'
    dtype='nhand'
    n_app=len(app_id)
    print(target)
    final_models=[]
    data1,att,id1,id2=add_targetV(target,D,mode)
    #id1=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,19]
    id2=id1#=[13,14,15,16,17,19]
    d1=len(id1)
    #final_labels=np.zeros((n_app-1,d1))
    label=np.zeros((n_app-1,d1))
    Acc_Gender=np.zeros(n_app-1)
    Acc_pass=np.zeros(n_app-1)
    F1_pass=np.zeros(n_app-1)
    #select data type:
    if select=='emo':
      fe=emo[6]
      fE=change_nameFeatures(fe)
      #print(fE)
      d=Emotion(data1,fE)
      d[target] = data1[target]
    else:
      d=data1
            #print(d)
   #calculate accuracy for each
    for i in range(1,n_app):
      print('The following app is: Train/Test',app_id[0],app_id[i])
      d_h1=data_preProcess(d,app_id[0],target)
      d_h=data_preProcess(d,app_id[i],target)
      #print(d_h.columns)
      X_train,y_train,X_testn,y_testn,y_sdn=train_test(d_h1,target,id1,id2)
      X_test,y_test,X_trainn,y_trainn,y_sd=train_test(d_h,target,id1,id2)
      if drop_target:
        columns_to_drop = [col for col in drop_target if col in X_train.columns]
        if columns_to_drop:
            X_train1 = X_train.drop(columns=columns_to_drop, errors='ignore')
            X_test1 = X_test.drop(columns=columns_to_drop, errors='ignore')
            print(X_train1.columns)
      else:
           X_train1=X_train
           X_test1=X_test
      print('training_data',X_train1.shape)

      #get data from each user pole
      sd=index_dev(y_sd)

      if (mode=='RF'):
          #model_RF = RandomForestClassifier()
          if dtype=='hand':
            print('handling Hand-data')
            model_RF = RandomForestClassifier()
            model_RF.fit(np.array(X_train1),y_train)
            feature_importances = model_RF.feature_importances_
            top_500_indices = feature_importances.argsort()[-500:][::-1]
            X_train1 = X_train1.iloc[:, top_500_indices]
            X_test1=X_test1.iloc[:, top_500_indices]
        #RF model Training Start
          print('train data size for RF', X_train1.shape)
          print('train data size for RF', X_test.shape)
          model_RF = RandomForestClassifier()
          #model_RF,Acc_pass[i],F1_pass[i]=RF_tuning(X_train1,y_train,5)
          model_RF.fit(np.array(X_train1),y_train)
          model=model_RF
      elif (mode=='XGB'):
          le = LabelEncoder()
          y_train = le.fit_transform(y_train)
          y_test=le.fit_transform(y_test)

          if dtype=='hand':
            print('handling Hand-data')
            model_XB = xgb.XGBClassifier()
            model_XB.fit(np.array(X_train1),y_train)
            feature_importances = model_XB.feature_importances_
            top_500_indices = feature_importances.argsort()[-500:][::-1]
            X_train1 = X_train1.iloc[:, top_500_indices]
            X_test1=X_test1.iloc[:, top_500_indices]
          print('train data size', X_train.shape)
          model_XB = xgb.XGBClassifier()
          #model_XB=XB_tuning(X_train,y_train,5)
          # Train the model on training data
          model_XB.fit(np.array(X_train1),y_train)
          model=model_XB
          print(app_id[i])

      #save the final models for later use, i.e feature analysis
      final_models.append(model)
      #calculate prediction
      y_pred = model.predict(np.array(X_test1))
      #yr_pred = regressor.predict(X_test)
      #print((divide_pred(y_pred,sd)))
      #new_pred=np.array(divide_pred(y_pred,sd))
      new_pred=np.array(divide_pred(y_pred,sd),dtype=object)


      if (mode=='Reg'):
          final_labels[i,:] = [np.mean(arr) for arr in new_pred]
      elif (mode=='OSV'):
          y_pred[y_pred == -1] = 0
          final_labels[i,:]=final_label(new_pred,len(sd))
      else:
          final_labels=final_label(new_pred,len(sd))

      print(final_labels)
      #calculate true and final labels wrt each user
      true_labels=final_label(np.array(divide_pred(y_sd,sd),dtype=object),len(sd))
      #print(true_labels)
      true_labels=map_label(att,id1,id1)
      print(true_labels)
      #label[i,:]=(np.abs(true_labels - final_labels[i,:]) <= 5)
    #mean_squared_error(true_labels, final_labels[i,:])
      if (mode=='Reg'):
          Acc_Gender[i]=(np.sum(label[i,:])/d1)*100
      else:
          Acc_Gender=accuracy_score(true_labels, final_labels)*100
      Acc_Gender = f1_score(true_labels,final_labels)
      print("For the following app, F1 score",Acc_Gender)
      #print('true_labels',true_labels,app_id[i])
      #print('final_labels',final_labels[i,:])
    result_list_acc = list(zip(app_id, Acc_pass))
    result_list_F1 = list(zip(app_id, F1_pass))
    return Acc_Gender

#Clear attributes that already defined so they do not influence the results
def data_clear(dataset, drops=None):
    if drops:
        columns_to_drop = [col for col in drops if col in dataset]
        if columns_to_drop:
            data_final = dataset.drop(columns=columns_to_drop, errors='ignore')
        else:
            data_final = dataset.copy()  # Return a copy of the original dataset if no columns need to be dropped
    else:
        data_final = dataset
    return data_final
