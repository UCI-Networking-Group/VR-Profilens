#This file provides processed data
import numpy as np
import pandas as pd
from util import change_nameFace, change_nameF, change_nameH, change_nameEye,change_nameE, RLeft_Right, change_nameHand, change_eliminate_Head, change_eliminate_RightHand, Emotion_units, change_nameFacialEexpression


def Final_feature(SG):
    if SG=='BM': #Body Motion Data

        print("Open World Data as Input:")
        #call abstracted Body Motion for open world settings
        data2 =pd.read_csv('.../data/BM/SG1_FBN_r2_feature.csv', sep=',')
        data1 = pd.read_csv('../data/BM/SG1_FBN_r1_feature.csv', sep=',')
        data_05 = pd.read_csv('../data/BM/SG1_FBN_r0.5_feature.csv', sep=',')
            

        #preprocess the data
        datar1=change_nameH(data1)
        datar2=change_nameH(data2)
        datar_05=change_nameH(data_05)
        D=[datar2,datar1,datar_05]

    elif SG=='EG': #Eye Tracking/Gaze Data
        #call abstracted Eye Gaze data for 20 users
        data2 =pd.read_csv('../data/EG/SG2_FTN_2s_feature.csv', sep=',')
        data1 = pd.read_csv('../data/EG/SG2_FBN_r1_feature.csv', sep=',')
        data_05 = pd.read_csv('../data/EG/SG2_FTN_1s_feature.csv', sep=',')
        
        datar1=change_nameE(data1)
        datar2=change_nameE(data2)
        datar_05=change_nameE(data_05)
        
        #preprocess the data
        dataE=[datar2,datar1,datar_05]
        
        dataEye=[]
        for i in range(len(dataE)):
          d=dataE[i]
          dnew1=RLeft_Right(d)
          dnew=change_nameEye(dnew1)
          dataEye.append(dnew)
        #Final eyedata
        D=dataEye


    elif SG=='HJ':  #Hand joint or Hand Tracking data
        #call abstracted Hand Joint data for 20 users
        data2 =pd.read_csv('../data/HJ/SG3_FTN_2s_feature.csv', sep=',')
        data1 = pd.read_csv('..R/data/HJ/SG3_FTN_1s_feature.csv', sep=',')
        print("before",data1.columns)
        data_05 = pd.read_csv('../data/HJ/SG3_FTN_0.5s_feature.csv', sep=',')

        #Change name
        datar1=change_nameHand(data1)
        print("after",data1.columns)
        datar2=change_nameHand(data2)
        datar_05=change_nameHand(data_05)
        
        #preprocess the data
        D=[datar2,datar1,datar_05]
        
    
    elif SG=='FE':

        #call abstracted Facial data for 20 users
        data2 =pd.read_csv('../data/FE/SG4_FBN_r2_feature.csv', sep=',')
        data1 = pd.read_csv('../data/FE/SG4_FBN_r1_feature.csv', sep=',')
        data_05 = pd.read_csv('../data/FE/SG4_FBN_r0.5_feature.csv', sep=',')
        
        #Preprocess the data
        datar2=change_nameF(data2)
        datar1=change_nameF(data1)
        datar_05=change_nameF(data_05)
        D=[datar2,datar1,datar_05]
        

    elif SG=='BM_FE':

        #call abstracted Facialand Body Data (multi-sensor adversary) data for 20 users
        data2 =pd.read_csv('../FE/SG4_FBN_r2_feature.csv', sep=',')
        data1 = pd.read_csv('..ta/PrevAdv/ftn-BM-FE.csv', sep=',')
        data_05 = pd.read_csv('../FE/SG4_FBN_r0.5_feature.csv', sep=',')
        
        #Preprocess the data
        datar2=change_nameF(data2)
        datar1=change_nameH(change_nameF(data1))
        datar_05=change_nameF(data_05)
        D=[datar2,datar1,datar_05]

    elif SG=='BM_FE_EG':

        #call abstracted Facial data for 20 users
        data2 =pd.read_csv('../FE/SG4_FBN_r2_feature.csv', sep=',')
        data1 = pd.read_csv('../data/PrevAdv/ftn-BM-EG-FE.csv', sep=',')
        data_05 = pd.read_csv('../data/FE/SG4_FBN_r0.5_feature.csv', sep=',')
        
        #Preprocess the data
        datar2=change_nameF(data2)
        datar1=change_nameH(change_nameF(data1))
        datar_05=change_nameF(data_05)
        D=[datar2,datar1,datar_05]
        
    return D



        

