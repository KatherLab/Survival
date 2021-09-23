# -*- coding: utf-8 -*-
"""
Created on Thu Jun 17 09:45:00 2021

@author: Narmin Ghaffari Laleh
"""

import random
import numpy as np
import pandas as pd
import os
import argparse
from Survival_Utils import CreateCleanedTable, GetTiles, Predictor
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from sksurv.metrics import concordance_index_censored
import gc
from keras_preprocessing.image import ImageDataGenerator
from tqdm import tqdm

##############################################################################

parser = argparse.ArgumentParser(description = 'Survival Prediction')
parser.add_argument('--datadir_test', type = str, default = r'')
parser.add_argument('--slide_dir', type = str, default = "")
parser.add_argument('--clini_dir', type = str, default = "")
parser.add_argument('--outputPath', type = str, default = r"")

parser.add_argument('--modelPath', type = str, default = r"")


parser.add_argument('--num_classes', type = int, default = 1)
parser.add_argument('--batch_size', type = int, default ="64")
parser.add_argument('--maxBlockNum', type = int, default ="150")

parser.add_argument('--gpuNo', type = int, default = 1)

parser.add_argument('--evenCol', type = str, default = "")
parser.add_argument('--timeCol', type = str, default = "")
parser.add_argument('--timeInDays', type = bool, default = False)
parser.add_argument('--patientCol', type = str, default = "")

##############################################################################

if __name__ == '__main__':
    
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpuNo)

    random.seed(23)
    args.projectFolder = args.outputPath
    
    if not os.path.exists(args.projectFolder):
        os.mkdir(args.projectFolder) 
    
            
    cleanedTable, _ = CreateCleanedTable(imagesPath = args.datadir_test, cliniTablePath = args.clini_dir, slideTablePath = args.slide_dir,
                                         targetNames = [args.evenCol, args.timeCol],outputPath = args.outputPath)
    
    
    cleanedTable.dropna(subset = [args.evenCol], inplace = True) 
    cleanedTable.dropna(subset = [args.timeCol], inplace = True) 
    cleanedTable = cleanedTable[~cleanedTable[args.timeCol].isin(['na', 0])]
    cleanedTable = cleanedTable[~cleanedTable[args.evenCol].isin(['na'])]
    patientID = list(cleanedTable[args.patientCol].unique())
    
    time = list(cleanedTable[args.timeCol])
    time = [float(i) for i in time]
    
    if args.timeInDays:
        time = [i / 30 for i in time]
    
    event = list(cleanedTable[args.evenCol])
    if 'yes' in event:
        event = [1 if i == 'yes' else 0 for i in event]
    else:
        event = [int(i) for i in event]
            
    patientID = np.asarray(cleanedTable[args.patientCol].unique())
    time = np.asarray(time)
    event = np.asarray(event)
    
    args.split_dir = os.path.join(args.projectFolder, 'SPLITS')
    if not os.path.exists(args.split_dir):
        os.mkdir(args.split_dir) 

    args.results = os.path.join(args.projectFolder, 'RESULTS')
    if not os.path.exists(args.results):
        os.mkdir(args.results) 
        
    test_data = GetTiles(patients = patientID, time = time, event = event, imgsList = args.datadir_test,
                         cleanedTable = cleanedTable, maxBlockNum = args.maxBlockNum, test = True)
    
    test_x = list(test_data['TILEPATH'])
    test_time = list(test_data['TIME'])
    test_event = list(test_data['EVENT'])
    test_pid = list(test_data['PATIENT'])
    
    df = pd.DataFrame(list(zip(test_x, test_time, test_event)), columns =['TILEPATH', 'TIME', 'EVENT'])
    df.to_csv(os.path.join(args.split_dir, 'SPLIT_TEST_TOTAL.csv'), index = False) 
            
    base_model = ResNet50(weights='imagenet', include_top=False)
    
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(args.num_classes, activation='linear')(x)
    
    model = Model(inputs = base_model.input, outputs=predictions)
                
    temp_df = pd.DataFrame(test_x)
    
    test_datagen = ImageDataGenerator()
    
    test_generator=test_datagen.flow_from_dataframe(
                                dataframe=temp_df,
                                x_col = 0,
                                batch_size = args.batch_size,
                                seed = 42,
                                shuffle = False,
                                class_mode = None,
                                target_size = (224,224))
        
    test_time = np.asarray(test_time)
    test_event = np.asarray(test_event)
    
    
    predictor = Predictor(model, args.modelPath)
    sample_predictions = []
    
    #ds = tf.data.Dataset.from_tensor_slices(test_x).batch(64)
    #it = tf.compat.v1.data.make_one_shot_iterator(ds)
    batches = 0  
    for tensor in tqdm(test_generator):
        sample_predictions.extend(predictor.predict(tensor))
        gc.collect() 
        batches += 1
        if batches >= len(test_generator):
            # we need to break the loop by hand because
            # the generator loops indefinitely
            break            
        #tensor = it.get_next()
    
     
    
    df = pd.DataFrame(list(zip(test_pid, test_x, list(sample_predictions), test_time, test_event)),columns =['PATIENT', 'TILEPATH', 'SCORE', 'TIME', 'EVENT'])
    df.to_csv(os.path.join(args.results, 'TEST_RESULTS_ORIGINAL_FULL.csv'), index = False)
    scores = []
    patients = []
    t = []
    e = []
    
    patientID_unique = set(test_pid)
    for pi in patientID_unique:
        patients.append(pi)
        data_temp = df.loc[df['PATIENT'] == pi]
        data_temp = data_temp.reset_index()
        score = np.mean(data_temp['SCORE'])[0]
        scores.append(score)
        t.append(data_temp['TIME'][0])
        e.append(data_temp['EVENT'][0]) 
    
    label = []
    hazard = []
    surv_time = []
    for i in range(len(scores)):
        if not np.isnan(scores[i]):
            label.append(e[i])
            hazard.append(scores[i])
            surv_time.append(t[i])
    
    new_label = np.asarray(label)
    new_hazard = np.asarray(hazard)
    new_surv = np.asarray(surv_time)
    
    cindex = concordance_index_censored(
      event_indicator=new_label == 1,
      event_time=new_surv,
      estimate=new_hazard)
            
    df = pd.DataFrame(list(zip(patients, t, e, scores)), columns = ['PATIENT', 'TIME', 'EVENT', 'SCORE'])
    df.to_csv(os.path.join(args.results, 'TEST_RESULTS_FULL.csv'), index = False)
    
    print('\n')   
    print('#################################################################')
    print('Patient-Wise Concordance_Index For Test Data Set {}:'.format(cindex))
    print('#################################################################')