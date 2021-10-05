# -*- coding: utf-8 -*-
"""
Created on Thu May  6 10:53:38 2021

@author: Narmin Ghaffari Laleh
"""

###############################################################################

import random
import numpy as np
import pandas as pd
import os
import argparse
from Survival_Utils import CreateCleanedTable, GetTiles, InputFunction, TrainAndEvaluateModel, Predictor, MergeResults
import tensorflow as tf
import warnings
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from pathlib import Path
from sksurv.metrics import concordance_index_censored
from tqdm import tqdm
import gc
from keras_preprocessing.image import ImageDataGenerator

###############################################################################

parser = argparse.ArgumentParser(description = 'Survival Prediction')
parser.add_argument('--datadir_train', type = str, default = r'')
parser.add_argument('--slide_dir', type = str, default = "")
parser.add_argument('--clini_dir', type = str, default = "")
parser.add_argument('--folds', type = int, default = 3)
parser.add_argument('--outputPath', type = str, default = r"")
parser.add_argument('--checkPointName', type = str, default = r'oe02-biopsy-TrainFull')

parser.add_argument('--lr', default = 1e-5, type=float, help = 'learning rate (default: 1e-4)')
parser.add_argument('--num_classes', type = int, default = 1)
parser.add_argument('--opt', type = str, default = 'adam')
parser.add_argument('--reg', type = float, default = 0.0001)
parser.add_argument('--num_epochs', type = int, default = 10)
parser.add_argument('--batch_size', type = int, default ="64")
parser.add_argument('--maxBlockNum', type = int, default ="150")

parser.add_argument('--gpuNo', type = int, default = 1)
parser.add_argument('--trainFull', type = bool, default = True)

parser.add_argument('--evenCol', type = str, default = "")
parser.add_argument('--timeCol', type = str, default = "")
parser.add_argument('--timeInDays', type = bool, default = False)
parser.add_argument('--patientCol', type = str, default = "")

###############################################################################

# Create Cleaned Table
resultDict = {}

if __name__ == '__main__':
    
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpuNo)

    random.seed(23)
    args.projectFolder = args.outputPath
    
    if not os.path.exists(args.projectFolder):
        os.mkdir(args.projectFolder) 
    else:
        raise ValueError('This project exists! Please Remove the folder')

    
            
    cleanedTable, _ = CreateCleanedTable(imagesPath = args.datadir_train, cliniTablePath = args.clini_dir, slideTablePath = args.slide_dir,
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
        
    if args.trainFull:
        patientID =list(cleanedTable[args.patientCol].unique())
        print('\nLOAD TRAIN DATASET\n')
           
        train_data = GetTiles(patients = patientID, time = time, event = event, imgsList = args.datadir_train,
                              cleanedTable = cleanedTable, maxBlockNum = args.maxBlockNum, test = False)
        train_x_tiles = list(train_data['TILEPATH'])
        train_time = list(train_data['time'])
        train_event = list(train_data['event'])
                
        train_x = [image.load_img(i, target_size=(224, 224)) for i in tqdm(train_x_tiles)]
        train_x = [image.img_to_array(i) for i in train_x]
        train_x = np.asarray(train_x)
        train_time = np.asarray(train_time)
        train_event = np.asarray(train_event)
        train_fn = InputFunction(train_x, train_time, train_event, drop_last = True, shuffle = True)
        del train_x
        df = pd.DataFrame(list(zip(train_x_tiles, train_time, train_event)), columns =['TILEPATH', 'time', 'event'])
        df.to_csv(os.path.join(args.split_dir, 'SPLIT_TRAIN_TOTAL.csv'), index = False)
        print()
        
        base_model = ResNet50(weights='imagenet', include_top=False)
            
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(1024, activation='relu')(x)
        predictions = Dense(args.num_classes, activation='linear')(x)
        
        model = Model(inputs=base_model.input, outputs=predictions)
        counter = '_FULL'       
        trainer = TrainAndEvaluateModel(
            model = model,
            model_dir = Path(os.path.join(args.outputPath, args.checkPointName + counter)),
            train_dataset = train_fn(),
            eval_dataset = None,
            learning_rate = args.lr,
            num_epochs = args.num_epochs,
        )
        val_logits = trainer.train_and_evaluate()
        
    else:
                    
        kf = StratifiedKFold(n_splits = args.folds, random_state = 23, shuffle = True)
        counter = 0
        
        for train_index, test_index in kf.split(patientID, event):
            testData_patientID = patientID[test_index]   
            testData_time = time[test_index]
            testData_event = event[test_index]    
           
            if not len(patientID) < 50 :
                val_index = random.choices(train_index, k = int(len(train_index) * 0.05))
                valData_patientID = patientID[val_index]   
                valData_time = time[val_index]
                valData_event = event[val_index]
                  
            train_index = [i for i in train_index if i not in val_index]
            
            trainData_patientID = patientID[train_index]   
            trainData_time = time[train_index]
            trainData_event = event[train_index] 
            
            print('\nLOAD TRAIN DATASET\n')
           
            train_data = GetTiles(patients = trainData_patientID, time = trainData_time, event = trainData_event, imgsList = args.datadir_train,
                                  cleanedTable = cleanedTable, maxBlockNum = args.maxBlockNum, test = False)
            
            train_x_tiles = list(train_data['TILEPATH'])
            train_time = list(train_data['TIME'])
            train_event = list(train_data['EVENT'])
            
            
            train_x = [image.load_img(i, target_size=(224, 224)) for i in tqdm(train_x_tiles)]
            train_x = [image.img_to_array(i) for i in train_x]
            train_x = np.asarray(train_x)
            train_time = np.asarray(train_time)
            train_event = np.asarray(train_event)
            train_fn = InputFunction(train_x, train_time, train_event, drop_last = True, shuffle = True)
            del train_x
            df = pd.DataFrame(list(zip(train_x_tiles, train_time, train_event)), columns =['TILEPATH', 'TIME', 'EVENT'])
            df.to_csv(os.path.join(args.split_dir, 'SPLIT_TRAIN_' + str(counter)+ '.csv'), index = False)
            print()   
            
            if not len(patientID)<50:
                print('LOAD Validation DATASET\n')           
                val_data = GetTiles(patients = valData_patientID, time = valData_time, event = valData_event, imgsList = args.datadir_train,
                                    cleanedTable = cleanedTable, maxBlockNum = args.maxBlockNum, test = True)
                
                val_x_tiles = list(val_data['TILEPATH'])   
                val_time = list(val_data['TIME'])
                val_event = list(val_data['EVENT'])
                
                val_x = [image.load_img(i, target_size=(224, 224)) for i in tqdm(val_x_tiles)]
                val_x = [image.img_to_array(i) for i in val_x]
                val_x = np.asarray(val_x)
                val_time = np.asarray(val_time)
                val_event = np.asarray(val_event)
                val_fn = InputFunction(val_x, val_time, val_event, drop_last = True, shuffle = True)
                del val_x
                df = pd.DataFrame(list(zip(val_x_tiles, val_time, val_event)), columns =['TILEPATH', 'TIME', 'EVENT'])
                df.to_csv(os.path.join(args.split_dir, 'SPLIT_VAL_' + str(counter)+ '.csv'), index = False)    
                print()
            
            print('LOAD TEST DATASET')  
            test_data = GetTiles(patients = testData_patientID, time = testData_time, event = testData_event, imgsList = args.datadir_train,
                                 cleanedTable = cleanedTable, maxBlockNum = args.maxBlockNum, test = True)
            
            test_x = list(test_data['TILEPATH'])
            test_time = list(test_data['TIME'])
            test_event = list(test_data['EVENT'])
            test_pid = list(test_data['PATIENT'])
            
            df = pd.DataFrame(list(zip(test_x, test_time, test_event)), columns =['TILEPATH', 'TIME', 'EVENT'])
            df.to_csv(os.path.join(args.split_dir, 'SPLIT_TEST_' + str(counter)+ '.csv'), index = False) 
        
            print()
            print("=========================================")
            print("====== K FOLD VALIDATION STEP => %d =======" % (counter))
            print("=========================================")
            
            base_model = ResNet50(weights='imagenet', include_top=False)
            
            x = base_model.output
            x = GlobalAveragePooling2D()(x)
            x = Dense(1024, activation='relu')(x)
            x = Dense(512, activation='relu')(x)
            x = Dense(128, activation='relu')(x)
            x = Dense(64, activation='relu')(x)
            predictions = Dense(args.num_classes, activation='linear')(x)
            
            model = Model(inputs = base_model.input, outputs=predictions)
            
            
            trainer = TrainAndEvaluateModel(
                model = model,
                model_dir = Path(os.path.join(args.outputPath, args.checkPointName + str(counter))),
                train_dataset = train_fn(),
                eval_dataset = val_fn(),
                learning_rate = args.lr,
                num_epochs = args.num_epochs,
            )
                            
            warnings.filterwarnings('ignore')
            val_logits = trainer.train_and_evaluate()
            
            #test_x = [image.load_img(i, target_size=(224, 224)) for i in tqdm(test_x)]
            #test_x = [image.img_to_array(i) for i in tqdm(test_x)]
            #test_x = np.asarray(test_x)
            temp_df = pd.DataFrame(test_x)
            
            test_datagen = ImageDataGenerator()

            test_generator = test_datagen.flow_from_dataframe(
                                        dataframe=temp_df,
                                        x_col = 0,
                                        batch_size = args.batch_size,
                                        seed = 42,
                                        shuffle = False,
                                        class_mode = None,
                                        target_size = (224,224))
                
            test_time = np.asarray(test_time)
            test_event = np.asarray(test_event)
    
    
            predictor = Predictor(model, trainer.model_dir)
            sample_predictions = []
            
            #ds = tf.data.Dataset.from_tensor_slices(test_x).batch(64)
            #it = tf.compat.v1.data.make_one_shot_iterator(ds)
            batches = 0  
            for tensor in test_generator:
                sample_predictions.extend(predictor.predict(tensor))
                gc.collect() 
                batches += 1
                if batches >= len(test_generator):
                    # we need to break the loop by hand because
                    # the generator loops indefinitely
                    break            
                #tensor = it.get_next()
    
             
    
            df = pd.DataFrame(list(zip(test_pid, test_x, list(sample_predictions), test_time, test_event)),columns =['PATIENT', 'TILEPATH', 'SCORE', 'TIME', 'EVENT'])
            df.to_csv(os.path.join(args.results, 'TEST_RESULTS_TILE_BASED' + str(counter) + '.csv'), index = False)
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
            df.to_csv(os.path.join(args.results, 'TEST_RESULTS_PATIENT_BASED' + str(counter) + '.csv'), index = False)
            print('\n')   
            print('#################################################################')
            print('Patient-Wise Concordance_Index For Test Data Set {}:'.format(cindex))
            print('#################################################################') 
            
            counter += 1


        ##############################################################################
        
        totalCIndex = MergeResults(args.results)
        print('TOTAL C_INDEX: {}'.format(totalCIndex))
        

        files = [os.path.join(args.results , i) for i in os.listdir(args.results)]
        files = [i for i in files if 'TILE_BASED' in i]
        df = pd.DataFrame()
        for file in files:
            df = df.append(pd.read_csv(file), ignore_index=True) 
            
        df.to_csv(os.path.join(args.results, 'TOTAL_TEST_RESULTS_TILE_BASED.csv'))
        
        ##############################################################################


































