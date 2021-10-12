#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Tue Sep 1 20:33:32 2019

@authors: 
    Luz Santamaría Granados (luzsantamariag@gmail.com)
    Juan Francisco Mendoza Moreno (jfmendozam@gmail.com)
"""

from Dataset import Dataset
from DataPlot import DataPlot
from Preprocessing import Preprocessing
from EmotionalSlice import EmotionalSlice
from EmotionBalanced import EmotionBalanced
from EmotionRecognition import EmotionRecognition
import numpy as np


class EmotionRecognitionSystem:
    
    def __init__(self):

        self.parent_dir             = 'yourdirectory'                         # Project directory
        self.firebase_filename      = 'yourServiceAccountKey.json'            # JSON file with the authenticate parameters to firebase
        self.database_url           = 'https://yourdb.firebaseio.com/'        # Firebase project URL
        self.model_dir              = 'models/'                               # Emotion recognition models subdirectory 
        self.meb_plot_dir           = 'myemotionband/'                        # Plot subdirectory of emotion, activity, and participants data.
        self.slice_plot_dir         = 'slice/'                                # Emotional Slicing of HR instances subdirectory
        self.hr_plot_dir            = 'miband/'                               # Heart rate plot subdirectory    
        self.url_mongo_client       = 'mongodb://localhost:27017'             # MongoDb localhost URL
        self.mongo_db               = 'terser'                                # MongoDB database name
        self.participant            = -1                                      # Experiment participants. All participants -1 or participant 11
        self.classType              = 1                                       # Label class for balancing. Emotion (0), AV (1)
        self.emotionMinSlices       = 100                                     # Slices minimum for each emotion class. Default 100         
        self.classMinNum            = 2                                       # Emotion classes minimum number. Default 2. 
        self.cluster_balance        = 0.05                                    # Cluster Balance. All participants: 0.05 or An participant: 0.01 

        self.createEmotionHRCollection()   # first method
                     
#%% 
    def createEmotionHRCollection(self):
        """
        # It defines the parameters for generating the emotion and heart rate collection in MongoDB
        # of experiment participants. 
        Returns
        -------
        None.
        """
        firebase_collection    = 'myemotionband'  # Emotion collection obtained from the firebase database.
        hr_collection          = 'hrband'         # Heart rate data collection obtained from participant's wearables devices.
        
        # Load connection parameters to MongoDB and Firebase.
        ds = Dataset()
        ds.loadDataset(self.parent_dir, self.firebase_filename, self.database_url, self.url_mongo_client,
                       self.mongo_db, firebase_collection, hr_collection)
        
        # It generates the statistical plots of the emotions and heart rate collections.
        dp = DataPlot()
        dp.generatePlot(self.hr_plot_dir, self.url_mongo_client, self.mongo_db,
                        firebase_collection, hr_collection, self.meb_plot_dir)        
        
        # Retrieve the collections of raw heart rate and raw emotional states.
        ds = Dataset()
        mebData, hrData = ds.getRawData(firebase_collection, hr_collection)        
        
        self.taggedDataset(mebData, hrData)
        
#%% 
    def taggedDataset(self, mebData, hrData):
        """
        It defines the parameters for preprocessing the emotion, activity, location, and
        physiological signals dataset of the participants.
        Parameters
        ----------
        mebData : dict
            DESCRIPTION. Emotion, activity, and location dataset
        hrData : dict
            DESCRIPTION. Heart rate dataset tagged with emotion class            
        Returns
        -------
        None.
        """
        max_size_window        = 180               # Maximum size window for time-series data tagging algorithm.  
        tagged_collection      = 'mebTag'          # Labeled HR collection with emotions and unlabeled HR records.
        filtered_tagged_coll   = 'mebFilteredTag'  # Labeled HR collection with emotions.        

        pcds = Preprocessing()
        pcds.preprocessDataset(mebData, hrData, max_size_window)
        
        # It saves the tagged collections in MongoDB - Load preprocessing dictionary 
        ds = Dataset()
        ds.taggedCollectionCreate(hrData, tagged_collection, filtered_tagged_coll)

        self.generateEmotionalSlice(filtered_tagged_coll)

#%% 
    def generateEmotionalSlice(self, filtered_tagged_coll):
        """
        Perform the slicing of emotion and heart rate instances
        Parameters
        ----------        
        filtered_tagged_coll : String
            DESCRIPTION. Collection of heart rate dataset tagged with emotion class          
        Returns
        -------
        None.
        """     
        timeBetweenInstances  = 5           # Time distance between instances (seconds).
        self.sliceSize        = 30          # Slice size. 
        sliceLimit            = 29          # Slice size limit.
        sliced_collection     = 'mebSliced' # Sliced emotion and HR instances collection        
        
        # It recoveries from MongoDB the mebFilteredTag collection.
        ds = Dataset()
        taggedHR = ds.getTaggedHR(filtered_tagged_coll)
        
        es = EmotionalSlice()      # It performs the slicing of emotion and heart rate instances
        es.buildSlices(self.slice_plot_dir, taggedHR, timeBetweenInstances, self.sliceSize, sliceLimit)
        features = es.features   
        
        #Save file in MongoDB 
        ds = Dataset()
        ds.mebSliced_collection(features, sliced_collection)        
        
        self.generateClassesBalanced(features)

#%% 
    def generateClassesBalanced(self, features):
        """
        It fixes the Imbalance in the emotion, heart rate and activities dataset.
        Parameters
        ----------
        features : List
            DESCRIPTION. features: 0 imei, 1 slicesCounter, 2 hr, 3 instancesCounter, 4 sliceDuration, 
                         5 activity, 6 catActivity, 7 catEmotion, 8 emotion, 9 emotionInstances, 10 catVA, 
                         11 VA, 12 ts, 13 emotionts, 14 hr_sliceSize_norm, 15 longitude, 16 latitude,
                         17 resampled_hr, 18 duplic_TimeSeries_hr     
        Returns
        -------
        None.
        """
        syntheticData       = True    # Synthetic data for emotion class
        imbalancedType      = 1       # Algorithm for imbalanced dataset. 0: KMeansSMOTE or 1: TomekLinks
        split               = 0.8     # Percentage of training data. 
        
        eb = EmotionBalanced()     # It fixes the Imbalance in the emotion Dataset
        eb.buildDataBalanced(features, syntheticData, self.emotionMinSlices, imbalancedType,
                             self.participant, self.model_dir, self.classMinNum, self.classType,
                             self.cluster_balance, self.sliceSize, split)
        
        self.detectionEmotion(eb)

#%% 
    def detectionEmotion(self, emotionBalanced):
        """
        It performs emotion recognition based on heart rate and context activities slices
        of experiment participants. In each iteration, it validates that the test
        prediction average is greater than 0.6.
        Parameters
        ----------
        emotionBalanced : object
            DESCRIPTION. Object of generateClassesBalanced Class 
        Returns
        -------
        None.
        """  
        eb = emotionBalanced
        er_collection          = 'emotionRecognition'   # It contains the ER results generated from the HR data of each participant.
        filters                = 128            # The dimensionality of the output space in the convolution. 
        kernel_sizes           = [10, 4, 1, 1]  # Specify the length of the 1DCNN window.    
        pool_sizes             = [2, 1, 1]      # Size of the max-pooling windows for temporal data operation between CNN. 
        drop_rate              = 0.5            # Define input unit less than 1.  
        neurons                = [256, 128, 64] # Specify the dimensionality of the array output space for a FCN.  
        batch                  = 32             # Batch size defines the sample number of the dataset. 
        number_epochs          = 10             # Define the iterations number on the dataset during the training time. 
        model_filename         = 'model.hdf5'   # Emotion recognition model File 
        history_filename       = 'history.csv'  # Accuracy results from all epochs of training and testing of the model
        pred_prefix            = 'preds_'       # Filename prefix of the prediction results
        confusion_prefix       = 'result_conf'  # Confusion matrix data 
        emotionMinAverage      = 0.1            # Minimum average prediction for emotion classes.
        modelType              = 0              # Define the CNN or LSTM model. Default 0 (CNN_Flatten), 1 (CNN LSTM), and 2(DCNN)        
        
        if eb.classNum >= 2: # Are there enough label classes?
            if len(eb.durationList) >= self.classMinNum:  # Are enough emotion classes?
                print(eb.durationList)
                print(eb.targetClass)
                i = 0
                eRpred = []
                emotionList = []
                eRprediction = []
                er = EmotionRecognition(eb.x_features, eb.X_train, eb.Y_train, eb.X_val, eb.Y_val)    
                er.run_model(len(eb.emotionLabel), filters, kernel_sizes,
                             pool_sizes, drop_rate, neurons, self.model_dir, model_filename,
                             history_filename, batch, number_epochs, pred_prefix,
                             confusion_prefix, eb.targetClass, eb.emotionLabel,
                             emotionMinAverage, modelType)
                eRpred.append([i, len(eb.emotionLabel), np.mean((er.hist.history['val_accuracy'])), 
                               np.mean((er.hist.history['accuracy'])), max(er.hist.history['val_accuracy']),
                               max(er.hist.history['accuracy']), eb.targetClass, er.emotionPrediction,
                               eb.emotionSlicesNum, eb.durationList]) 
                emotionList.append(er.emotionLabelTemp)
                
            # Save the emotion recognition collection in MongoDB. 
            eRprediction = eRpred[0]
            ds = Dataset()
            ds.emotionRecognition(eRprediction, er_collection, self.participant, self.classType)

                
#%%         
if __name__ == "__main__":
    ers = EmotionRecognitionSystem()
