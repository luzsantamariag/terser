#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Tue Sep 1 20:33:32 2019

@authors: 
    Luz Santamaría Granados (luzsantamariag@gmail.com)
    Juan Francisco Mendoza Moreno (jfmendozam@gmail.com)
"""

import firebase_admin as fb                 
from firebase_admin import credentials, db
from pymongo import MongoClient            
import pandas as pd
# import os 
import matplotlib
matplotlib.use('TkAgg')

class Dataset:
    """ 
    Load the emotion, activity, location, and physiological signals datasets
    of the participants. 
    """
    
    def __init__(self):
        """ Default constructor     
        """   
        self.mebData                     = {}
        self.parentDir                   = './meb'                           # Project directory
        self.firebase                    = 'server/ServiceAccountKey.json'   # JSON file with the authenticate parameters to firebase
        self.database                    = 'https://project.firebaseio.com/' # Firebase project URL
        self.mongo_client                = 'mongodb://localhost:27017'       # MongoDB server URL
        self.mongo_db                    = 'terser'                          # MongoDB database name
        self.firebase_collection         = 'myemotionband'      # Emotion, activity, and location collection obtained from firebase database.
        self.hr_collection               = 'hrband'             # Heart rate collection obtained from participant's wearables devices.
        self.tagged_collection           = 'mebTag'             # Labeled Heart rate collection with emotions and unlabeled heart rate records.
        self.filtered_tagged_coll        = 'mebFilteredTag'     # Labeled Heart rate collection with emotions. 


#%%   
    def loadDataset(self, parent_dir, firebase_filename, database_url, url_mongo_client, mongo_db,
                     firebase_collection, hr_collection):
        """
        Load connection parameters to MongoDB and Firebase.
        Parameters
        ----------
        parent_dir : String
            DESCRIPTION. Project directory
        firebase_filename : String
            DESCRIPTION. JSON file with the authenticate parameters to firebase
        database_url : String
            DESCRIPTION. Firebase project URL
        url_mongo_client : String
            DESCRIPTION. MongoDb server URL
        mongo_db : String
            DESCRIPTION. MongoDB database name
        firebase_collection : String
            DESCRIPTION. Emotion, activity, and location collection obtained from the firebase database.                     
        hr_collection : String
            DESCRIPTION. Heart rate data collection obtained from participant's wearables devices.           
        Returns
        -------
        None.
        """
        self.mongo_client = url_mongo_client
        self.mongo_db = mongo_db
        self.firebase_collection = firebase_collection                  
        self.hr_collection = hr_collection                              
        
        # Activate the getFirebaseConnection method if you have set the connection and database in Firebase.
        if firebase_filename != 'yourServiceAccountKey.json' and database_url != 'https://yourdb.firebaseio.com/':  
             self.getFirebaseConnection(parent_dir, firebase_filename, database_url)
        self.getConnectionMongoDB()
        
#%%    
    def getFirebaseConnection(self, parentDir, firebase, database):
        """
        Get connecting to the "meb" project in Firebase.
        Parameters
        ----------
        parentDir : String
            DESCRIPTION. Project directory
        firebase : String
            DESCRIPTION. JSON file with the authenticate parameters to firebase
        database : String
            DESCRIPTION. Firebase project URL
        Returns
        -------
        None.
        """
        if (not len(fb._apps)):
            # Fetch the service account key JSON file contents
            cred = credentials.Certificate(parentDir + firebase)
            # Initialize the app with a service account, granting admin privileges    
            fb.initialize_app(cred, {
                'databaseURL': database})
        
        # The app has access to read and write all data. The reference is a Firebase
        ref = db.reference()
        self.mebData = ref.get()
        self.manageMongoDB()

#%%     
    def getConnectionMongoDB(self):
        """
        It gets connecting to MongoDB.
        Returns
        -------
        None.
        """
        # Create a client instance of the MongoClient class
        self.client = MongoClient(self.mongo_client)        
        # Create database and collection instances
        self.dataBaseMongo = self.client[self.mongo_db]
        print("Get connecting to the MongoDB")  
                  
#%% 
    def manageMongoDB(self):
        """
        It gets connecting to MongoDB and creates database and collection. 
        - Emotion collection obtained from the firebase database with participants' activity and location data.     
        Returns
        -------
        None.
        """        
        # Create myemotionband collection
        self.emotionCollection = self.dataBaseMongo[self.firebase_collection]
          
        total_docs = self.emotionCollection.count_documents({})
        print (self.emotionCollection.name, "has", total_docs, "total documents.")
        
        if total_docs == 0:   
            self.collectionDataInsert() 
        else:
            self.collectionDataUpdate()

        self.client.close()

#%%      
    def collectionDataInsert(self):
        """
        It inserts the emotion, activity, and location data using the insert_one() method.
        Returns
        -------
        None.
        """    
        print('Insert new file in MyEmotionBand collection ')

        self.dataList_meb = []
        i = 0
        for k, v in self.mebData.items():
            for k1, v1 in v.items():
                if 'key' in v1.keys():
                    v1['_id'] = v1['key']
                    v1['deviceID'] = int(v1['deviceID'])  
                    v1['eTimeStamp'] = int(v1['eTimeStamp'])
            
                self.emotionCollection.insert_one(v1)
                self.dataList_meb.append(v1)
                i = i + 1
                
#%%              
    def collectionDataUpdate(self):
        """
        It deletes the collection 'myemotionband', creates the collection again, and inserts the data.
        Returns
        -------
        None.

        """
        """ Update the data in to the collection using the update_one() method
        """
        self.dataBaseMongo.drop_collection(self.firebase_collection)
        print("Drop the db and collection")
        self.getConnectionMongoDB()                
        self.collectionDataInsert()    
        

#%%       
    def getRawData(self, firebase_collection, hr_collection):
        """
        Retrieve the raw data collections from MongoDB. These collections of emotions and HR
        are collected from the experiment participants' wearable devices and smartphone apps.
        Parameters
        ----------
        firebase_collection : String
            DESCRIPTION. Name of emotion, activity and geographic location collection.
        hr_collection : String
            DESCRIPTION. Name of Heart Rate collection.

        Returns
        -------
        emotionData : Dictionary
            DESCRIPTION. Dictionary of myemotionband collection.
        hrData : Dictionary
            DESCRIPTION. Dictionary of hrband collection.

        """
        emotionData = {}
        hrData = {}
        self.getConnectionMongoDB()   
        # It loads the myemotionband collection
        emotionCollection = self.dataBaseMongo[firebase_collection]        
        total_docs = emotionCollection.count_documents({})
        print (emotionCollection.name, "has", total_docs, "total documents.")
        # It loads the heart rate collection 
        hrCollection = self.dataBaseMongo[hr_collection]   
        total_docs = hrCollection.count_documents({})
        print (hrCollection.name, "has", total_docs, "total documents.")        
        
        #  Create a list and append dictionary into it.
        for k in range(1, 19):
            ecursor = emotionCollection.find({'deviceID' : k})
            emotionTmp = {}
            for i in ecursor:
                j = i.copy()
                del(j['_id'])
                emotionTmp[i['_id']] = j
            emotionData[k] = emotionTmp        
                
            hrcursor = hrCollection.find({'imei' : k})
            hrTmp = {}
            for i in hrcursor:
                hrTmp[i['time']] = i['hr']
                    
            hr = {int(k1):int(v) for k1,v in hrTmp.items()}
            hrData.update({k: hr})
            
        return emotionData, hrData

#%%    
    def taggedCollectionCreate(self, taggedHr, tagged_collection, filtered_tagged_coll):
        """
        It inserts the data into the labeled Heart rate collection with emotions.
        It also includes the unlabeled heart rate records.
        Parameters
        ----------
        taggedHr : Dictionary
            DESCRIPTION. Dictionary keys are: imei, hrTimestamp, emotionts, emotion, 
            activity, hr, longitude, latitude, and altitude.
        tagged_collection : String
            DESCRIPTION. Labeled Heart rate collection with emotions and unlabeled heart rate records.
        filtered_tagged_coll : String
            DESCRIPTION. Labeled Heart rate collection with emotions. 
        Returns
        -------
        None.
        """     
        self.getConnectionMongoDB()    # Create a client instance of the MongoClient class    
        print('Insert new file in mebTag collection ')       
        self.taggedAllCollection = self.dataBaseMongo[tagged_collection] 
        self.taggedFilCollection = self.dataBaseMongo[filtered_tagged_coll] 
        # Drop the collections
        self.dataBaseMongo.drop_collection(self.taggedAllCollection)  # taggedMeb collection
        self.dataBaseMongo.drop_collection(self.taggedFilCollection) # FilteredTaggedMeb collection        
        
        #imei hrTimestamp emotionts emotion activity hr longitude latitude altitude
        for k, v in taggedHr.items():
            for k1, v1 in v.items():
                v1['imei'] = int(v1['imei'])
                v1['hrTimestamp'] = int(k1)
                if 'emotion' in v1:
                    v1['emotionts'] = int(v1['emotionts'])
                    v1['emotion'] = v1['emotion']
                    v1['activity'] = v1['activity'] 
                    v1['hr'] = int(v1['hr'])
                    v1['longitude'] = float(v1['longitude'])
                    v1['latitude'] = float(v1['latitude'])              
                self.taggedAllCollection.insert_one(v1)
                
        for k, v in taggedHr.items():
            for k1, v1 in v.items():
                if 'emotion' in v1:
                    v1['imei'] = int(v1['imei'])
                    v1['hrTimestamp'] = int(k1)
                    v1['emotionts'] = int(v1['emotionts'])
                    v1['emotion'] = v1['emotion']
                    v1['activity'] = v1['activity'] 
                    v1['hr'] = int(v1['hr'])
                    v1['longitude'] = float(v1['longitude'])
                    v1['latitude'] = float(v1['latitude'])                     
                    self.taggedFilCollection.insert_one(v1)

        self.client.close()
        
#%%
    def getTaggedHR(self, filtered_tagged_coll):
        """
        It recoveries from MongoDB the mebFilteredTag collection.
        Parameters
        ----------
        filtered_tagged_coll : String
            DESCRIPTION. Name of labeled Heart rate collection with emotions.  
        Returns
        -------
        hrData : Dictionary
            Dictionary of mebFilteredTag collection.
        """

        self.getConnectionMongoDB()    # Create a client instance of the MongoClient class    
        # It loads the mebFilteredTag collection
        mebFilteredTagCollection = self.dataBaseMongo[filtered_tagged_coll]
        total_docs = mebFilteredTagCollection.count_documents({})
        print (mebFilteredTagCollection.name, "has", total_docs, "total documents.")
        
        #  Create a list and append dictionary into it.
        hrData = {}
        for k in range(1, 19):
            cursor = mebFilteredTagCollection.find({'imei' : k})
            
            hrTmp = {}
            for i in cursor:
                j = i.copy()
                del(j['_id'])
                del(j['hrTimestamp'])
                hrTmp[i['hrTimestamp']] = j
            
            hrData[k] = hrTmp

        self.client.close()

        return hrData
         
#%%
    def mebSliced_collection(self, slices, sliced_collection):
        """
        It inserts the sliced emotion and heart rate instances collection using the insert_one() method.
        Returns
        Parameters
        ----------
        slices : List
            DESCRIPTION : Emotional Slicing features: 0 imei, 1 slicesCounter, 2 hr, 3 instancesCounter, 4 sliceDuration, 
                         5 activity, 6 catActivity, 7 catEmotion, 8 emotion, 9 emotionInstances, 10 catVA, 
                         11 VA, 12 ts, 13 emotionts, 14 hr_sliceSize_norm, 15 longitude, 16 latitude,
                         17 resampled_hr, 18 duplic_TimeSeries_hr       
        sliced_collection : String
            DESCRIPTION. Sliced data collection name
        slice_filename : String
            DESCRIPTION. File name       
        Returns
        -------
        None.
        """
        self.getConnectionMongoDB()    # Create a client instance of the MongoClient class           
        slicedCollection = self.dataBaseMongo[sliced_collection] 
        self.dataBaseMongo.drop_collection(slicedCollection)
        
        column = ['imei', 'slicesNumber', 'hr', 'instancesNumber','sliceDuration','activity',
                  'catActivity','catEmotion','emotion','emotionInstances', 'catVA', 'VA', 'ts',
                  'emotionts','hr_sliceSize_norm','longitude','latitude','resampledhr','duplicTimeSeries_hr']
        df_features = pd.DataFrame(slices, columns=column)

        df = df_features.filter(['imei','slicesNumber', 'hr', 'instancesNumber','sliceDuration',
                  'activity','catActivity','catEmotion','emotion','catVA', 'VA',
                  'ts','hrNormalized','longitude', 'latitude'], axis=1)
        df = df.to_dict('records')
        for i in range(len(df)):
            slicedCollection.insert_one(df[i])
            
        self.client.close()

#%% 
    def emotionRecognition(self, eRprediction, er_collection, participant, classType):
        """
        It inserts the emotion recognition instances collection using the insert_one() method.
        Parameters
        ----------
        eRpred : list
            DESCRIPTION. It contains the emotion recognition results generated 
            from the heart rate data and activities of each participant.
        emotionRecognition_collection : TYPE
            DESCRIPTION.
        url_mongo_client : String
            DESCRIPTION. MongoDB server URL 
        mongo_db : String
            DESCRIPTION. MongoDB database name
        participant : Int
            DESCRIPTION. Experiment participant            
        Returns
        -------
        None.
        """
        self.getConnectionMongoDB()    # Create a client instance of the MongoClient class   
        collectionER = self.dataBaseMongo[er_collection] # Emotion recognition collection
        if participant != -1 and classType == 0: 
            headers = ["imei","emotion","duration","slices"]                   
            erData = []
            
            for i in range(len(eRprediction[6])):
                for j in range (len(eRprediction[9])):
                    if eRprediction[6][i] == eRprediction[9][j][1]:
                        duration = round(float(eRprediction[9][j][2]/60), 2)    # Emotion duration in seconds 
                        break
    
                zipER = dict(zip(headers, [int(participant), eRprediction[6][i],
                                                duration, int(eRprediction[8][i])]))                      
                collectionER.insert_one(zipER)
                erData.append(zipER)  
    
            self.client.close()  