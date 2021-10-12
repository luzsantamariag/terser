#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Wed Oct 26 17:49:21 2019

@authors: 
    Luz Santamaría Granados (luzsantamariag@gmail.com)
    Juan Francisco Mendoza Moreno (jfmendozam@gmail.com)
"""

from pymongo import MongoClient           
from datetime import date
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')


class DataPlot:
    """ 
    Consolidates and preprocesses the emotion, location, and 
    physiological signals dataset of the participants. 
    """
   
    def __init__(self):
        """ Default constructor 
            Parameters:
        """
        self.color_sequence = ['#1f77b4', '#aec7e8', '#ff7f0e', '#ffbb78', '#2ca02c',
                               '#98df8a', '#d62728', '#ff9896', '#9467bd', '#c5b0d5',
                               '#8c564b', '#c49c94', '#e377c2', '#f7b6d2', '#7f7f7f',
                               '#c7c7c7', '#bcbd22', '#dbdb8d', '#17becf', '#9edae5',
                               '#EEE8AA', '#800080', '#000080', '#DA70D6', '#C0C0C0',
                               '#D8BFD8', '#FF6347', '#40E0D0', '#EE82EE', '#F5DEB3',
                               '#FFFFFF', '#F5F5F5', '#FFFF00', '#9ACD32', '#FAF0E6'
                               ]
 
#%% 
    def generatePlot(self, hr_plot_dir, url_mongo_client, mongo_db,
                     firebase_collection, hr_collection, meb_plot_dir):
        """

        Parameters
        ----------
        hr_plot_dir : String
            DESCRIPTION. Plot subdirectory of Heart Rate collection
        url_mongo_client : String
            DESCRIPTION. MongoDb server URL
        mongo_db : String
            DESCRIPTION. MongoDB database name
        firebase_collection : String
            DESCRIPTION. Emotion, activity, and location collection obtained from the firebase database.           
        hr_collection : String
            DESCRIPTION. Heart rate data collection obtained from participant's wearables devices.
        meb_plot_dir : String
            DESCRIPTION. Plot subdirectory of Heart Rate collection            
        Returns
        -------
        None.

        """
        self.hr_plot_dir = hr_plot_dir
        self.meb_plot_dir = meb_plot_dir
        self.mongo_client = url_mongo_client
        self.mongo_db = mongo_db
        self.firebase_collection = firebase_collection                  
        self.hr_collection = hr_collection                              

        self.getConnectionMongoDB()

        self.hrPlot(self.hrCollection, self.emotionCollection)
        self.emotionPlot(self.emotionCollection)
        
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
        # Create myemotionband collection
        self.emotionCollection = self.dataBaseMongo[self.firebase_collection]
        # Create heart rate collection 
        self.hrCollection = self.dataBaseMongo[self.hr_collection]
        print("Get connecting to the MongoDB")  
        total_docs = self.emotionCollection.count_documents({})
        print (self.emotionCollection.name, "has", total_docs, "total documents.")
        total_docs_hr = self.hrCollection.count_documents({})
        print (self.hrCollection.name, "has", total_docs_hr, "total documents.")        
        

#%% 
    def hrPlot(self, hrc, ec):
        """
        It obtains the "miband" collection, the heart rate, IMEI, and the
        registration date of each participant. 
        It shows the subplot of the heart rate data of the experiment participants. 
        It shows a plot of the heart rate data recorded by each participant daily.
        Parameters
        ----------
        hrc : MongoDB object
            DESCRIPTION. Heart rate collection.
        ec : MongoDB object
            DESCRIPTION. Emotion, activity and location collection.
        Returns
        -------
        None.
        """
       
        hrCursor = hrc.aggregate([
            	{
        		"$group": 
        		{
        			"_id": 
        			{
        				"imei":"$imei",
        				"date":
        				{ 
        				"$dateToString": 
        					{
        						"format": "%Y-%m-%d",
        						"date":
        						{
        							"$toDate":
        							{
        								"$multiply": [1000, "$time"]
        							}
        						}
        					}
        				}
        			},
        			"count": { "$sum": 1 }
        		}
        	},
        	{
        		"$sort": {"_id.imei": 1, "_id.date": 1}
        	}
        ])
        
        # Consolidate the heart rate record of all the participants by IMEI, date, and record count.   
        hrList = []
        for i in hrCursor:
            tmp = [i['_id']['imei'], i['_id']['date'], i['count']]
            hrList.append(tmp)
        hrdf = pd.DataFrame(hrList, columns = ['imei', 'date', 'count'])
        hrdf['month_day'] = hrdf['date'].str[-5:]
        # Consolidate the data record dates of all participants.        
        values = (np.sort(hrdf['month_day'].unique())).tolist()
        values_df =pd.DataFrame(values, columns = ['month_day'])        
                
        # It shows the subplot of the heart rate data of the experiment participants. 
        fig, ax = plt.subplots(figsize = (12, 6))
        plt.rc('legend', fontsize = 4)
        plt.title('Record of heart rate data',loc='center')        
        for i, user in enumerate(ec.find().distinct("deviceID")):
            this_data = hrdf.loc[hrdf['imei'] == int(user)] 
            if (this_data.size > 0):
                this_data.plot(x = 'month_day', y = 'count', ax = ax,
                               label = self.users(user), fontsize = 7,
                               color = self.color_sequence[i])
        plt.xticks(range (0, len(values_df.index)), values_df['month_day'])  
        plt.setp(ax.get_xticklabels(), rotation=90, ha="right", rotation_mode="anchor")     
        fig.savefig(self.hr_plot_dir + str(date.today()) + '_' + 'hr_history.svg', format='svg')    
    
        # It shows a plot of the heart rate data recorded by each participant daily.
        for c, user in enumerate(ec.find().distinct("deviceID")):
            this_data = hrdf.loc[hrdf['imei'] == int(user)]
            if (this_data.size > 0):  
                this_data.plot(x = 'month_day', y = 'count', 
                               label = self.users(user), fontsize=7,
                               color = self.color_sequence[c])
                plt.title('Record of heart rate data per participant', fontsize=8)
                plt.savefig(self.hr_plot_dir + str(user) + '_' + 'hr.svg', format='svg')
                plt.close()

#%% 
    def emotionPlot(self, emotionc):
        """
        It obtains the MongoDB "myEmotionBand" collection, the emotion, IMEI, and the
        registration date of each participant.
        It shows the plot of the emotion, activity, and location data of the experiment
        participants.
        It shows a plot of the emotion, activity, and location data recorded by 
        each participant daily.
        Parameters
        ----------
        emotionc : MongoDB object
            DESCRIPTION. Emotion, activity and location collection.
        Returns
        -------
        None.
        """

        eCursor = emotionc.aggregate([
           	{
        		"$group": 
        		{
        			"_id": 
        			{
        				"imei":"$deviceID",
        				"date":
        				{ 
        				"$dateToString": 
        					{
        						"format": "%Y-%m-%d",
        						"date":
        						{
                                "$toDate":
                                    {
        							       "$multiply": [1000, "$eTimeStamp"]                                        
                                    }
        						}
        					}
        				}
        			},
        			"count": { "$sum": 1 }
        		}
        	},
        	{
        		"$sort": {"_id.imei": 1, "_id.date": 1}
        	}
        ])
        
        # Consolidate the emotion record of all the participants by IMEI, date, and record count.        
        eList = []
        for i in eCursor:
            tmp = [i['_id']['imei'], i['_id']['date'], i['count']]
            eList.append(tmp)       
        emotiondf = pd.DataFrame(eList, columns = ['imei', 'date', 'count'])
        emotiondf['month_day'] = emotiondf['date'].str[-5:]
        # Consolidate the data record dates of all participants.
        values = (np.sort(emotiondf['month_day'].unique())).tolist()
        values_df =pd.DataFrame(values, columns = ['month_day'])
                 
        # It shows the subplot of the emotion, activity, and location data of the experiment participants.        
        fig, ax = plt.subplots(figsize = (12, 6))  
        plt.rc('legend', fontsize = 4)
        plt.title('Record of emotion and traceability data',loc='center') 
        for i, user in enumerate(emotionc.find().distinct("deviceID")): 
            this_data = emotiondf.loc[emotiondf['imei'] == int(user)]
            if (this_data.size > 0):  
                this_data.plot(x = 'month_day', y = 'count', ax = ax,
                                    label = self.users(user), fontsize = 7, 
                                    color = self.color_sequence[i])
        plt.xticks(range (0, len(values_df.index)), values_df['month_day'])  
        plt.setp(ax.get_xticklabels(), rotation=90, ha="right", rotation_mode="anchor")       
        plt.savefig(self.meb_plot_dir + str(date.today()) + ' emotion_history.svg', format='svg')
        
        # It shows a plot of the emotion, activity, and location data recorded by each participant daily.
        for c, user in enumerate(emotionc.find().distinct("deviceID")):
            this_data = emotiondf.loc[emotiondf['imei'] == int(user)]
            if (this_data.size > 0):
                this_data.plot(x = 'month_day', y = 'count',
                                     label = self.users(user), fontsize=7,
                                     color = self.color_sequence[c])
                plt.title('Data record of emotion, activity, and location per participant', fontsize=8)
                plt.savefig(self.meb_plot_dir + str(user) + ' emotion.svg', format='svg')
                plt.close()  

#%% 
    def users(self, user):
        """
        It creates a participant's dictionary with the names and IMEI of their mobile devices.
        Parameters
        ----------
        k : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION. 
        """
        imeiID = {
            1: 'Experiment participant 1',
            2: 'Experiment participant 2',
            3: 'Experiment participant 3',
            4: 'Experiment participant 4',
            5: 'Experiment participant 5',
            6: 'Experiment participant 6',
            7: 'Experiment participant 7',
            8: 'Experiment participant 8',
            9: 'Experiment participant 9',
            10: 'Experiment participant 10',
            11: 'Experiment participant 11',
            12: 'Experiment participant 12',
            13: 'Experiment participant 13',
            14: 'Experiment participant 14',
            15: 'Experiment participant 15',
            16: 'Experiment participant 16',
            17: 'Experiment participant 17',
            18: 'Experiment participant 18'
        }
        return imeiID.get(int(user))