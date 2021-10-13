#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Tue Nov 1 20:33:32 2019

@authors: 
    Luz Santamaría Granados (luzsantamariag@gmail.com)
    Juan Francisco Mendoza Moreno (jfmendozam@gmail.com)
"""

from datetime import datetime

class Preprocessing:
    """ Consolidates and preprocesses the dataset of the participants.
    """
   
    def __init__(self):
        """ Default constructor 
        """
        self.hrDir = 'miband/'                # Heart rate data subdirectory
        self.mebData = {}                     # Dictionary of myemotionband collection.
        self.hrData = {}                      # Dictionary of hrband collection.
        self.max_size_window = 180            # Maximum size window for time-series data tagging algorithm.
 
#%%        
    def preprocessDataset(self, mebData, hrData, max_size_window):
        """
        Consolidates and preprocesses the emotion, activity, location, and 
        physiological signals dataset of the participants.         
        Parameters
        ----------
        mebData : dictionary
            DESCRIPTION. Dictionary of myemotionband collection.
        hrData : dictionary
            DESCRIPTION.
        max_size_window : int
            DESCRIPTION. Maximum size window for time-series data tagging algorithm
        Returns
        -------
        None.
        """        
        self.mebData = mebData
        self.hrData = hrData
        self.max_size_window = max_size_window         
        self.getMebData()
        self.getTagDataset()


#%% 
    def getMebData(self):
        """
        It generates a dictionary of the participants' emotions collection 
        filtered by timestamp's feature.
        Returns
        -------
        None.
        """
        self.emotionReport = []
        self.emotion = {}                         
        for k, v in self.mebData.items():
            temp = {}
            for k1, v1 in v.items():
                temp.update({v1['eTimeStamp']: v1})
            self.emotion.update({k: temp})
            # Generates a list of emotions per participant with unduplicated timestamp key.
            self.emotionReport.append([k,len(v),len(temp)])  
      
#%% 
    def getTagDataset(self):
        """
        Sliding and adjustable window for time-series data tagging algorithm.         
        Returns
        -------
        None.
        """

        reg = 0
        start_hour = datetime.now()
        
        # Prepare two dataset
        for k, v in self.hrData.items(): #Iterate the loop through the heart rate dictionary of the experiment participants.
            print('imei: ' + str(k))
            hr = {} 
            
            for hrTimestamp, heartRate in v.items(): # Get a timestamp of the heart rate register.
                reg = reg + 1 #tag = tag + 1
                print('reg: ' + str(reg))
                #print('ts: ' + str(hrTimestamp) + ' hr: ' + str(heartRate))
                window_delta = 0
        
                for k1, v1 in self.emotion.items():  # Iterate the loop through the emotion, activity, and location dictionary of the experiment participants.
                       
                    if (k1 == k):                    # Are the IMEI the same in both datasets?
                        
                        for key, value in v1.items(): # Get the items of the emotion instance by each participant.
                            
                            # Does the maximum size window for labeling time series data are less than the window's delta time?
                            while (window_delta <= self.max_size_window): 
                                
                                window_start =  hrTimestamp - window_delta   # Adjust the window start
                                window_end   =  hrTimestamp + window_delta   # Adjust the window end
                                find = False    # Control variable for data tagging. 
                                
                                for eTimestamp in v1.keys():  # Get the items of the emotion instance
                                    # Control of the sliding and adjustable window for time-series data tagging.
                                    if (window_start <= eTimestamp <= window_end): 
                                        e = v1[eTimestamp]['emotion']
                                        activity = v1[eTimestamp]['activity']
                                        longitude = v1[eTimestamp]['longitude']
                                        latitude = v1[eTimestamp]['latitude']
                                        
                                        hr.update({hrTimestamp: {
                                                'imei': k,
                                                'emotionts': eTimestamp,
                                                'emotion': e,
                                                'activity': activity,
                                                'hr': heartRate,
                                                'longitude': longitude,
                                                'latitude': latitude
                                                                 }}); 
                                        find = True   # Label the emotion in the instances of time series of the heart rate
                                        break;
                            
                                if find:
                                    break;
                                else:
                                    hr.update({hrTimestamp: {'imei': k, 'hr': heartRate}})
                                window_delta = window_delta + 1
            
            self.hrData.update({k: hr})           # tag hrData Update
            end_hour = datetime.now()
            print(start_hour, end_hour)
