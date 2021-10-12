#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Tue Dec 1 20:33:32 2019

@authors: 
    Luz Santamaría Granados (luzsantamariag@gmail.com)
    Juan Francisco Mendoza Moreno (jfmendozam@gmail.com)
"""

from collections import Counter
from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')


class EmotionalSlice:
    
    """ 
    This class performs the slicing of emotion and heart rate instances
    of the experiment participants. Besides, it generates synthetic data 
    of the HR vector according to the size of the emotional slice.
    """
   
    def __init__(self):
        """
        Default constructor 
        Returns
        -------
        None.
        """        
        self.hr_plot_dir            = 'miband/plot/'    # Heart rate plot subdirectory
        self.hrData                 = {}                # Heart rate dataset tagged with emotion class
        self.timeBetweenInstances   = 60                # Time distance between instances (seconds) 
        self.sliceSize              = 30                # Slice size 
        self.sliceLimit             = 3                 # Slice size limit          
 
#%% dataLoad

    def dataLoad(self):
        """
        It loads the tagged dataset from a dictionary to a list.
        tagHrList: imei, hrTimestamp, emotionts, emotion, activity,
        hr, longitude, latitude, and altitude
        Returns
        -------
        None.
        """
   
        self.tagHrList = [] 
        for k, v in self.hrData.items():
            for k1, v1 in v.items():
                if 'emotion' in v1: 
                    temp = []
                    for key, value in v1.items():
                        temp.append(value)
                    self.tagHrList.append([int(temp[0]), int(k1), int(temp[1]), temp[2],
                                           temp[3], int(temp[4]), float(temp[5]),
                                           float(temp[6])])                    

#%% initSlice
                    
    def initSlice(self):
        """ Init a slice
        """
        self.start = int(self.previous[1])  # First time-stamp
        self.hr = []                        # Heart rate list initialization
        self.ts = []                        # Time-stamp list initialization
        self.emotions = []                  # Emotion list initialization
        self.instancesCounter = 0           # Instances counter
    
#%% hr_norm
    
    def hr_norm(self, hr_data):
        """
        Normalize heart rate data  
        Parameters
        ----------
        hr_data : List
            DESCRIPTION. Heart rate data
        Returns
        -------
        list
            DESCRIPTION. Normalized heart rate
        """
        min_value = list(filter(lambda x: (x[0] == self.previous[0]), self.minmax))[0][1]
        max_value = list(filter(lambda x: (x[0] == self.previous[0]), self.minmax))[0][2]        
        return ([(i - min_value) / (max_value - min_value) for i in hr_data])
            
    
#%% duplication_ts method

    def duplication_ts(self, hr_norm, sliceSize):
        """
        Generates the duplication of time series values to adjust the number of HR instances
        according to the fixed size of the Emotional Slicing.
        Parameters
        ----------
        hr_norm : list
            DESCRIPTION. Heart Rate list normalized
        sliceSize : int
            DESCRIPTION. Slice size
        Returns
        -------
        hr : list
            DESCRIPTION. Heart Rate list with duplication of time series values
        """
        
        hr = np.zeros(sliceSize)
        i = 0
        flag = True
        
        while flag:
            if sliceSize - len(hr_norm) <= 0:
                segment = np.abs(sliceSize - len(hr_norm))       
                hr[i] = hr_norm[i:segment]  # it Loses the initial segment of the signal data
            else:
                b = hr_norm[0 : sliceSize - len(hr_norm)]
                goal = np.hstack((hr_norm, b))
                while len(goal) != sliceSize:
                    b = hr_norm[0 : sliceSize - len(goal)]
                    goal = np.hstack((goal, b))
                hr = goal
            i += 1
            flag = False 
        return hr                
    

#%% closeSlice

    def closeSlice(self):
        """
        It closes a slice conformed by instances of heart rate and timestamp
        with emotion, activity, and location data.
        Returns
        -------
        None.
        """

        if (self.instancesCounter > self.sliceLimit):
            sliceDuration = int(self.ts[self.instancesCounter - 1]) - int(self.ts[0])
            
            emotionsCounter = Counter(self.emotions)
            hrCounter = Counter(self.hr)
            if len(hrCounter) > 1:
                
                self.predominantEmotion = max(zip(
                        emotionsCounter.values(),   # Emotion counting
                        emotionsCounter.keys()      # Emotion name
                ))
                
                if self.instancesCounter < self.sliceSize:
                    self.ohr = self.hr
                    self.ots = self.ts
                    self.addSyntheticData()
                else:
                    self.slicePlot()               # It plots an emotional slicing with a fixed size of HR instances. 
                    self.ohr = []
                    self.ots = []
                                
                self.labels.append([
                        self.current[0],            # IMEI
                        self.slicesCounter,         # Slice consecutive
                        self.catEmotion(),          # Categorical emotion
                        self.predominantEmotion[1], # Emotion name
                        self.predominantEmotion[0]  # Total instances with this emotion
                ])
                             
                self.hr_sliceSize_norm = self.hr_norm(self.hr) if len(self.hr) == self.sliceSize else []
                self.ohr_norm = self.hr_norm(self.ohr) if self.ohr else []
                quadrant = self.catVA()               # 10 Valence and Arousal quadrant 
                
                self.features.append([
                        self.previous[0],                           # 0 IMEI 
                        self.slicesCounter,                         # 1 Slice consecutive
                        self.hr,                                    # 2 Heart rate list
                        self.instancesCounter,                      # 3 Instances number 
                        sliceDuration,                              # 4 Slice duration (seconds)                 
                        self.previous[4],                           # 5 Activity name
                        self.catActivity(),                         # 6 Categorical activity
                        self.catEmotion(),                          # 7 Categorical emotion
                        self.predominantEmotion[1],                 # 8 Emotion name
                        self.predominantEmotion[0],                 # 9 Emotion instances total
                        list(quadrant.keys())[0],                   # 10 Valence and Arousal quadrant number
                        quadrant.get(list(quadrant.keys())[0]),     # 11 Valence and Arousal quadrant 
                        self.ts,                                    # 12 Heart Rate time-stamp
                        int(self.previous[2]),                      # 13 Emotion time-stamp
                        self.hr_sliceSize_norm,                     # 14 Normalized Heart rate y with the same lenght of slice size                        
                        self.previous[6],                           # 15 longitude
                        self.previous[7],                           # 16 latitude 
                        # 17 Assigns normalized HR with resampled HR time series values if the segment size limit is lower than the slice size.
                        self.hr_sliceSize_norm if self.hr_sliceSize_norm else self.ohr_norm,  
                        # 18 Assigns normalized HR with duplicate time series values if the segment size limit is lower than the slice size.
                        list(self.hr_sliceSize_norm if self.hr_sliceSize_norm else self.duplication_ts(
                            self.hr_norm(self.hr), self.sliceSize)) 
                ])

                self.slicesCounter = self.slicesCounter + 1
                self.initSlice()
        
            else:
                self.initSlice()
        
        else:
            self.initSlice()
 
 #%% addInstanceToSlice           
    
    def addInstanceToSlice(self):
        """
        Add data instances to a slice.
        Returns
        -------
        None.
        """
        
        self.ts.append(self.current[1])
        self.hr.append(self.current[5])
        self.emotions.append(self.current[3])
        
        self.instancesCounter = self.instancesCounter + 1
        
        if self.instancesCounter >= self.sliceSize:
            self.closeSlice()
    
    
#%% buildSlices
       
    def buildSlices(self, slice_plot_dir, hrData, timeBetweenInstances, 
                    sliceSize, sliceLimit):       
        """
        Slice of emotions algorithm. Define the heart rate vectors with
        the emotional blocks and it balances the vector size with the 
        resampling of the signal through the Fourier series.

        Parameters
        ----------
        slice_plot_dir : String
            DESCRIPTION. Emotional slicing plot subdirectory
        hrData : dictionary
            DESCRIPTION. Heart rate dataset tagged with emotion class.
        timeBetweenInstances : int
            DESCRIPTION. Time distance between instances (seconds)
        sliceSize : int
            DESCRIPTION. Slice size
        sliceLimit : int
            DESCRIPTION. Slice size limit

        Returns
        -------
        None.
        """
        self.plotDir = slice_plot_dir
        self.hrData = hrData
        self.dataLoad()
        
        self.timeBetweenInstances = timeBetweenInstances  # Time distance between instances (seconds), default = 120
        self.features = []                   # Fatures list 
        self.labels = []                     # Labels list
        self.sliceSize = sliceSize           # Slice size, default = 20
        self.sliceLimit = sliceLimit         # Slice limit size, default = 2 
        self.slicesCounter = 0
        self.previous = self.tagHrList[0]
        self.initSlice()
        
        self.current = self.previous         # Save the first instance
        self.addInstanceToSlice()

        self.getMinMaxByImei()
        
        for row in range(1, len(self.tagHrList)):
            self.previous = self.tagHrList[row - 1]
            self.current = self.tagHrList[row]
            
            if self.previous[0] == self.current[0]:  # Are the IMEIs the same?
                if int(self.current[1]) - int(self.previous[1]) <= self.timeBetweenInstances: # Is the instance in the same slice?
                    if self.current[4].lower() == 'pelicula'.lower(): # Is the activity a movie?
                        if self.previous[3] != self.current[3]: # Is the emotion the same?
                            self.closeSlice()
                        self.addInstanceToSlice()
                    else:
                        self.addInstanceToSlice()
                else:
                    self.closeSlice()
                    self.addInstanceToSlice()
            else:
                self.closeSlice()
                self.addInstanceToSlice()

                    
#%% getMinMaxByImei

    def getMinMaxByImei(self):
        """
        Get the minimum and maximum heart rate values ​​of all participants.
        Returns
        -------
        None.

        """
        imeiList = list(set(map(lambda x:x[0], self.tagHrList)))
        newList = [[] for i in range(len(imeiList))]
    
        for imei, hrts, ets, e, a, hr, lo, la in self.tagHrList:
            newList[imeiList.index(imei)].append(hr)
        
        self.minmax = []
        for i in range(len(newList)):
            self.minmax.append([imeiList[i], min(newList[i]), max(newList[i])])

#%% addSyntheticData

    def addSyntheticData(self):
        """
        This method verifies slices of less than 30 instances and 
        adds synthetic data with resampling of heart rate and timestamp.
        Resample ts to num samples using Fourier method.
        Returns
        -------
        None.

        """
        x = np.asarray(self.ots)
        y = np.asarray(self.ohr)
        f1 = signal.resample(y, self.sliceSize)
        f = [int(round(i)) for i in f1]
        max_x = len(x) - 1
        xnew1 = np.linspace(x[0], x[max_x], self.sliceSize, endpoint = True)
        xnew = [int(i) for i in xnew1]
        self.ots = xnew
        self.ohr = f
        
        #self.resampledSlicePlot(x, y, f, xnew, max_x)   # It plots a adjusted slice 
        
#%% resampledSlicePlot

    def resampledSlicePlot(self, x, y, f, xnew, max_x):
        """
        This method generates the resampled heart rate and emotion slice plot.        
        Parameters
        ----------
        x : array
            DESCRIPTION. Timestamp
        y : array
            DESCRIPTION. Heart rate
        f : array
            DESCRIPTION. Heart rate resampled using Fourier method
        xnew : array
            DESCRIPTION. timestamp resampled
        max_x : int
            DESCRIPTION. Maximum timestamp
        Returns
        -------
        None.
        """

        plt.figure(figsize = (18, 8))
        plt.xticks(fontsize = 20)
        plt.yticks(fontsize = 20)
        plt.plot(x, y, 'go-', xnew, f, '.-', x[max_x], y[0], 'ro')
        text = 'Slice of ' + str(self.sliceSize)+' HR instances of participant ' + str(self.previous[0]) +' labeled "'+ self.predominantEmotion[1] + '"'
        plt.title(text, fontsize = 22)
        plt.xlabel('Time series', fontsize = 20)
        plt.ylabel('Heart rate', fontsize = 20)
        plt.legend(['data', 'Fourier method resampled'], loc='best', prop={'size': 22})
        plt.savefig(self.plotDir + 'hr_resampled_'+ str(self.slicesCounter) + '_' + str(self.previous[0]) + '.svg')
        plt.close()

#%% slicePlot

    def slicePlot(self):
        """
        This method plots an Emotional Slicing with a fixed size of HR instances. 
        Returns
        -------
        None.
        """
        x = [int(i) for i in self.ts]
        y = [int(round(i)) for i in self.hr]
        plt.figure(figsize=(18,8))
        plt.xticks(fontsize = 20)
        plt.yticks(fontsize = 20)
        text = 'Slice of ' + str(self.sliceSize)+' HR instances of participant ' + str(self.previous[0]) +' labeled "'+ self.predominantEmotion[1] + '"'
        plt.title(text, fontsize = 18)
        plt.xlabel('Time series', fontsize = 20)
        plt.ylabel('Heart rate', fontsize = 20)
        plt.plot(x, y)
        plt.savefig(self.plotDir + 'hr_'+ str(self.slicesCounter) + '_' + str(self.previous[0]) + '.svg')
        plt.close()

    #%% Categorical imei
            
    def catEmotion(self):
        """
        Convert nominal emotion to categorical emotion.
        Returns
        -------
        catEmotion : int
            DESCRIPTION. Categorical emotion
        """
        emotionID = {
            'feliz': 0,              
            'divertido': 1,          
            'alegre': 2,            
            'contento': 3,           
            'satisfecho': 4,        
            'calmado': 5,           
            'relajado': 6,           
            'cansado': 7,            
            'aburrido':8,            
            'deprimido':9,           
            'avergonzado':10,        
            'triste':11,             
            'estresado':12,          
            'asustado':13,           
            'enojado':14,            
            'en pánico':15,          
            'neutral':16             
        }        
        return (emotionID.get(self.predominantEmotion[1]))


    #%% Categorical Valence and Arousal quadrant
                
    def catVA(self):          
        """
        It groups emotions by arousal and valence quadrant.
        Returns
        -------
        quadrant : int
            DESCRIPTION. Categorical quadrant
        """
        quadrant = {}
        emotion = self.catEmotion()  
        if emotion >= 0 and emotion <= 3:  
            quadrant = {0: 'HVHA'}
        elif emotion >= 4 and emotion <= 7:
            quadrant = {1: 'HVLA'}
        elif emotion >= 8 and emotion <= 11: 
            quadrant = {2: 'LVLA'}            
        elif emotion >= 12 and emotion <= 15: 
            quadrant = {3: 'LVHA'} 
        else:
            quadrant = {4: 'Neutral'}          
        return (quadrant)
        
 #%% Categorical activity
            
    def catActivity(self):
        """
        Convert nominal activity to categorical activity.
        Returns
        -------
        catActivity : int
            DESCRIPTION. Categorical activity
        """ 
        activityID = {
            'trabajar': 0,
            'estudiar': 1,
            'leer': 2,
            'ejercitar': 3,
            'conducir': 4,
            'cita': 5,
            'descansar': 6,
            'comprar': 7,
            'amigos':8,
            'festejar':9,
            'alimentar':10,
            'pelicula':11,
            'limpiar':12,
            'viajar':13,
            'jugar':14,
            'dormir':15,
            'asearse':16,
            'caminar': 17,
            'esperar': 18,
            'casa': 19,
            'cuidarse': 20
        }
        return (activityID.get(self.previous[4]))
  