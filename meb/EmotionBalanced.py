#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Tue Dec 8 20:33:32 2019

@authors: 
    Luz Santamaría Granados (luzsantamariag@gmail.com)
    Juan Francisco Mendoza Moreno (jfmendozam@gmail.com)
"""

import numpy as np
import pandas as pd
from imblearn.over_sampling import KMeansSMOTE
from imblearn.under_sampling import TomekLinks 
from keras.utils.np_utils import to_categorical
import seaborn as sns
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split


class EmotionBalanced:
    """ 
    It checks the Imbalance in the emotional slicing dataset for the training stage.
    """
   
    def __init__(self):
        """ Default constructor 
        """                    
        self.syntheticData = True           # Synthetic data for emotion class
        self.emotionMinSlices = 10          # Minimum of slices for each emotion class
        self.participant = 6                # Participant's in the experiment
        self.datAverage = False             # Data average for emotion class
        self.classMinNum = 4                # Emotion classes minimum number  
    
#%%
    def getEmotionalSlicingData(self, features, classType):
        """
        Filter the data of emotional segments of the experiment participants according to the
        label's class: emotion state, emotional quadrant, and activity type.
        Parameters
        ----------
        features : List
            DESCRIPTION. Emotional Slicing features: 0 imei, 1 slicesCounter, 2 hr, 3 instancesCounter, 4 sliceDuration, 
                         5 activity, 6 catActivity, 7 catEmotion, 8 emotion, 9 emotionInstances, 10 catVA, 
                         11 VA, 12 ts, 13 emotionts, 14 hr_sliceSize_norm, 15 longitude, 16 latitude,
                         17 resampled_hr, 18 duplic_TimeSeries_hr  
        classType : int
            DESCRIPTION. Label class for data balancing: (0) emotion, (1) AV, and (2) activity.
        Returns
        -------
        hr_slice : list
            DESCRIPTION. Emotional Slicing

        """
        hr_slice = [] 
        hr_tmp = []
          
        for i in range(len(features)): # Cycle through the items in the emotional slicing list.
            #Is there an experiment participant with an emotional state label?
            
            if classType == 0:            
                if features[i][0] == self.participant and features[i][18]:
                    hr_tmp = features[i][18].copy()                       
                    hr_tmp.append(features[i][7])  # Categorical emotion  
                    hr_tmp.append(features[i][6])  # Categorical activity        
                    hr_tmp.append(features[i][4])  # Slice duration      
                    hr_tmp.append(features[i][5])  # Activity     
                    hr_tmp.append(features[i][8])  # Emotion   
                    hr_slice.append(hr_tmp)
                
                # Are there experiment participants with the same emotional state label?   
                if self.participant == -1 and  features[i][18]:   
                    hr_tmp = features[i][18].copy()                                                       
                    hr_tmp.append(features[i][7])  # Categorical emotion  
                    hr_tmp.append(features[i][6])  # Categorical activity        
                    hr_tmp.append(features[i][4])  # Slice duration      
                    hr_tmp.append(features[i][5])  # Activity     
                    hr_tmp.append(features[i][8])  # Emotion
                    hr_slice.append(hr_tmp)
                    
            elif classType == 1:                      
                # Is there an experiment participant with an emotional quadrant label?
                if features[i][0] == self.participant and features[i][8] != 'neutral' and features[i][18]:
                    hr_tmp = features[i][18].copy()
                    hr_tmp.append(features[i][10]) # Arousal and Valence quadrant number 
                    hr_tmp.append(features[i][7])  # Categorical emotion   
                    hr_tmp.append(features[i][4])  # Slice duration      
                    hr_tmp.append(features[i][11]) # Valence and Arousal quadrant      
                    hr_tmp.append(features[i][8])  # Emotion   
                    hr_slice.append(hr_tmp) 
                 
                # Are there experiment participants with the same emotional quadrant label? 
                if self.participant == -1 and features[i][8] != 'neutral' and features[i][18]:       
                    hr_tmp = features[i][18].copy()
                    hr_tmp.append(features[i][10]) # Arousal and Valence quadrant number 
                    hr_tmp.append(features[i][7])  # Categorical emotion   
                    hr_tmp.append(features[i][4])  # Slice duration      
                    hr_tmp.append(features[i][11]) # Valence and Arousal quadrant      
                    hr_tmp.append(features[i][8])  # Emotion     
                    hr_slice.append(hr_tmp)
        
            else:  
                # Is there an experiment participant with an activity label?   
                if features[i][0] == self.participant and features[i][18]:    
                    hr_tmp = features[i][18].copy()
                    hr_tmp.append(features[i][6])  # Categorical activity
                    hr_tmp.append(features[i][7])  # Categorical emotion   
                    hr_tmp.append(features[i][4])  # Slice duration      
                    hr_tmp.append(features[i][5])  # Activity     
                    hr_tmp.append(features[i][8])  # Emotion    
                    hr_slice.append(hr_tmp)
                    
                # Are there experiment participants with the same activity label?  
                if self.participant == -1 and features[i][18]:     
                    hr_tmp = features[i][18].copy()
                    hr_tmp.append(features[i][6])  # Categorical activity
                    hr_tmp.append(features[i][7])  # Categorical emotion   
                    hr_tmp.append(features[i][4])  # Slice duration      
                    hr_tmp.append(features[i][5])  # Activity     
                    hr_tmp.append(features[i][8])  # Emotion   
                    hr_slice.append(hr_tmp)
                
        return hr_slice
        
#%%  
    def getDataHeader(self, classType, sliceSize):
        """
        Defines the header of the emotional slicing data.  
        Parameters
        ----------
        classType : int
            DESCRIPTION. Label class for data balancing: (0) emotion, (1) AV, and (2) activity.
        sliceSize : int
            DESCRIPTION. Slice size of the HR instances
        Returns
        -------
        header : dictionary
            DESCRIPTION. Header of the emotional slicing data

        """
        header = {}
        for i in range(0, sliceSize):
            label = 'hr' + str(i)
            header.update({i: label})
        
        if classType == 0: # emotion label
            header.update({sliceSize: 'emotion'})    
            header.update({sliceSize + 1: 'activity'})             
            header.update({sliceSize + 2: 'duration'}) 
            header.update({sliceSize + 3: 'activities'})                 
            header.update({sliceSize + 4: 'emotions'})   
        
        if classType == 1: # AV quadrant label
            header.update({sliceSize: 'VAquadrant'})  
            header.update({sliceSize + 1: 'emotion'})               
            header.update({sliceSize + 2: 'duration'})       
            header.update({sliceSize + 3: 'VA'})    
            header.update({sliceSize + 4: 'emotions'})              
        
        if classType == 2: # activity label
            header.update({sliceSize: 'activity'})    
            header.update({sliceSize + 1: 'emotion'})               
            header.update({sliceSize + 2: 'duration'}) 
            header.update({sliceSize + 3: 'activities'})   
            header.update({sliceSize + 4: 'emotions'}) 
        
        return header
    
#%%    
    def buildDataBalanced(self, features, syntheticData, emotionMinSlices, imbalancedType, participant,
                          model_dir, classMinNum, classType, cluster_balance, sliceSize, split):
        """
         It defines the data balance algorithm for the classes of the participants' emotion labels.       

        Parameters
        ----------
        features : List
            DESCRIPTION. A dataset with heart rate features and emotion labels classes.
        syntheticData : Boolean
            DESCRIPTION. Synthetic data for emotion class          
        emotionMinSlices : Int
            DESCRIPTION. Minimum of slices for each emotion class           
        imbalancedType : int
            DESCRIPTION. Define the algorithm for imbalanced datasets
        participant : Int
            DESCRIPTION. Participant's in the experiment without anonymous    
        model_dir : String
            DESCRIPTION. Directory name.            
        classMinNum : Int
            DESCRIPTION. Emotion classes minimum number      
        classType : int
            DESCRIPTION. Label class for data balancing: (0) emotion, (1) AV, and (2) activity.
        cluster_balance : float
            DESCRIPTION. Cluster balance value              
        sliceSize : int
            DESCRIPTION. Slice size of the HR instances   
        split : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        self.modelDir = model_dir
        self.emotionMinSlices = emotionMinSlices
        self.participant = participant
        self.emotionLabel = {}
        self.classType = classType
        
        self.headers = self.getDataHeader(classType, sliceSize)
        self.hr_slice = self.getEmotionalSlicingData(features, classType)
        
        if participant == -1: 
            print ('Participants have ' + str(len(self.hr_slice)) + ' emotional slicings')  
        else: 
            print ('Participant ' +str(participant) + ' has ' + str(len(self.hr_slice)) + ' emotional slicings') 
        
        if len(self.hr_slice) > 0:
            self.x, self.y = self.dataLoad(classMinNum, sliceSize, classType) # Load and dimension the dataframe
            
            if self.classNum >= classMinNum:   # Are there enough emotional slicing instances in label classes?  
                if len(self.x) > 1 and len(self.y) > 1:        
                    self.xyData = self.dataShow(self.x, self.y)
                    self.emotionSlicesNum = self.emotionSlNum
                    self.dataBalanced(syntheticData, cluster_balance, split, imbalancedType) # Generate the syntetic data
            else:
                print('Participant ' + str(participant) +
                          ' does not have enough Emotional Slicing instances in label classes')                    
        else:
            print("The participant hasn't emotion slices data.")


#%%        
    def dataLoad(self, classMinNum, sliceSize, classType):
        """
        Load and dimension the dataset in a dataframe. 
        Parameters
        ----------
        classMinNum : Int
            DESCRIPTION. Emotion classes minimum number  
        sliceSize : int
            DESCRIPTION. Slice size of the HR instances   
        classType : int
            DESCRIPTION. Label class for data balancing: (0) emotion, (1) AV, and (2) activity.

        Returns
        -------
        features : list
            DESCRIPTION. HR instances
        labels : list
            DESCRIPTION. Class label

        """
        self.classNum = 0
        self.df = pd.DataFrame(self.hr_slice) # Creates the DataFrame
        self.df.rename(index = str, columns = self.headers, inplace = True)
        self.df_hr = self.df.iloc[:,0:sliceSize + 1]  # Select heart rate and emotion columns.
        
        if classType == 0:
            self.label = 'emotion'
            self.labels = 'emotions'        
        elif classType == 1:
            self.label = 'VAquadrant'
            self.labels = 'VA'
        else:
            self.label = 'activity'
            self.labels = 'activities'
                
        values = self.df_hr[self.label].value_counts().tolist()  
        for i in range(len(values)):   # Are there enough emotional slicing instances in label classes?  
            if values[i] >= self.emotionMinSlices:
                self.classNum = self.classNum + 1
    
        ds = self.emotionSlicesCheck() # Check the slices minimum for each emotion class
        
        if len(self.durationList) >= classMinNum:
                
            plt.figure(figsize=(20,14))
            self.plot_correlation('Dataset Correlation', self.df_hr)     # dataset correlation heatmap
            plt.savefig(self.modelDir + 'correlation.svg', format='svg')
            plt.close()
            
            self.data = ds.astype('float32')
                    
            features = self.data[:, :-1]
            labels = self.data[:, -1]
        else:
            features = 0 # x = 0
            labels = 0   #y = 0 

        return features, labels    

#%%      
    def emotionSlicesCheck(self):
        """
        It checks the slices minimum for each label class. By default,
        the slice minimum value is 10. It eliminates the slices of the
        Dataframe that are minors by label class.   

        Returns
        -------
        label : array
            DESCRIPTION. Heart rate instances and emotion label 

        """
        self.df_emotion = pd.DataFrame(columns = self.df_hr.columns.values.tolist())
        values = self.df_hr[self.label].value_counts()       # label: emotion, AVquadrant or activity
        print(values)
        self.duration = self.df.groupby([self.label, self.labels]).sum()['duration']   # labels: emotions, AV or activities
        self.durationList = []

        # Filter the emotion classes according to the number of segments per class.
        for emotion in values.index:
            if values[emotion] >= self.emotionMinSlices:
                # Add a row to emotion Dataframe    
                self.df_emotion = self.df_emotion.append(self.df_hr.loc[self.df_hr[self.label] == emotion], ignore_index = True)
                
                # Add the data to the list and dictionary of the emotion class  
                for i in range(len(self.duration)):
                    if emotion == self.duration.index[i][0]:
                        self.durationList.append([self.duration.index[i][0], self.duration.index[i][1], int(self.duration[emotion][0]), values[emotion]])        
                        self.emotionLabel[self.duration.index[i][0]] = self.duration.index[i][1] 
                        break;
                                 
        print (self.df_emotion[self.label].value_counts())
        label = self.df_emotion.values  
        
        return label
    
#%%      
    def labelsAdjustment(self, y):
        """
        It Checks the index number of the emotion labels and adjust them
        consecutively. That is, it does not include classes that have 0 instances.       
        Parameters
        ----------
        y : array
            DESCRIPTION. Data of label class

        Returns
        -------
        label : array
            DESCRIPTION. Data of label class

        """
        self.targetClass = []
        self.y_labelsAdj = np.full((y.shape[0],len(self.emotionLabel)),0)
                           
        emotion = 0
        for i in range(y.shape[1]):  # Columns 2D array
            for k, v in self.emotionLabel.items():
                if (k == i):
                    self.y_labelsAdj[:,emotion] = y[:,i]
                    emotion = emotion + 1
                    self.targetClass.append(v)   

        label = self.y_labelsAdj

        return label

#%%          
    def split_train_normal(self, split = 0.8):
        """
        It divides the features arrays and labels vector into training and test data.
        Parameters
        ----------
        split : float
            DESCRIPTION. The default is 0.8.

        Returns
        -------
        None.

        """
        y_label = self.y_labels
        label = self.labelsAdjustment(y_label)
        np.random.seed(7)   # Shuffle emotion labels and heart rate features slices.
        values = [i for i in range(len(self.x_features))]
        permutations = np.random.permutation(values)
        x_set = self.x_features[permutations, : ]
        label_set = label[permutations, : ]

        size = len(x_set)
        self.X_train = x_set[ : int(split * size), : ]
        self.Y_train = label_set[ : int(split * size), : ]
        self.X_val = x_set[int(split * size) : , : ]
        self.Y_val = label_set[int(split * size) : , : ]
        
#%%      
    def split_train_balanced(self, split, imbalancedType, cluster_balance):
        """
        It applies the algorithms for balancing minority classes on the training data.
        Parameters
        ----------
        split : int
            DESCRIPTION. The default is 0.8. Percentage of training data
        imbalancedType : int
            DESCRIPTION. Define the algorithm for imbalanced datasets
        cluster_balance : float
            DESCRIPTION. Cluster balance value  
        Returns
        -------
        None.

        """        
        X_train, X_test, y_train, y_test = train_test_split(
            self.x, self.y, train_size = split, random_state=0)        
        
        if imbalancedType == 0: # Method 1. Only SMOTE (oversampling)
            sm = KMeansSMOTE(random_state = 0, cluster_balance_threshold = cluster_balance) 
            x_res, y_res = sm.fit_resample(X_train, y_train)        

        if imbalancedType == 1: # Method 2. TomekLinks (oversampling + cleaning)
            sm = KMeansSMOTE(random_state = 0, cluster_balance_threshold = cluster_balance) 
            x_res_clean, y_res_clean = sm.fit_resample(X_train, y_train)
            sm = TomekLinks()    
            x_res, y_res = sm.fit_resample(x_res_clean, y_res_clean)   

        self.X_train, ytrain = np.expand_dims(x_res, axis=2), to_categorical(y_res)
        self.Y_train = self.labelsAdjustment(ytrain)
        
        self.X_val, yval = np.expand_dims(X_test, axis=2), to_categorical(y_test)  
        self.Y_val = self.labelsAdjustment(yval) 
    
   
#%% 
    def dataBalanced(self, syntheticData, cluster_balance, split, imbalancedType):
        """
        It uses the SMOTE algorithm to balance the training set. It also dimensions
        the features array and converts the labels classes to categorical data.   
        Parameters
        ----------
        syntheticData : Boolean
            DESCRIPTION. Synthetic data for emotion class 
        cluster_balance : float
            DESCRIPTION. Cluster balance value  
        split : int
            DESCRIPTION. The default is 0.8. Percentage of training data
        imbalancedType : int
            DESCRIPTION. Define the algorithm for imbalanced datasets

        Returns
        -------
        None.

        """

        X_balanced = self.x.reshape(self.x.shape[0], self.x.shape[1])
        Y_balanced = self.y.reshape(self.x.shape[0])
        self.dataShow(X_balanced, Y_balanced)
        self.x_features, self.y_labels = np.expand_dims(X_balanced, axis=2), to_categorical(Y_balanced)
        
        if syntheticData:                  # Are the syntethics data? 
            self.split_train_balanced(split, imbalancedType, cluster_balance)              
        else:
            self.split_train_normal(split)
                 
#%%    
    def plot_correlation(self, title, data): 
        """
        This method generates the dataset correlation heatmap.
        Parameters
        ----------
        title: String
            Plot title 
        data : DataFrame
            Dataset in a dataframe
        Returns
        -------
        None.
        """
        data = data.round(1)
        corr = data.corr()
        ax = sns.heatmap(corr, 
                    xticklabels=corr.columns.values,
                    yticklabels=corr.columns.values, annot=True, fmt = ".1g",
                    cmap="Blues", rasterized=True)
        ax.set_title(title, fontsize = 20)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        plt.setp(ax.get_yticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        bottom, top = ax.get_ylim()
        ax.set_ylim(bottom + 0.5, top - 0.5)
   
#%% 
    def dataShow(self, x, y):
        """
        It prints the results of the balanced dataset.       
        Parameters
        ----------
        x : array
            DESCRIPTION. Features
        y : array
            DESCRIPTION. Labels
        Returns
        -------
        label : array
            DESCRIPTION. Labels
        """
        print("Features: ", x.shape)
        print("Label: ", y.shape)
        print("Emotion classes slices:")
        self.emotionSlNum = []
        label = np.bincount(y.astype('int32').flatten())
        for i in label:
            if i != 0:
                self.emotionSlNum.append(i)
                print(i, end =" ")
        print("\n")
        return label