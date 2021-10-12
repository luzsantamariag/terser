#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Tue Dec 24 01:58:15 2019

@authors: 
    Luz Santamaría Granados (luzsantamariag@gmail.com)
    Juan Francisco Mendoza Moreno (jfmendozam@gmail.com)
"""

from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from tensorflow.keras.callbacks import ModelCheckpoint 
import pandas as pd
import numpy as np
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dropout
from tensorflow.keras.layers import Dense, LSTM, Flatten
from tensorflow.keras import optimizers
from tensorflow.keras.layers import GlobalAveragePooling1D
from matplotlib.ticker import MaxNLocator
import matplotlib.pyplot as plt
from tensorflow.keras.utils import plot_model


class EmotionRecognition:
    """ 
    Recognize the affective state of the experiment participants using HR vectors.
    """
    
    def __init__(self, X, X_train, Y_train, X_val, Y_val):
        """ Default constructor 
        """
        self.X = X
        self.X_train =  X_train
        self.Y_train = Y_train
        self.X_val = X_val
        self.Y_val = Y_val
        
#%%         
    def create_directory(self, model_dir = 'models/'):
        """
        Valid that the directory exists.
        Parameters
        ----------
        model_dir : TYPE, optional
            DESCRIPTION. The default is 'models/'.
        """

        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

         
#%% 
    def convert_vector(self, x):
        """
        Converts a binary class matrix to class vector.
        Parameters
        ----------
        x : TYPE
            DESCRIPTION.
        Returns
        -------
        label_mapped : array 
            DESCRIPTION. Integer vector
        """
        label_mapped = np.zeros((np.shape(x)[0]))
        for i in range(np.shape(x)[0]):
            max_value = max(x[i, : ])
            max_index = list(x[i, : ]).index(max_value)
            label_mapped[i] = max_index
            
        return label_mapped.astype(np.int)
    
#%%         
    def create_model(self, number_of_classes = 4, filters = 128, kernel_size = [10, 4, 1, 1],
                      pool_size = [2, 1, 1], drop_rate = 0.5, neurons = [256, 128, 64],
                      model_filename = 'best_model.hdf5', history_filename = 'history.csv',
                      batch = 32, number_epochs = 50, modelType = 0):
        """
        It creates an emotion recognition model based on DNN algorithms.
        Parameters
        ----------
        number_of_classes : int
            DESCRIPTION. Number of labels classes
        filters : int
            DESCRIPTION. The dimensionality of the output space in the convolution
        kernel_size : list
            DESCRIPTION. Specify the length of the 1D CNN window
        pool_size : list
            DESCRIPTION. Size of the max-pooling windows for the operation of the CNN layers
        drop_rate : int
            DESCRIPTION. Define input unit less than 1
        neurons : int
            DESCRIPTION. Specify the dimensionality of the array output space for a FCN
        model_filename : String
            DESCRIPTION. Model file name
        history_filename : String
            DESCRIPTION. Accuracy results from all epochs of training and testing of the model
        batch : int
            DESCRIPTION. Batch size
        number_epochs : int
            DESCRIPTION. Define the iterations number on the dataset during the training time.
        modelType : int
            DESCRIPTION. DNN model
        Returns
        -------
        None.
        """

        if modelType == 0:            # Test model 1: 1D CNN Fatten
            self.model = Sequential()
            self.model.add(Conv1D(filters, kernel_size[0], activation = 'relu',
                                  input_shape = self.X.shape[1:]))   # input_shape = self.X.shape[1:]
            #self.model.add(Dropout(drop_rate))            
            self.model.add(MaxPooling1D(pool_size[0]))
            self.model.add(Flatten())   
            self.model.add(Dense(neurons[0], kernel_initializer = 'normal', activation = 'relu'))            
            self.model.add(Dropout(drop_rate))         

        if modelType == 1:            # Test model 2: 1D CNN LSTM         
            
            self.model = Sequential()
            self.model.add(Conv1D(filters, kernel_size[0], activation='relu',
                                  input_shape = self.X.shape[1:]))
            self.model.add(MaxPooling1D(pool_size[0]))    
            self.model.add(Dropout(drop_rate))
            self.model.add(LSTM(neurons[0] * 2, activation = 'relu'))    
            self.model.add(Dropout(drop_rate))                         
         
        if modelType == 2:       # Test model 3: DCNN
            self.model = Sequential()
            self.model.add(Conv1D(filters, kernel_size[0], activation = 'relu', 
                                  input_shape = self.X.shape[1:]))   # input_shape = self.X.shape[1:]
            self.model.add(MaxPooling1D(pool_size[0]))
            self.model.add(Dropout(drop_rate))   
    
            self.model.add(Conv1D(filters, kernel_size[1], activation = 'relu'))
            self.model.add(MaxPooling1D(pool_size[1]))
            self.model.add(Dropout(drop_rate))
    
            self.model.add(Conv1D(filters, kernel_size[2], activation = 'relu'))
            self.model.add(MaxPooling1D(pool_size[2]))
            self.model.add(Dropout(drop_rate))
    
            self.model.add(Conv1D(filters, kernel_size[3], activation = 'relu'))
            self.model.add(GlobalAveragePooling1D())
            
            self.model.add(Dense(neurons[0], kernel_initializer = 'normal',
                                  activation = 'relu'))
            self.model.add(Dropout(drop_rate))
    
            self.model.add(Dense(neurons[1], kernel_initializer = 'normal',
                                  activation = 'relu'))
            self.model.add(Dropout(drop_rate))
    
            self.model.add(Dense(neurons[2], kernel_initializer = 'normal',
                                  activation = 'relu'))
            self.model.add(Dropout(drop_rate))
        
        self.model.add(Dense(number_of_classes, kernel_initializer = 'normal',
                              activation = 'softmax'))
        optimize = optimizers.Adam(lr=0.001)
        self.model.compile(loss = 'categorical_crossentropy', 
                            optimizer = optimize, metrics = ['accuracy'])
        
        self.checkpointer = ModelCheckpoint(filepath = model_filename,
                                        monitor = 'val_accuracy', verbose = 1,
                                        save_best_only = True, mode='max')
        
        self.hist = self.model.fit(self.X_train, self.Y_train,
                                    validation_data = (self.X_val, self.Y_val),
                                    batch_size = batch, epochs = number_epochs,
                                    verbose = 2, shuffle = True,
                                    callbacks = [self.checkpointer])
        pd.DataFrame(self.hist.history).to_csv(path_or_buf = history_filename)
        

#%% 
    def run_model(self, number_of_classes = 4, filters = 128, kernel_sizes = [10, 4, 1, 1],
                  pool_sizes = [2, 1, 1], drop_rate = 0.5, neurons = [256, 128, 64],
                  model_dir ='models/', model_filename = 'model.hdf5', 
                  history_filename = 'history.csv', batch = 32, number_epochs = 50,
                  pred_prefix = 'preds_', confusion_prefix = 'result_conf', 
                  targetClass = ['HVHA', 'HVLA', 'LVLA', 'LVHA'],
                  emotionLabel = {0: 'HVHA',1: 'HVLA', 2:'LVLA', 3: 'LVHA'},
                  emotionMinAverage = 0.1, modelType = 0):
        """
        It defines the parameters for emotion recognition with deep learning algorithms.
        Parameters 
        ----------
        number_of_classes : int
            DESCRIPTION. Number of labels classes
        filters : int
            DESCRIPTION.  The dimensionality of the output space in the convolution
        kernel_sizes : list
            DESCRIPTION. Specify the length of the 1D CNN window
        pool_size : list
            DESCRIPTION. Size of the max-pooling windows for the operation of the CNN layers
        drop_rate : int
            DESCRIPTION. Define input unit less than 1
        neurons : int
            DESCRIPTION. Specify the dimensionality of the array output space for a FCN
        model_dir : String
            DESCRIPTION. Path of model subdirectory.
        model_filename : String
            DESCRIPTION. Model file name
        history_filename : String
            DESCRIPTION. Accuracy results from all epochs of training and testing of the model
        batch : int
            DESCRIPTION. Batch size
        number_epochs : int
            DESCRIPTION. Define the iterations number on the dataset during the training time.
        pred_prefix : String
            DESCRIPTION. The default is 'preds_'.
        confusion_prefix : String
            DESCRIPTION. The default is 'result_conf'.
        targetClass : list
            DESCRIPTION. Label class
        emotionLabel : dictionary
            DESCRIPTION. Label class
        emotionMinAverage : float
            DESCRIPTION. Minimum average prediction for emotion classes
        modelType : int
            DESCRIPTION. DNN model
        Returns
        -------
        None.

        """         
        self.create_directory(model_dir)
        
        self.create_model(number_of_classes, filters, kernel_sizes, 
                          pool_sizes, drop_rate, neurons,
                          os.path.join(model_dir, model_filename),
                          os.path.join(model_dir, history_filename), 
                          batch, number_epochs, modelType)

        self.prediction(os.path.join(model_dir, pred_prefix), 
                        os.path.join(model_dir, confusion_prefix))
            
        self.plotMetrics(targetClass, model_dir, modelType)
        self.predictionAverageCheck(emotionMinAverage, emotionLabel, 
                                    targetClass, history_filename, model_dir)

#%% 
    def prediction(self, pred_prefix = 'preds_',  confusion_prefix = 'result_conf'):
        """
        Verify the accuracy of the deep learning model for emotion recognition.
        Parameters
        ----------
        pred_prefix : String
            DESCRIPTION. The default is 'preds_'. Filename prefix of the prediction results
        confusion_prefix : String
            DESCRIPTION. The default is 'result_conf'. Confusion matrix data  
        Returns
        -------
        None.
        """
        self.predictions = self.model.predict(self.X_val)
        self.score = accuracy_score(self.convert_vector(self.Y_val), 
                                    self.convert_vector(self.predictions))
        print('Last epoch\'s validation score is ', self.score)
        
        f1 = f1_score(self.convert_vector(self.Y_val),
                      self.convert_vector(self.predictions), pos_label = 1, average = 'micro')        
        print('The f1-score of classifier on test data is ', f1)
                   
        df = pd.DataFrame(self.convert_vector(self.predictions))
        filename = pred_prefix + str(format(self.score, '.4f')) + '.csv'
        df.to_csv(path_or_buf = filename, index = None, header = None)

        filename = confusion_prefix + str(format(self.score, '.4f')) 
        filename += '.csv'
        self.matrix = confusion_matrix(
            self.convert_vector(self.Y_val), self.convert_vector(self.predictions))
        pd.DataFrame(self.matrix).to_csv(
            path_or_buf = filename, index = None, header = None)


#%%         
    def predictionAverageCheck(self, emotionMinAverage, emotionLabel, 
                                targetClass, history_filename, model_dir):
        """
        Verify the predicted average for emotion recognition.
        Parameters
        ----------
        emotionMinAverage : float
            DESCRIPTION. Minimum average prediction for emotion classes
        emotionLabel : dict
            DESCRIPTION. The default is {0: 'feliz',1: 'divertido', 2:'alegre', 3: 'contento', 4:'satisfecho',...}. Emotion labels.
        targetClass : list
            DESCRIPTION. The default is ['feliz','divertido','alegre','contento','satisfecho',...]. Emotion labels.
        history_filename : String
            DESCRIPTION. Accuracy results from all epochs of training and testing of the model
        model_dir : String
            DESCRIPTION. Emotion recognition models subdirectory 
        Returns
        -------
        None.
        """        
        predHeader = {0:'Num', 1:'val_loss', 2:'val_accuracy', 3:'loss', 4:'accuracy'}
        self.predResult = pd.read_csv(model_dir + history_filename, engine='python', header = None)
        self.predResult.rename(index = str, columns = predHeader, inplace = True)
        self.predResult = self.predResult.drop(self.predResult.index[0])
        self.testAccMean = round(np.mean((self.predResult['val_accuracy'].values).astype(np.float)),2)
        self.trainAccMean = round(np.mean((self.predResult['accuracy'].values).astype(np.float)),2)
        
        temp=[]
        self.emotionPrediction = []
        for i in range(len(self.matrix)):  # 2D array
            total = 0
            for j in range(len(self.matrix[i])):
                total = (total + self.matrix[i][j]).astype(np.float)
                if i == j:
                    value = (self.matrix[i][j]).astype(np.float)
            pred = round(value / total, 2)
            self.emotionPrediction.append(pred)            
            
            for k, v in emotionLabel.items():
                if targetClass[i] == v and pred < emotionMinAverage:
                    temp.append(k)
                    break

        self.emotionLabelTemp={}
        for key, value in emotionLabel.items(): 
            find = False
            for emotion in range(len(temp)):
                if key == temp[emotion]:
                    find = True
                    break
            if not find:
                self.emotionLabelTemp[key] = value   
        

#%%         
    def plotMetrics(self, targetClass, model_dir, modelType):
        """
        Precision result plots of emotion recognition.
        Parameters
        ----------
        targetClass : list
            DESCRIPTION. The default is ['feliz','divertido','alegre','contento','satisfecho',...]. Emotion labels.
        model_dir : string
            DESCRIPTION. Emotion recognition models subdirectory 
        Returns
        -------
        None.
        """   
        if modelType == 0:
            name = '1D CNN Fatten'
        elif modelType == 1:
            name = '1D CNN LSTM'
        else: name = 'DCNN'             
        
        fig, ax = plt.subplots(figsize = (10,5))        
        plt.plot(self.hist.history['loss'], "--", color = '#ff7f0eff', label = "Train loss")
        plt.plot(self.hist.epoch, self.hist.history['accuracy'], "--", color = '#1f77b4ff', label = "Train")
        plt.plot(self.hist.history['val_loss'], "-", color = '#ff7f0eff', label = "Test loss")
        plt.plot(self.hist.epoch, self.hist.history['val_accuracy'], "-", color = '#1f77b4ff', label = "Test")
        ax.set_xlim(left=0, right = self.hist.epoch[-1])
        chartBox = ax.get_position()
        ax.set_position([chartBox.x0, chartBox.y0, chartBox.width*1, chartBox.height])
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.title("Heart rate dependent experiment on ES dataset")
        ax.legend(loc='upper center', bbox_to_anchor=(0.85, 0.95), shadow=True, ncol=1) 
        plt.ylabel('Loss/accuracy')
        plt.xlabel('Epoch')
        #plt.ylim(0) 
        ax.grid(ls = 'solid')
        plt.savefig(model_dir + 'train_val_'+ name + '_emotion_' + str(len(targetClass)) + '.svg', format='svg')
        plt.close() 

        print (self.model.summary())
        print (targetClass)
        plot_model(self.model, to_file = model_dir +'model_plot.png', show_shapes=True, show_layer_names=True)