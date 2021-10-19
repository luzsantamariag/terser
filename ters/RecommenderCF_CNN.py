#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Tue Dec 20 01:58:15 2020

@authors: 
    Luz Santamaría Granados (luzsantamariag@gmail.com)
    Juan Francisco Mendoza Moreno (jfmendozam@gmail.com)
"""

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv1D, MaxPooling1D
from tensorflow.keras.layers import Input, Embedding, Concatenate, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from tensorflow.keras.utils import plot_model
import pandas as pd
from sklearn.preprocessing import LabelEncoder 
from tensorflow.keras import metrics
import tensorflow as tf

class RecommenderCF_CNN:
    
    def __init__(self, hotelRating, hotel):

        self.hotelRating = hotelRating
        self.hotel = hotel
        self.embedding_factor = 50
        self.dropout_rate = 0.1
        

#%% shuffle method 
        
    def getDataRec(self):
        # Perform some preprocessing to encode users and hotels as integer indices.
        # Method for creating a new column with user number
        mode = pd.options.mode.chained_assignment
        pd.options.mode.chained_assignment = None
        userId = LabelEncoder()
        self.hotelRating['user_id'] = userId.fit_transform(self.hotelRating['userId'].values) # 0 to 5460
        self.user = pd.Series(self.hotelRating.user_id.values,index=self.hotelRating.userId).to_dict() 
               
        item_enc = LabelEncoder()
        self.hotelRating['hotel'] = item_enc.fit_transform(self.hotelRating['hotelID'].values) # 0 to 354
        self.hotelID_encoded = pd.Series(self.hotelRating.hotel.values,index=self.hotelRating.hotelID).to_dict() 
        self.hotel_encoded = pd.Series(self.hotelRating.hotelID.values,index=self.hotelRating.hotel).to_dict()
        pd.options.mode.chained_assignment = mode
        # Add a new column named 'user_id'
        num_users = len(self.user)
        num_hotels = len(self.hotel)
                
        self.hotelRating = self.hotelRating.sample(frac=1, random_state=42)
        x = self.hotelRating[["user_id", "hotel"]].values
        y = self.hotelRating["y"].values
    
        return num_users, num_hotels, x, y
    
#%% split_data method     
        
    def split_data(self, x, y, split = 0.8):
        """
        It divides the features arrays and labels vector into training and test data.
        Parameters
        ----------
        split : TYPE, optional
            DESCRIPTION. The default is 0.8.
        """
        size = len(x)
        x_train = x[ : int(split * size)]
        y_train = y[ : int(split * size)]
        x_test = x[int(split * size) :]
        y_test = y[int(split * size) :]          
             
        return x_train, y_train, x_test, y_test
    
#%% embeddingLayer method

    def embeddingLayer(self, num_items, num_factors, id_input):
        x = Embedding(num_items, num_factors, embeddings_initializer='he_normal',
                      embeddings_regularizer=l2(1e-6))(id_input)
        return x
 
#%% recommender method
    
    def recommender(self):
        # Each instance will consist of two inputs: a single user id, and a single movie id
        user = Input(shape=(1,), name='user_id')
        hotel = Input(shape=(1,), name='hotel')
    
        userEmbedded = self.embeddingLayer(len(self.hotelRating.user_id.unique()), self.embedding_factor, user)
        hotelEmbedded = self.embeddingLayer(len(self.hotelRating.hotel.unique()), self.embedding_factor, hotel)
        
        # Concatenate the embeddings (and remove the useless extra dimension)
        x = Concatenate()([userEmbedded, hotelEmbedded])
        x = Conv1D(filters = 128, kernel_size = 1) (x) 
        x = MaxPooling1D(pool_size = 1) (x)   
        x = Dropout(self.dropout_rate)(x)
        x = Dense(64, kernel_initializer = 'he_normal')(x)  
        x = Dense(1, activation = 'relu', name = 'prediction')(x)
        # Create the recommendation model
        model = Model(inputs = [user, hotel], outputs = x)
        return model   

#%%
    def validation_model(self, metric, x_train, y_train, x_test, y_test, loss_val):
        
        self.model.compile(optimizer = Adam(lr=0.001), 
                            loss = loss_val, metrics = metric)      
        
        x_train = pd.DataFrame(x_train, columns = ['user_id', 'hotel'])
        x_test = pd.DataFrame(x_test, columns = ['user_id', 'hotel'])
        y_train = pd.DataFrame(y_train, columns = ['y'])
        y_test = pd.DataFrame(y_test, columns = ['y'])
        
        history = self.model.fit(
            x = [x_train.user_id, x_train.hotel], y = y_train.y,
            batch_size=64, epochs=10, verbose=1, 
            validation_data=(
                [x_test.user_id, x_test.hotel],  y_test.y)
            )
        return history                      

#%% model method 

    def run_model(self, user_id, split, figurePath):

        results = []    
        num_users, num_hotels, x, y = self.getDataRec()
        x_train, y_train, x_test, y_test = self.split_data(x, y, split)
        # Model create  
        self.model = self.recommender()
        plot_model(self.model, to_file = figurePath + "model_TERS_CNN.png")
         
        # Compile the model - RMSE
        self.history_RMSE = self.validation_model(
            metrics.RootMeanSquaredError(name='RMSE'), x_train, y_train,
            x_test, y_test, 'mse')        # tf.keras.losses.MeanAbsoluteError()
        hist = self.history_RMSE
        self.plotMetrics(figurePath, 'RMSE', hist,
                         title = 'Root Mean Squared Error of the CNN-based recommender')      

        # Compile the model - MAE
        self.model = self.recommender()
        self.history_MAE = self.validation_model(
            metrics.MeanAbsoluteError(name = 'MAE'), x_train, y_train,
            x_test, y_test, 'mse')      # tf.keras.losses.MeanSquaredError()
        hist = self.history_MAE
        self.plotMetrics(figurePath, 'MAE', hist,
                         title = 'Mean Absolute Error of the CNN-based recommender')
        
        result_metrics = {
            'RMSE': np.mean(self.history_RMSE.history['val_RMSE']),
            'MAE':np.mean(self.history_MAE.history['val_MAE'])}
        results.append([self.prediction(user_id), result_metrics]) 

        return results


#%% plotMetrics method 
        
    def plotMetrics(self, figurePath, name, history, title):
        
        fig, ax = plt.subplots(figsize = (10,5))
        ax.plot(history.epoch, history.history[name], label='train')
        ax.plot(history.epoch, history.history['val_'+ name], label='test')
        ax.set_xlabel('Epoch')
        ax.set_ylabel(name)
        ax.set_xlim(left=0, right = history.epoch[-1])
        chartBox = ax.get_position()
        ax.set_position([chartBox.x0, chartBox.y0, chartBox.width*1, chartBox.height])
        ax.legend(loc='upper center', bbox_to_anchor=(0.85, 0.95), shadow=True, ncol=1)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(ls = 'solid')
        plt.title(title, loc='center', fontsize=15)
        plt.savefig(figurePath + name + '_CF_CNN.svg', format='svg')
        plt.show()
             
        fig, ax = plt.subplots(figsize = (10,5))
        ax.plot(history.history["loss"], label='train')
        ax.plot(history.history["val_loss"], label='test')
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.title("Model loss: " + name + ' of the CNN-based recommender', fontsize=15)
        plt.ylabel("Loss")
        plt.xlabel("Epoch")
        ax.legend(loc='upper center', bbox_to_anchor=(0.85, 0.95), shadow=True, ncol=1)        
        #plt.legend(["train", "test"], loc = "upper right")
        ax.grid(ls = 'solid')
        plt.savefig(figurePath + 'loss_'+ name + '_CF_CNN.svg', format='svg') 
        plt.show()   

        fig, ax = plt.subplots(figsize = (10,5))        
        plt.plot(history.history['loss'], "--", color = '#ff7f0eff', label = "Train loss")
        plt.plot(history.epoch, history.history[name], "--", color = '#1f77b4ff', label = "Train")
        plt.plot(history.history['val_loss'], "-", color = '#ff7f0eff', label = "Test loss")
        plt.plot(history.epoch, history.history['val_'+ name], "-", color = '#1f77b4ff', label = "Test")
        ax.set_xlim(left=0, right = history.epoch[-1])
        chartBox = ax.get_position()
        ax.set_position([chartBox.x0, chartBox.y0, chartBox.width*1, chartBox.height])
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.title("Training and Testing session's progress over iterations of the CF-CNN recommender")
        ax.legend(loc='upper center', bbox_to_anchor=(0.85, 0.95), shadow=True, ncol=1) 
        plt.ylabel('Loss/'+ name)
        plt.xlabel('Epoch')
        #plt.ylim(0) 
        ax.grid(ls = 'solid')
        plt.savefig(figurePath + 'train_val'+ name + '_CF_CNN.svg', format='svg')
        plt.show()        
        
#%% prediction method

    def prediction(self, user_id):
        # Let us get a user and see the top recommendations.

        uidTest = self.user[user_id] 
        hotels_visited = self.hotelRating[self.hotelRating.userId == user_id]
        hotels_not_visited = list(self.hotel[~self.hotel["hotelID"].isin(hotels_visited.hotelID.values)]["hotelID"])   
        
        hotels_not_visited = list(
            set(hotels_not_visited).intersection(set(self.hotelID_encoded.keys())) # list 270
        )
        hotels_not_visited = pd.DataFrame([[self.hotelID_encoded.get(x)] for x in hotels_not_visited], columns = ['hotel']) 
        hotels_not_visited['user_id'] = uidTest
        
        # Prediction over the hotels list that the candidate user has not yet visited.
        ratings = self.model.predict([hotels_not_visited.user_id, hotels_not_visited.hotel]).flatten()
        hotels_not_visited['ratingEstimated'] = ratings 
        hotels_not_visited = hotels_not_visited.sort_values(by='ratingEstimated', ascending=False)
        hotels_not_visited['hotelID'] = hotels_not_visited['hotel'].map(self.hotel_encoded)
        recommended_hotelTE = self.hotel[self.hotel["hotelID"].isin(hotels_not_visited.hotelID)]

        return pd.merge(hotels_not_visited, recommended_hotelTE).head(10)          
        
