#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Tue Dec 1 01:58:15 2020

@authors: 
    Luz Santamaría Granados (luzsantamariag@gmail.com)
    Juan Francisco Mendoza Moreno (jfmendozam@gmail.com)
    
    Based on: Banerjee, Siddhartha. Collaborative Filtering for Movie Recommendations. 2020. 
    https://keras.io/examples/structured_data/collaborative_filtering_movielens/
"""

from tensorflow.keras.layers import Embedding
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from tensorflow.keras.utils import plot_model
import pandas as pd
from tensorflow.keras import metrics

class RecommenderCF:
    """
    Tourist Experiences Recommender based on Collaborative Filtering
    """
    def __init__(self, hotelRating, hotel):
        """
        Default constructor
        Parameters
        ----------
        hotelRating : dataframe
            DESCRIPTION. Set of user ratings by tourist experiences in hotels. 
        hotel : dataframe
            DESCRIPTION. Set of hotels' TE 
        Returns
        -------
        None.

        """        
        self.hotelRating = hotelRating
        self.hotel = hotel
        self.embedding_factor = 50
        
#%% 
    def getDataRec(self):
        """
        Gets the training and testing data set.
        Returns
        -------
        num_users : int
            DESCRIPTION. Users number
        num_hotels : int
            DESCRIPTION. hotels number
        x : array
            DESCRIPTION. Hotel review attributes
        y : array
            DESCRIPTION. User rating

        """
        # It performs the preprocessing to encode users and hotels as integer indices.
        mode = pd.options.mode.chained_assignment
        pd.options.mode.chained_assignment = None
        user = self.hotelRating["userId"].unique().tolist()
        self.user2user_encoded = {x: i for i, x in enumerate(user)}
        self.user_encoded2user = {i: x for i, x in enumerate(user)}
        hotel = self.hotelRating["hotelID"].unique().tolist()
        self.hotel2hotel_encoded = {x: i for i, x in enumerate(hotel)}
        self.hotel_encoded2hotel = {i: x for i, x in enumerate(hotel)}
        self.hotelRating["user"] = self.hotelRating["userId"].map(self.user2user_encoded)
        self.hotelRating["hotel"] = self.hotelRating["hotelID"].map(self.hotel2hotel_encoded)
        pd.options.mode.chained_assignment = mode
        num_users = len(self.user_encoded2user)
        num_hotels = len(self.hotel_encoded2hotel)
        self.hotelRating = self.hotelRating.sample(frac=1, random_state=42)
        x = self.hotelRating[["user", "hotel"]].values
        y = self.hotelRating["y"].values
        return num_users, num_hotels, x, y
    
#%% 
    def split_data(self, x, y, split = 0.8):
        """
        It divides the features arrays and labels vector into training and test data.        
        Parameters
        ----------
        x : array
            DESCRIPTION. Hotel review attributes
        y : array
            DESCRIPTION. User rating
        split : float
            DESCRIPTION. The default is 0.8. Percentage of training data

        Returns
        -------
        x_train : array
            DESCRIPTION. Training feature set
        y_train : array
            DESCRIPTION. Training label set
        x_test : array
            DESCRIPTION. Testing feature set
        y_test : array
            DESCRIPTION. Testing label set
        """
        size = len(x)
        x_train = x[ : int(split * size)]
        y_train = y[ : int(split * size)]
        x_test = x[int(split * size) :]
        y_test = y[int(split * size) :]   
        return x_train, y_train, x_test, y_test
    
#%%
    def validation_model(self, metric, x_train, y_train, x_test, y_test, loss_val):
        """
        Model validation with training and testing set.
        Parameters
        ----------
        metric : object
            DESCRIPTION. RMSE and MAE metrics
        x_train : array
            DESCRIPTION. Training feature set
        y_train : array
            DESCRIPTION. Training label set
        x_test : TYPE
            DESCRIPTION. Testing feature set
        y_test : TYPE
            DESCRIPTION. Testing label set
        loss_val : TYPE
            DESCRIPTION.
        Returns
        -------
        history : list
            DESCRIPTION. Training and testing results 
        """
        self.model.compile(optimizer = Adam(lr=0.001), 
                            loss = loss_val, metrics = metric)      
        history = self.model.fit(
            x = x_train, y = y_train, batch_size=64, epochs=10, 
            verbose=1, validation_data = (x_test, y_test)
            )
        return history                      
    
#%% 
    def run_model(self, user_id, split, figurePath):
        """
        It defines the parameters for the recommender model and metrics results.
        Parameters
        ----------
        user_id : int
            DESCRIPTION. 
        split : int
            DESCRIPTION.
        figurePath : String
            DESCRIPTION. Path name
        Returns
        -------
        results : list
            DESCRIPTION. Metrics results 
        """
        results = [] 
        num_users, num_hotels, x, y = self.getDataRec()
        x_train, y_train, x_test, y_test = self.split_data(x, y, split)
        self.model = RecommenderNet(num_users, num_hotels, self.embedding_factor)
        plot_model(self.model, to_file= figurePath + "model.png")
        # Compile the model - RMSE
        self.history_RMSE = self.validation_model(
            metrics.RootMeanSquaredError(name='RMSE'), x_train, y_train,
            x_test, y_test, 'mse')      # tf.keras.losses.MeanAbsoluteError()
        hist = self.history_RMSE
        self.plotMetrics(figurePath, 'RMSE', hist,
                         title = 'Root Mean Squared Error of the CF-Net recommender')      
        # Compile the model - MAE
        self.model = RecommenderNet(num_users, num_hotels, self.embedding_factor)
        self.history_MAE = self.validation_model(
            metrics.MeanAbsoluteError(name = 'MAE'), x_train, y_train,
            x_test, y_test, 'mse') # tf.keras.losses.MeanSquaredError()
        hist = self.history_MAE
        self.plotMetrics(figurePath, 'MAE', hist,
                         title = 'Mean Absolute Error of the CF-Net recommender')
        result_metrics = {
            'RMSE': np.mean(self.history_RMSE.history['val_RMSE']),
            'MAE':np.mean(self.history_MAE.history['val_MAE'])}
        results.append([self.prediction(user_id), result_metrics]) 
        return results

#%% 
    def plotMetrics(self, figurePath, name, history, title):
        """
        Precision result plots of TERS.
        Parameters
        ----------
        figurePath : String
            DESCRIPTION. Path name
        name : String
            DESCRIPTION. Metric name
        history : list
            DESCRIPTION. Training and testing results 
        title : String
            DESCRIPTION. Plot description
        Returns
        -------
        None.
        """        
        fig, ax = plt.subplots(figsize = (10,5))    
        plt.plot(history.history['loss'], "--", color = '#ff7f0eff', label = "Train loss")
        plt.plot(history.epoch, history.history[name], "--", color = '#1f77b4ff', label = "Train")
        plt.plot(history.history['val_loss'], "-", color = '#ff7f0eff', label = "Test loss")
        plt.plot(history.epoch, history.history['val_'+ name], "-", color = '#1f77b4ff', label = "Test")        
        ax.set_xlim(left=0, right = history.epoch[-1])
        chartBox = ax.get_position()
        ax.set_position([chartBox.x0, chartBox.y0, chartBox.width*1, chartBox.height])
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.title("Training and Testing session's progress over iterations of the CF-Net recommender")
        ax.legend(loc='upper center', bbox_to_anchor=(0.85, 0.95), shadow=True, ncol=1) 
        plt.ylabel('Loss/'+ name)
        plt.xlabel('Epoch')
        ax.grid(ls = 'solid')
        plt.savefig(figurePath + 'train_val'+ name + '_CF.svg', format='svg')
        plt.show()        

#%% 
    def prediction(self, user_id):
        """
        Generates the top-N list of Tourist Experiences of the hotels for a
        candidate user.
        # Author: Banerjee, Siddhartha. Collaborative Filtering for Movie Recommendations. 2020. 
        # https://keras.io/examples/structured_data/collaborative_filtering_movielens/   
        Parameters
        ----------
        user_id : int
            DESCRIPTION. Candidate user
        Returns
        -------
        recommendationList : list
            DESCRIPTION. top-N list of Tourist Experiences of the hotels
        """
        hotels_visited = self.hotelRating[self.hotelRating.userId == user_id]
        hotels_not_visited = self.hotel[~self.hotel["hotelID"].isin(hotels_visited.hotelID.values)]["hotelID"]         
        hotels_not_visited = list(set(hotels_not_visited).intersection(set(self.hotel2hotel_encoded.keys()))        )
        hotels_not_visited = [[self.hotel2hotel_encoded.get(x)] for x in hotels_not_visited] 
        user_encoder = self.user2user_encoded.get(user_id)    
        user_hotel_array = np.hstack(([[user_encoder]] * len(hotels_not_visited), hotels_not_visited))
        ratings = self.model.predict(user_hotel_array).flatten()
        top_ratings_indices = ratings.argsort()[-10:][::-1]
        recommended_hotelTE = [
            self.hotel_encoded2hotel.get(hotels_not_visited[x][0]) for x in top_ratings_indices]
        return self.hotel[self.hotel["hotelID"].isin(recommended_hotelTE)]

#%%
class RecommenderNet(Model):
    # Author: Banerjee, Siddhartha. Collaborative Filtering for Movie Recommendations. 2020. 
    # https://keras.io/examples/structured_data/collaborative_filtering_movielens/    
    
    def __init__(self, num_users, num_hotels, embedding_factor, **kwargs):
        super(RecommenderNet, self).__init__(**kwargs)
        self.num_users = num_users
        self.num_hotels = num_hotels
        self.embedding_factor = embedding_factor
        self.user_embedding = Embedding(
            num_users, embedding_factor, embeddings_initializer="he_normal",
            embeddings_regularizer = l2(1e-6))
        self.user_bias = Embedding(num_users, 1)
        
        self.hotel_embedding = Embedding(
            num_hotels, embedding_factor, embeddings_initializer="he_normal",
            embeddings_regularizer= l2(1e-6))
        self.hotel_bias = Embedding(num_hotels, 1)

    def call(self, inputs):
        user_vector = self.user_embedding(inputs[:, 0])
        user_bias = self.user_bias(inputs[:, 0])
        hotel_vector = self.hotel_embedding(inputs[:, 1])
        hotel_bias = self.hotel_bias(inputs[:, 1])
        dot_user_hotel = tf.tensordot(user_vector, hotel_vector, 2)
        # Add all the components (including bias)
        x = dot_user_hotel + user_bias + hotel_bias
        # The sigmoid activation forces the rating to between 0 and 1
        return tf.nn.sigmoid(x)        