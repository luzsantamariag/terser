#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Tuesday Mar 31 22:01:24 2020

@authors: 
    Luz Santamaría Granados (luzsantamariag@gmail.com)
    Juan Francisco Mendoza Moreno (jfmendozam@gmail.com)
"""

from RecommenderData import RecommenderData
from surprise import AlgoBase
from surprise import PredictionImpossible
import math as mt
import numpy as np
from haversine import haversine
from difflib import SequenceMatcher
from scipy import spatial
# pip install fuzzywuzzy
# pip install python-Levenshtein
from fuzzywuzzy import fuzz
import heapq as hq


class ContentRecommenderAlgorithm(AlgoBase):
    """
    It is a class derived from the AlgoBase Class of the Surprise Library. It implements
    the nearest neighbors' approach that involves the similarity computation of the
    training set of the hotels' Booking. The similarity measure between hotels, user,
    and item are required to estimate the ratings of the content during the prediction.
    """

    def __init__(self, k = 40, sim_options = {'name': 'cosine', 'user_based': False}): 
        """
        Initializes the abstract class that defines the prediction algorithm behavior.
        Parameters
        ----------
        k : int
            The neighbors' actual number. The default is 40.
        sim_options : dict
            Similarity measure configuration. The default is 'cosine' and 'content_based'
        """

        AlgoBase.__init__(self)
        self.k = k
        self.estimations = []
        

#%%
    def fit(self, trainset):
        """
        SurpriseLib, during the algorithm training, enables the building of a
        2D matrix that correlates the content-based similarity score between
        two hotels. Similarity metrics use attributes of tourist experiences,
        geographic location, description, and category tags from hotel reviews.
        Parameters
        ----------
        trainset : object
            Hotels Booking training set
        Returns
        -------
        similarities: array
            Hotels' similarity score
        """
        # Train the KNN algorithm through the training set given to the fit method
        # of the AlgoBase class.
        AlgoBase.fit(self, trainset)  
        print ("Generating hotels-based similarity matrix")
        rd = RecommenderData()
        hotelTouristExp, hotelRating, ranking = rd.hotelRatingFilter()
        print ("Computing similarity of hotels' tourist experiences")
        hotelTECategories = rd.getTouristExperienceType()
        print ("Computing similarity of hotels' location geographic")
        hotelTElocation = rd.getTELocation()
        print ("Computing similarity of hotels' reviews category")
        hotelCategoricalScore = rd.getReviewsCategory()
        print ("Computing similarity of hotels' description") 
        hotelDescription = rd.getHotelDescription()
                    
        self.similarities = np.zeros((self.trainset.n_items, self.trainset.n_items))
                
        for priorRating in range(self.trainset.n_items):
            for nextRating in range(priorRating + 1, self.trainset.n_items):

                priorHotelID = int(self.trainset.to_raw_iid(priorRating))
                nextHotelID = int(self.trainset.to_raw_iid(nextRating))
                te = self.computeTouristExperienceSimilarity(
                    priorHotelID, nextHotelID, hotelTECategories)
                location = self.computeLocationSimilarity(
                    priorHotelID, nextHotelID, hotelTElocation)
                category = self.computeCategorySimilarity(
                    priorHotelID, nextHotelID, hotelCategoricalScore) 
                description = self.computeDescriptionSimilarity(
                    priorHotelID, nextHotelID, hotelDescription) 
                self.similarities[priorRating, nextRating] = te * location * category * description                  
                self.similarities[nextRating, priorRating] = self.similarities[priorRating, nextRating]
                
        return self
    
#%%
    def computeDescriptionSimilarity(self, hotel1, hotel2, description):
        """
       Use the 'fuzzy-wuzzy' algorithm to measure the similarity of the description between
       two hotels. This metric implements the Levenshtein distance. The fuzz.token function
       preprocesses lowercase strings and removes punctuation marks. Then it orders them 
       alphabetically and applies the fuzz.ratio function to obtain the similarity result.
        Parameters
        ----------
        hotel1 : int
            Hotel identifier
        hotel2 : int
            Hotel identifier
        description : string
            Hotels' description
        Returns
        -------
        result: float
            hotel description similarity
        """
        x = description[hotel1]
        y = description[hotel2]
        descriptionSimilarity = fuzz.token_sort_ratio(x, y) / 100
         
        return descriptionSimilarity
        
#%%
    def computeCategorySimilarity(self, hotel1, hotel2, category):
        """
        Use the 'SequenceMatcher' algorithm to measure the similarity of user reviews
        score categories between two booked hotels. Adds the sizes of all matching
        sequences returned by the get_matching_blocks function and calculate the
        matched elements ratio in both strings.
        Parameters
        ----------
        hotel1 : int
            Hotel identifier
        hotel2 : int
            Hotel identifier
        categories : string
            Hotels' review categorical scale
        Returns
        -------
        result: float
            Hotel categories similarity
        """
        x = category[hotel1]
        y = category[hotel2]
        scaleSimilarity = SequenceMatcher(None, x, y).ratio()
         
        return scaleSimilarity
    
#%%    
    def computeTouristExperienceSimilarity(self, hotel1, hotel2, experience):
        """
        The compute of tourist experiences similarity between two hotels 
        is based on the cosine similarity metric.
        Parameters
        ----------
        hotel1 : int
            Hotel identifier
        hotel2 : int
            Hotel identifier
        experience : array
            Hotel tourist experiences
        Returns
        -------
        result: float
            Tourist experiences similarity
        """
        te1 = experience[hotel1]
        te2 = experience[hotel2]
        result = 1 - spatial.distance.cosine(te1, te2)
        # sumxx, sumxy, sumyy = 0, 0, 0
        # for te in range(len(te1)):
        #     x = te1[te]
        #     y = te2[te]
        #     sumxx += x * x
        #     sumyy += y * y
        #     sumxy += x * y 
        # result = sumxy/mt.sqrt(sumxx*sumyy)
        
        return result

#%%
    def computeLocationSimilarity(self,  hotel1, hotel2, location):
        """
        Compute the haversine distance between the hotels' geographic locations
        and using an exponential decay function to give more weight to the 
        closest hotels.
        Parameters
        ----------
        hotel1 : integer
            Hotel identifier
        hotel2 : integer
            Hotel identifier
        location : string
           Latitude and longitude cordinates
        Returns
        -------
        float
           Haversine distance
        """  
        
        location1 = np.fromstring(location[hotel1], dtype=float, sep=', ')
        location2 = np.fromstring(location[hotel2], dtype=float, sep=', ')
        
        distance = haversine(location1, location2)
        distanceSimility = mt.exp(-distance / 1000)

        return distanceSimility   
        
#%%
    def estimate(self, user, item):
        """    
        Estimate method created on Thu May  3 10:48:02 2018
        @author: Frank Kane
        It is based on the KNNBasic class estimation method and is required by
        the AlgoBase class prediction method. It receives as parameters an
        internal user identifier, an internal hotel identifier, and returns the
        estimated rating for the hotel (r^ui).
        Parameters
        ----------
        user : int
            Internal user identifier
        item : int
            Internal item identifier
        Returns
        -------
        estimatedRating : float
            Estimated rating
        """
        
        # iid = self.trainset.to_inner_iid(item)
        # uid = self.trainset.to_inner_uid(user)
        print ('user: ' + str(user) + ' item: ' + str(item)) 
        
        if not (self.trainset.knows_user(user) and self.trainset.knows_item(item)):
            raise PredictionImpossible('User and/or item is unkown.')
        
        # Calculate the similarity scores between the hotel item and the ratings made by the user.
        self.neighbors = []

        for rating in self.trainset.ur[user]:   # ur_full = trainset.ur -- sim_options['user_based' == False]
            similarity = self.similarities[item, rating[0]]   # Hotels' similarity matrix
            self.neighbors.append((similarity, rating[1]))  
        
        # Get the hotels list with the highest similarity score.
        self.k_neighbors = hq.nlargest(self.k, self.neighbors, key=lambda t: t[0]) # neighbors list sort by categorySimilarity (catSim, rating)
        
        # Calculate the weighted average by user ratings
        similaritySum = ratingTotal = 0
        for (similarityScore, rating) in self.k_neighbors:
            if similarityScore > 0:
                similaritySum += similarityScore
                ratingTotal += similarityScore * rating
                
        if similaritySum == 0:
            raise PredictionImpossible('No neighbors')  # def default_prediction(self): (float): The mean of all ratings in the trainset.

        self.estimations.append([user, item, ratingTotal / similaritySum])
        return ratingTotal / similaritySum  
    