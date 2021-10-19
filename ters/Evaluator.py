#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  3 10:22:34 2018
@author: Frank Kane

"""

from EvaluationData import EvaluationData
from EvaluatedAlgorithm import EvaluatedAlgorithm


class Evaluator:
    
    """
    It loads the datasets of tourist experiences of the hotels, the reviews
    of the users of the experiences lived in the hotels, the emotions felt, and the
    geographical location. It splits the datasets into training and testing set 
    to evaluate the recommendation algorithms.
    """
    algorithms = []
    
    def __init__(self, dataset, rankings):
        """
        Default constructor
        """
        print ("Split the hotels' Booking dataset in train/test set.")
        ed = EvaluationData(dataset, rankings)  
        self.dataset = ed
        self.results = {}
                
#%% 
    def AddAlgorithm(self, algorithm, name):
        """
        This creates an EvaluatedAlgorithm    
        Parameters
        ----------
        algorithm : object
            Evaluated algorithm.
        name : string
            Algorithm name.
        Returns
        -------
        None.

        """
        print('Evaluator.AddAlgorithm: ' + str(name)) 
        alg = EvaluatedAlgorithm(algorithm, name)
        self.algorithms.append(alg)

#%%                
    def Evaluate(self):        
        """
        It evaluates each EvaluatedAlgorithm
        Parameters
        ----------
        doTopN : boolean
            Report activate.
        Returns
        -------
        None.

        """
        for algorithm in self.algorithms:  
            print("Evaluating ", algorithm.GetName(), "...")
            self.results[algorithm.GetName()] = algorithm.Evaluate(self.dataset) 
          
#%%
    def TopNRecs(self, rd, testSubject='user', k=10):
        """
        Generates the top-N list of Tourist Experiences of the hotels for a
        candidate user.
        Parameters
        ----------
        ml : class object
            DESCRIPTION.
        testSubject : string
            User name
        k : int, optional
            DESCRIPTION. The default is 10.
        Returns
        -------
        None.
        """
        print('Evaluator.SampleTopNRecs ... ')
        predictionRecommendation = []
        
        for algo in self.algorithms:
            print("\nUsing recommender ", algo.GetName())
            print("\nBuilding recommendation model...")
            trainSet = self.dataset.GetFullTrainSet()
            algo.GetAlgorithm().fit(trainSet)
            print("Computing recommendations...")
            testSet = self.dataset.GetAntiTestSetForUser(testSubject) 
            predictions = algo.GetAlgorithm().test(testSet)
            predictionRecommendation.append(predictions)
            recommendations = []
            print ("\nWe recommend:")
            for userID, hotelID, actualRating, estimatedRating, _ in predictions:
                hotel_ID = int(hotelID)
                recommendations.append((hotel_ID, estimatedRating))
            recommendations.sort(key=lambda x: x[1], reverse=True)
            hotels = rd.hotels 
            for ratings in recommendations[:10]:
                print(hotels[hotels.hotelID == ratings[0]]['hotelName'].to_string(index = False) +
                      ', ' + str(ratings[1]))   
        return predictionRecommendation    