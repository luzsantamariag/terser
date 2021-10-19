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
        For each algorithm you want to compare – this creates an EvaluatedAlgorithm under the hood 
        within the Evaluator, and adds it to a list internally called “algorithms”     
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
        It evaluates each EvaluatedAlgorithm, and prints out the results from each one in
        a nice tabular form The “doTop N” parameter – this allows us to bypass the 
        computation of hit rank metrics
        Parameters
        ----------
        doTopN : boolean
            Report activate.
        Returns
        -------
        None.

        """
        for algorithm in self.algorithms:  # algorithms --> EvaluatedAlgorithm(algorithm, name)
            print("Evaluating ", algorithm.GetName(), "...")
            self.results[algorithm.GetName()] = algorithm.Evaluate(self.dataset)  # original
          
#%%
    def TopNRecs(self, rd, testSubject='user', k=10):
        """
        Sometimes it’s helpful to look at the actual recommendations being produced for
        a known users whose tastes you understand at a more intuitive level. It can be a
        helpful sanity check.
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
                
