#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  3 10:48:02 2018
@author: Frank Kane

"""

from surprise.model_selection import train_test_split
from surprise import KNNBaseline

class EvaluationData:
    """
    Split the hotels' Booking dataset in training/testing set and build a
    KNN Evaluator from the AlgoBase Class of the Surprise Library.
    Use a Leave-One-Out cross validation iterator for providing train/test
    indices to split data in train test sets.
    """
    
    def __init__(self, data, popularityRankings):
        """
        Default constructor
        """

        self.rankings = popularityRankings
        
        # Build a full training set for evaluating overall properties
        self.fullTrainSet = data.build_full_trainset()  
        self.fullAntiTestSet = self.fullTrainSet.build_anti_testset() 
        
        # Build a 80/20 train/test split for measuring accuracy 
        self.trainSet, self.testSet = train_test_split(data, test_size=.20, random_state=1)
                
        #Compute similarity matrix between items ***** for measuring diversity ****
        # Define the parameters for the content-based recommender. The SVD, SVD++, and
        # Random algorithms create a KNN evaluator of the AlgoBase class from the
        # Surprise Library. 
        sim_options = {'name': 'cosine', 'user_based': False}
        self.simsAlgo = KNNBaseline(sim_options=sim_options) 
        self.simsAlgo.fit(self.fullTrainSet)
 
#%%        
            
    def GetFullTrainSet(self):
        print("EvaluationData.GetFullTrainSet ")
        return self.fullTrainSet

#%%    
    def GetFullAntiTestSet(self):
        print("EvaluationData.GetFullTrainSet ")
        return self.fullAntiTestSet

#%%    
    def GetAntiTestSetForUser(self, testSubject):  # Evaluator class --> TopNRecs()
        """
        Which gives back a list of empty prediction values for every hotel
        a given user hasn’t rated already.
        Parameters
        ----------
        testSubject : string
            User identifier.
        Returns
        -------
        anti_testset : List
            list of empty prediction values for every hotel a given user hasn’t rated already
        """
        print("EvaluationData.GetAntiTestSetForUser: Compute anti_test_set... ")
        # It is just extracting a list of the items this user has rated already from the full training set
        trainset = self.fullTrainSet
        fill = trainset.global_mean
        anti_testset = []
        u = trainset.to_inner_uid(str(testSubject)) 
        print('testSubject ' + testSubject + 'id: ' + str(u))
        # We construct a bunch of prediction structures, each consisting of an inner user ID, 
        # an inner item ID, and a placeholder (marcador de posición) rating which for now is
        # just the global mean of all ratings.
        user_items = set([j for (j, _) in trainset.ur[u]])
        anti_testset += [(trainset.to_raw_uid(u), trainset.to_raw_iid(i), fill) for
                                 i in trainset.all_items() if
                                 i not in user_items]
        return anti_testset

#%%
    def GetTrainSet(self):
        print("EvaluationData.GetTrainSet ")
        return self.trainSet

#%%    
    def GetTestSet(self):
        print("EvaluationData.GetTestSet ")
        return self.testSet

#%%    
    def GetSimilarities(self):
        print("EvaluationData.GetSimilarities ")
        return self.simsAlgo
#%%    
    def GetPopularityRankings(self):
        print("EvaluationData.GetPopularityRankings ")
        return self.rankings