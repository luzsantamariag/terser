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
    Split the hotels dataset in training/testing set and build a
    KNN Evaluator from the AlgoBase Class of the Surprise Library.
    """
    
    def __init__(self, data, popularityRankings):
        """
        Default constructor
        """
        self.rankings = popularityRankings
        self.fullTrainSet = data.build_full_trainset()  
        self.fullAntiTestSet = self.fullTrainSet.build_anti_testset() 
        self.trainSet, self.testSet = train_test_split(data, test_size=.20, random_state=1)
        sim_options = {'name': 'cosine', 'user_based': False}
        self.simsAlgo = KNNBaseline(sim_options=sim_options) 
        self.simsAlgo.fit(self.fullTrainSet)
 
#%%    
    def GetAntiTestSetForUser(self, testSubject): 
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
        trainset = self.fullTrainSet
        fill = trainset.global_mean
        anti_testset = []
        u = trainset.to_inner_uid(str(testSubject)) 
        print('testSubject ' + testSubject + 'id: ' + str(u))
        user_items = set([j for (j, _) in trainset.ur[u]])
        anti_testset += [(trainset.to_raw_uid(u), trainset.to_raw_iid(i), fill) for
                                 i in trainset.all_items() if
                                 i not in user_items]
        return anti_testset

#%%        
    def GetFullTrainSet(self):
        print("EvaluationData.GetFullTrainSet ")
        return self.fullTrainSet

#%%
    def GetTrainSet(self):
        print("EvaluationData.GetTrainSet ")
        return self.trainSet

#%%    
    def GetTestSet(self):
        print("EvaluationData.GetTestSet ")
        return self.testSet