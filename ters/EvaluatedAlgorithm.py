#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Thu May  3 10:45:33 2018
@author: Frank Kane
"""

from surprise import accuracy

class EvaluatedAlgorithm:
    """
    It defines a wrapper for calling the Recommender Metrics module.
    """    
    
    def __init__(self, algorithm, name):
        """
        Default constructor
        """
        self.algorithm = algorithm   
        self.name = name
        print('EvaluatedAlgorithm.init: Evaluated Algorithm ' + name)

#%%     
    def Evaluate(self, evaluationData, verbose=True):
        """
        It has an option for doing RMSE and MAE metrics. 
        It puts all of the evaluation results into a dictionary called “metrics”.
        The Evaluator prints out the results of each EvaluatedAlgorithm side by side.

        Parameters
        ----------
        evaluationData : dataset object
            Evaluation data.
        verbose : bolean
            The default is True.

        Returns
        -------
        metrics : dictionary
            the evaluation results.
        """
        #print('EvaluatedAlgorithm.Evaluate: recommender metrics' + str(doTopN))
        metrics = {}
        # Compute accuracy
        if (verbose):
            print("Evaluating accuracy...")
            
        # Train the algorithm through the training set given to the fit method
        # of the AlgoBase class.            
        self.algorithm.fit(evaluationData.GetTrainSet())
        
        # class AlgoBase() --> def test(self, testset, verbose=False)
        #           test() --> def predict(self, uid, iid, r_ui=None, clip=True, verbose=False)
        #        predict() --> est = self.estimate(iuid, iiid) 
        
        # Validate the algorithm through the testing set given to the AlgoBase class
        # test method. Then calculate the rating prediction for the items specified.
        predictions = self.algorithm.test(evaluationData.GetTestSet())
        
        metrics["RMSE"] = accuracy.rmse(predictions, verbose=True)
        print('Root Mean Square error: ' + str(metrics["RMSE"]))
        metrics["MAE"] = accuracy.mae(predictions, verbose=True)
        print('Mean Absolute Error metric: ' + str(metrics["MAE"]))

        
        if (verbose):
            print("Analysis complete.")
    
        return metrics

#%%    
    def GetName(self):
        return self.name

#%%    
    def GetAlgorithm(self):
        return self.algorithm
    
    