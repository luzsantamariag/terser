#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Thu May  3 10:45:33 2018
@author: Frank Kane
"""

from surprise import accuracy

class EvaluatedAlgorithm:
    """
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
        metrics = {}
        if (verbose):
            print("Evaluating accuracy...")
                      
        self.algorithm.fit(evaluationData.GetTrainSet())
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
    
    
