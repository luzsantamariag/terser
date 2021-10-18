#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 10:22:10 2020

@authors: 
    Luz Santamaría Granados (luzsantamariag@gmail.com)
    Juan Francisco Mendoza Moreno (jfmendozam@gmail.com)
    
"""

from RecommenderData import RecommenderData
from RecommenderContext import RecommenderContext
from surprise import NormalPredictor
from surprise import Dataset, Reader
from ContentRecommenderAlgorithm import ContentRecommenderAlgorithm
from RecommenderCF import RecommenderCF
from RecommenderCF_CNN import RecommenderCF_CNN
from RecommenderPlots import RecommenderPlots
from surprise import SVD, SVDpp
from Evaluator import Evaluator
import pandas as pd
import random
import numpy as np
from surprise.model_selection import cross_validate
import matplotlib.pyplot as plt 


class TouristExperienceRecommender:
    
    def __init__(self):
        """
        Default constructor
        """
        self.participant = 11             # Emotion recognition experiment participant
        self.figurePath = 'figure/'       # Figures directory path
        self.profilePath = 'participant/' # Participant directory path  
        self.hotelTEPath = 'tourist/'     # Directory path of hotels' Booking dataset  
        self.mongo_client = 'mongodb://localhost:27017' # MongoDb localhost URL
        self.mongo_db = 'meb'                           # MongoDB database name
        
        # Load up common dataset for the recommender algorithms
        self.rd, self.evaluationData, self.rankings = self.loadTERSData()
        self.buildEvaluationData()
        self.evaluateRecommenderAlgoritm()
        self.generateRecommenderReport()
        

#%% LoadTERSData method
  
    def loadTERSData(self):
        """
        Obtain the emotion recognition dataset and consolidate the hotels' tourist experiences
        portfolio and the users' reviews obtained through the OntoTouTra ontology.         
        Returns
        -------
        rd : RecommenderData
            Class instance
        reviewRating : object
            Hotels' tourist experiences ratings
        hotelTEranking : dict
            Hotels' popularity ranks
        """    
        rd = RecommenderData(self.mongo_client, self.mongo_db, self.hotelTEPath) 
        print ("Loading the users' emotion recognition and hotels' Booking datasets.")
    
        hotelTouristExp, hotelRating, hotelTEranking = rd.hotelRatingFilter()
        rd.getUserEmotionRecognition()
    
        reader = Reader(rating_scale = (1, 10))
        reviewRating = Dataset.load_from_df(hotelRating[['userId','hotelID','y']], reader)
        self.hotels = rd.hotels
        self.reviews = rd.reviews
        self.rating = rd.rating
            
        return rd, reviewRating, hotelTEranking 


#%% 2. buildEvaluationData (reviewRatingDataset, hotelTEranking)

    def buildEvaluationData(self):
        """
        Split the hotels' Booking dataset in training/testing set. 
        Build a KNN Evaluator from the AlgoBase Class of the Surprise Library 
        and define the parameters for the content-based recommender.
         
       (for measuring diversity).
        """
        self.evaluator = Evaluator(self.evaluationData, self.rankings)
        dataset = self.evaluator.dataset
        # Compute similarity matrix between all pairs items using alternating least squares.
        self.sim = dataset.simsAlgo.sim
    
        # Cleaning values NAN in similarity matrix of hotels
        if np.isnan(self.sim).any():
            self.evaluator.dataset.sim = np.where(np.isnan(self.sim ), 0, self.sim )
            self.sim = dataset.simsAlgo.sim        

#%% 3. evaluateRecommenderAlgoritm

    def evaluateRecommenderAlgoritm(self):
        """
        Instantiate a raw Surpriselib algorithm to convert it into an Evaluated
        Algorithm class and then add it to the list of algorithms to generate
        predictions and evaluate performance with recommender metrics.
        """
        algorithms = []
        np.random.seed(0)
        random.seed(0) 
        # Implement the nearest neighbors' approach
        contentAlgorithm = ContentRecommenderAlgorithm()
        self.evaluator.AddAlgorithm(contentAlgorithm, "CBF")
        
        # Just make random recommendations
        Random = NormalPredictor()
        self.evaluator.AddAlgorithm(Random, "Random")
        # trainset = dataset
        
        # Throw in an Singular Value Descomposition recommender
        SVDAlgorithm = SVD(random_state=10)
        self.evaluator.AddAlgorithm(SVDAlgorithm, "SVD")
        
        # Throw in an Singular Value Descomposition recommender
        # SVDppAlgorithm = SVDpp(random_state=10)
        # self.evaluator.AddAlgorithm(SVDppAlgorithm, "SVDpp")
        
        algorithms = self.evaluator.algorithms   # Algorithms evaluated list
        algorithms[0].GetName()
    
        self.evaluator.Evaluate()
        self.results = self.evaluator.results   # Recommender metrics result
        
        # Cross validation
                
        results_cv_rmse = []
        results_cv_mae = []
        names_alg = []
        for i in range(len(algorithms)):
            result_cv = cross_validate(
                algorithms[i].algorithm, self.evaluationData, measures=['RMSE', 'MAE'],
                cv=5, verbose=True)
            results_cv_rmse.append(list(result_cv['test_rmse']))     
            results_cv_mae.append(list(result_cv['test_mae']))
            names_alg.append(algorithms[i].GetName())
        
        # plot the results
        fig, ax = plt.subplots(figsize=(7, 4))
        plt.boxplot(results_cv_mae, labels=names_alg, showmeans=True)
        plt.ylabel('MAE', fontsize = 11)
        plt.xlabel('Algorithm', fontsize = 11)
        plt.title('Mean Absolute Error with varying Number of Fold')
        plt.savefig(self.figurePath + 'mae_surprise' + '.svg', format='svg')
        plt.show()
        
        fig, ax = plt.subplots(figsize=(7, 4))
        plt.boxplot(results_cv_rmse, labels=names_alg, showmeans=True)
        plt.ylabel('RMSE', fontsize = 11)
        plt.xlabel('Algorithm', fontsize = 11)
        plt.title('Root Mean Square Error with varying Number of Fold')
        plt.savefig(self.figurePath + 'rmse_surprise' + '.svg', format='svg')
        plt.show()        

#%%
    def cfPrediction(self, name, hotel, rc):
        
        hotelRating = self.rd.rating[['userId','hotelID','y']]
        userTest = self.candidateUser.iloc[0][0]
        split = 0.8
        
        if name == 'CF_CNN':
            cf = RecommenderCF_CNN(hotelRating, hotel)
            result = cf.run_model(userTest, split, self.figurePath)
            # filter the recommendations list according to the user's context
            result_rec = rc.getUserContext(result[0][0])  
            self.recommendation.append([result_rec, name, result[0][1]])
            self.results[name] = result[0][1]
        else:
            cf = RecommenderCF(hotelRating, hotel)
            result = cf.run_model(userTest, split, self.figurePath)
            # filter the recommendations list according to the user's context
            result_rec = rc.getUserContext(result[0][0])  
            self.recommendation.append([result_rec, name, result[0][1]])
            self.results[name] = result[0][1]
    

#%%
    def generateRecommenderReport(self):
        """
        Get a candidate user for generating the recommendation list based on an
        experiment participant profile and filter the recommendations list according
        to the user's context.    
        """
        self.recommendation = []
        hotels = self.rd.hotels
        # Get a candidate user for generating the recommendation list 
        rc = RecommenderContext(self.rd, self.participant, self.profilePath) 
        self.candidateUser, self.userProfile = rc.computeUserSimilarity()
        # Get the recommendation list for each algorithm. 
        self.recommenderPrediction = self.evaluator.TopNRecs(
            self.rd, self.candidateUser.iloc[0][0]) 
        #acc_knn=accuracy.rmse(self.recommenderPrediction[0], verbose=True)
        for algo in range(len(self.recommenderPrediction)):
            print(str(algo) + ' Algorithm: ' + str(self.evaluator.algorithms[algo].GetName()))
            predictionDF = pd.DataFrame(self.recommenderPrediction[algo])        
            # Order the recommendations list by the result of the estimation obtained.        
            predictionDF = predictionDF.sort_values(by =['est','r_ui'], ascending=False)
            predictionDF['result'] = predictionDF.apply(
                lambda x: x.details['was_impossible'], axis = 1)
            recommColumn = ['userId','hotelID','userRating','ratingEstimated','details','result']
            predictionDF.columns = recommColumn 
            # Unify the recommendation list with the hotels' Booking dataset.
            self.recommendationList = pd.merge(predictionDF, hotels, on = 'hotelID', how='left')  
            # filter the recommendations list according to the user's context
            result = rc.getUserContext(self.recommendationList)
            
            self.recommendation.append([
                result,
                self.evaluator.algorithms[algo].GetName(),
                self.results[self.evaluator.algorithms[algo].GetName()]
                ])
        
        self.userEmotion = rc.userEmotion
        self.userER = rc.userER
        
        # Collaborative Filtering Algorithms - Keras
        self.cfPrediction('CF_CNN', hotels, rc)    
        self.cfPrediction('CF_Net', hotels, rc) 
              
        # Check the results of the performance metrics and select the best algorithm.
        algorithmName = []
        mae = []
        rmse = []
        for k,v in self.results.items():
            algorithmName.append(k)
            for key, value in v.items():       
                if key == 'MAE':
                    mae.append(value)
                else:
                    rmse.append(value)

        res = mae.index(min(mae))
        print("The best recommender algorithm is " + str(algorithmName[res]))
        self.userRecommendation = self.recommendation[res][0]
        
        # Recommender plots 
        rp = RecommenderPlots(self.rd, self.results, self.sim, self.figurePath)
        #rp.plot_correlation()
        rp.ratingPlot()
        rp.metricPlot()

#%% Init
        
if __name__ == "__main__":
    ter = TouristExperienceRecommender()
    hotels = ter.hotels 
    reviews = ter.reviews
    rating = ter.rating
    metric = ter.evaluator.results 
    recommenderList = ter.recommendationList
    candidateUser = ter.candidateUser
    user = ter.candidateUser.iloc[0][0]
    recommendation = ter.recommendation
    userRecommendation = ter.userRecommendation
    userEmotion = ter.userEmotion
    userER = ter.userER
    userProfile = ter.userProfile
    results = ter.results
    