#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 1 13:20:13 2020

@authors: 
    Luz Santamaría Granados (luzsantamariag@gmail.com)
    Juan Francisco Mendoza Moreno (jfmendozam@gmail.com)
"""

import pandas as pd
import numpy as np
import ast
# -----    pip install gender-guesser
import gender_guesser.detector as gender
from scipy import spatial
# -----    pip install haversine
from haversine import haversine


class RecommenderContext:  
    """
    Obtain the participant profile of the emotion recognition experiment and filter
    the recommendations list according to the user's context.
    """
    
    def __init__(self, rd, participant = 11, path = 'participant/'):
        """
        Default constructor
        Parameters
        ----------
        rd : object
            RecommenderData class instance
        participant : int
            User identifier. The default is 11
        path: string
            Participant directory path
        Returns
        -------
        None.
        """
        self.participant = participant   
        self.profilePath = path + 'participantProfile.csv'     
        self.userProfile = pd.read_csv(self.profilePath, sep = ",")  
        self.emotionPath = path + 'emotionTE.csv'     
        self.userEmotion = pd.read_csv(self.emotionPath, sep = ",") 
        self.TEType = {'Adventure':0, 'Ecological':1, 'Entertainment':2, 'Family':3,
                   'Fitness':4, 'Heritage/Culture':5, 'Romantic':6, 'Relaxation':7}   
        self.userNumber = 25
        self.recommenderNumber = 10
        self.rd = rd                      
        
        self.tags = {
            'happyEmotion': {'tag': 'feliz'},
            'funnyEmotion': {'tag': 'divertido'},
            'CheerfulEmotion': {'tag': 'alegre'},
            'contentedEmotion': {'tag': 'contento'},
            'satisfiedEmotion': {'tag': 'satisfecho'},
            'calmedEmotion': {'tag': 'calmado'},
            'relaxedEmotion': {'tag': 'relajado'},
            'tiredEmotion': {'tag': 'cansado'},
            'boredEmotion': {'tag': 'aburrido'},
            'depressedEmotion': {'tag': 'deprimido'},
            'ashamedEmotion': {'tag': 'avergonzado'},
            'sadEmotion': {'tag': 'triste'},
            'stressedEmotion': {'tag': 'estresado'},
            'afraidEmotion': {'tag': 'asustado'},
            'angryEmotion': {'tag': 'enojado'},
            'panicEmotion': {'tag': 'En pánico'},            
            'neutralEmotion': {'tag': 'neutro'},   
            'angerQuadrant': {'tag': 'enojo'},
            'sadQuadrant': {'tag': 'triste'},
            'calmQuadrant': {'tag': 'calma'},
            'happyQuadrant': {'tag': 'feliz'},
            
            'relaxationTE': {'tag': 'Relaxation'},
            'ecologicalTE': {'tag': 'Ecological'},
            'fitnessTE': {'tag': 'Fitness'},
            'entertainmentTE': {'tag': 'Entertainment'},

            'romanticTE': {'tag': 'Romantic'},
            'familyTE': {'tag': 'Family'},   
            'adventureTE': {'tag': 'Adventure'},
            'hcE': {'tag': 'Heritage/Culture'}, 
            }
        

    def computeUserSimilarity(self):   
        """
        Calculate the similarity of the profiles of a user participating in the
        emotion recognition experiment with the users of the hotels Booking dataset. 
        ----------
        Returns
        -------
        candidate : dataframe
             Candidate users' dataframe
        """
        profile = self.getUserProfile()
        candidateUser = self.getCandidateUser(profile)
        candidateUser = candidateUser.sort_values(
            ['teSimilarity','meanRating'], ascending=False)
        
        return candidateUser, self.userProfile


    def getUserProfile(self):
        """
        Get a participant profile of the emotion recognition experiment. Return the
        user's tourist experiences preferences, and gender, and country data.
        ----------
        Returns
        -------
        profile : list
            Participant profile data
        """
       
        profile = []
        self.userProfile['TEType'] = self.userProfile.apply(
            lambda x: self.rd.catToNumerical(
                self.TEType, ast.literal_eval(x.touristExperience)), axis=1)
        self.userER= self.rd.getUserEmotionRecognition()

        if len(self.userER[self.userER.imei == self.participant]) != 0:
            gender = list(
                self.userProfile[self.userProfile.imei == self.participant]['participantGender'])[0]
            country = list(
                self.userER[self.userER.imei == self.participant]['country'])[0].lstrip()
            te = list(
                self.userProfile[self.userProfile.imei == self.participant]['TEType'])[0]
        profile.append([self.participant, gender, country, te])        
        
        return profile      
    

    def getCandidateUser(self, profile):
        """
        Get a candidate user of the hotels' Booking dataset based on a profile of 
        one emotion recognition experiment participant. The first filter uses the
        participant's gender and country, and their tourist experiences preferences.
        Subsequently, the hotels' Booking dataset is filtered by user ratings, the
        unique hotels visited number, and the hotel reviews average score.
        Parameters
        ----------
        profile : list
            Participant profile data.
        Returns
        -------
        candidate : string
             Candidate user name
        """
        detectorGender = gender.Detector()       
        reviews = self.rd.rating
        similarityUser = reviews.groupby('userId').filter(lambda x: x['userId'].count() > 1)
        similarityUser['gender'] = similarityUser.apply(
            lambda x: detectorGender.get_gender(x.userId), axis = 1) 
        similarityUser = similarityUser.filter(['userId', 'gender', 'country','hotelID', 
                                            'rating','emotion'], axis=1)
        similarityUser = similarityUser[similarityUser.gender == profile[0][1]]
        similarityUser = similarityUser[similarityUser.country == profile[0][2]]
        
        hotelTE = self.rd.getTouristExperienceType()
            
        userGroup = similarityUser.groupby(
            ['userId', 'gender','country']).agg(
                ['count','nunique']).sort_values(
                    [('hotelID','nunique')], ascending= False) 
        
        candidateUser = self.computeTESimilarity(hotelTE, similarityUser, userGroup, profile[0][3])
    
        return candidateUser
        
    
    def computeTESimilarity(self, hotelTE, sim, userGroup, te):
        """
        Calculate the cosine similarity metric between the tourist experiences 
        preferences from an emotion recognition experiment user and the user's
        reviews of the hotels Booking dataset. 

        Parameters
        ----------
        hotelTE : dictionary
            Hotels' tourist experiences
        sim : dataframe
            Candidate users' dataframe
        userGroup : dataframe
            Candidate users' filtered by the number of unique hotels
        te : list
            Participant user tourist experience
        Returns
        -------
        candidate : dataframe
             Candidate users' dataframe
        """
        candidateUser = []
        # Compute the cosine similarity metric between the tourist experiences
        for i in range(self.userNumber):
            groupSimilarity = {}
            user = sim[
                sim.userId == userGroup.index[i][0]].filter(['hotelID'])
            user = user.drop_duplicates()
            user['touristExperience'] = sim['hotelID'].map(hotelTE)
            user['teSimilarity'] = user.apply(
                lambda x: 1 - spatial.distance.cosine(x.touristExperience ,te), axis = 1)             
            groupSimilarity['userId'] = userGroup.index[i][0]
            averageRating = (sim[sim.userId == userGroup.index[i][0]]).groupby(['userId']).agg(['mean'])
            groupSimilarity['meanRating'] = averageRating.iloc[0][1]            
            groupSimilarity['teSimilarity'] = user.teSimilarity.mean()
            groupSimilarity['uniqueHotel'] = len(user)
            candidateUser.append(groupSimilarity)
            
        return pd.DataFrame(candidateUser)

    
    def getUserContext(self, recommenderList):  
        """
        Get the hotels' tourist experiences recommendations for the participant of
        the emotion recognition experiment. The filter of recommendation list 
        involves the user's felt emotions and geographic location.
        Parameters
        ----------
        recommenderList : dataframe
            Tourist experiences recommendations
        Returns
        -------
        recommendation: dataframe
            Participant's tourist experiences recommendations
        """
        # Get the participant's felt emotion and location coordinates.  
        self.userER= self.rd.getUserEmotionRecognition()        
        er = self.userER.loc[(self.userER.felt_emotion == (
            self.userER[self.userER.imei == self.participant].felt_emotion.max()))]
        emotion = er.iloc[0]['emotion']
        latitude = er.iloc[0]['latitude']
        longitude = er.iloc[0]['longitude']
        
        # Get the quadrant and tourist experiences related to participant's felt emotion.  
        self.userEmotion['te'] = self.userEmotion.apply(
            lambda x: self.rd.catToNumerical(
            self.TEType, ast.literal_eval(x.touristExperience)), axis=1)
        self.userEmotion['check'] = self.userEmotion.apply(
            lambda x: (emotion in x.emotions), axis = 1)
        self.userEmotion = self.userEmotion.loc[(self.userEmotion.check == True)]
        
        # Filter the recommendation list to 10 records and add the Tourist experiences
        # binary array.
        recommenderList = recommenderList.filter(
            ['hotelID','hotelName','touristExperience','hotelLon','hotelLat',
             'reviewScore','reviewNumber','emotion','hotelCity','hotelAddress',
             'hotelUrl'], axis=1).iloc[0:self.recommenderNumber]
        recommenderList['te'] = recommenderList['hotelID'].map(
            self.rd.getTouristExperienceType())
        
        recommendation = self.getUserRecommendation(recommenderList, latitude, longitude)
        
        return recommendation 


    def getUserRecommendation(self, recommendation, latitude, longitude):
        """
        Compute the cosine similarity metric between tourism experiences related to
        the participant's felt emotion and tourist experiences of the recommender list.
        Then calculate the haversine distance to determine the closeness of the user's
        geographic location with the location of the hotel recommended.       
        Parameters
        ----------
        recommendation : dataframe
            Tourist experiences recommendations
        latitude : float
            User Latitude.
        longitude : float
            User longitude
        Returns
        -------
        userRecommendation : dataframe
            Participant's tourist experiences recommendations    
        """
        # Compute the cosine similarity metric between the tourist experiences
        recommendation['teSimilarity'] = recommendation.apply(
            lambda x: 1 - spatial.distance.cosine(
                x.te ,self.userEmotion.iloc[0]['te']), axis = 1)  
    
        # Compute the haversine distance between the geographic locations
        recommendation['locationSimilarity'] = recommendation.apply(
            lambda x: haversine(
                np.fromstring(str(latitude +','+ longitude), dtype=float, sep=', '),
                np.fromstring(str(str(x.hotelLat) +','+ str(x.hotelLon)), dtype=float, sep=', ')
                ), axis = 1)
        # Normalize the haversine distance
        max_value = recommendation['locationSimilarity'].max()
        min_value = recommendation['locationSimilarity'].min()
        recommendation['locationNormalized']  = abs(
            (recommendation['locationSimilarity'] - max_value) / (min_value - max_value))
        # Compute the mean between the tourist experiences similarity and the distance of
        # Haversine of the geographic locations. 
        recommendation['similarity'] = recommendation[['teSimilarity', 'locationNormalized']].mean(axis = 1) 
        userRecommendation = recommendation.sort_values(['similarity'], ascending=False)
        
        return userRecommendation
