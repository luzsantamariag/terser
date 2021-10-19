#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 19:58:25 2020

@authors: 
    Luz Santamaría Granados (luzsantamariag@gmail.com)
    Juan Francisco Mendoza Moreno (jfmendozam@gmail.com)
"""

import pandas as pd
from pymongo import MongoClient             # MongoClient class of the PyMongo library
from geopy.geocoders import Nominatim       # conda install -c conda-forge geopy
import json
from SPARQLWrapper import SPARQLWrapper, JSON
# pip install sparqlwrapper

class RecommenderData:
    """ 
    It obtains the dataset of emotion recognition and location of users
    who participated in the preliminary phase of the experiment. Besides,
    it consolidates the datasets of the hotels' tourist experiences portfolio
    and the users' reviews obtained through the OntoTouTra ontology. 
    """    
    
    def __init__(self, mongo_client = 'mongodb://localhost:27017', mongo_db = 'ters', hotelTEPath = 'tourist/'):
        """
        Default constructor
        Parameters
        ----------
        hotelTEPath : string
            Dataset directory path
        mongo_client : string
            MongoDB Server
        mongo_db : string
            Database name
        """

        self.hotelPath = hotelTEPath + 'hotels.csv'     
        self.servicePath = hotelTEPath + 'services.csv'
        self.experiencePath = hotelTEPath + 'hotelExperiences.csv'
        self.reviewsPath = hotelTEPath + 'reviews.csv'  
        self.hotel = pd.read_csv(self.hotelPath, sep = ",")
        self.service = pd.read_csv(self.servicePath, sep = ",")        
        self.touristExp = pd.read_csv(self.experiencePath, sep = ",")  
        self.reviews = pd.read_csv(self.reviewsPath, sep = ",")         
        self.coordPath = hotelTEPath + 'coord.csv'

        self.mongo_client = mongo_client           # MongoDb localhost URL
        self.mongo_db = mongo_db                   # MongoDB database name
        self.sliced_collection = 'mebSliced'       # Emotional slicing with HR instances collection of the participants. 
        self.er_collection               = 'emotionRecognition' # It contains the emotion recognition results generated from the HR data
        self.sliced_location_collection = 'mebSlicedLocation' #  Emotional slicing and location collection.
        
        self.tags = {
            'activity': {'tag': 'Actividades'},
            'wellness': {'tag': 'Instalaciones de wellness'}, 
            'angerQuadrant': {'tag': 'enojo'},
            'sadQuadrant': {'tag': 'triste'},
            'calmQuadrant': {'tag': 'calma'},
            'happyQuadrant': {'tag': 'feliz'},
            }
        

#%% getTouristExperience method

    def getTouristExperience(self):
        """
        Create the tourism experience column in the Hotel Dataframe with the
        data obtained from the activities and wellness service categories.     
        Returns
        -------
        None.
        """
        self.service['activityExperience'] = self.experienceCategory(
            self.tags['activity']['tag']) 
        self.service['wellnessExperience'] = self.experienceCategory(
            self.tags['wellness']['tag'])
                
        self.hotel['touristExperience'] = self.service.apply(
        lambda x: list(set(x['activityExperience'] + x['wellnessExperience'])), axis = 1)
        
        return self
        
#%% experienceCategory method
        
    def experienceCategory(self, name):
        """
        Get the tourist experience of each hotel according to the classification
        of the activities and wellness services.
        Parameters
        ----------
        name : string
            Hotel service name.
        Returns
        -------
        tourExp: list
            Experience category
        """
        tourExp = []
        for j in range(len(self.service)):
            temp = []
            if str(self.service[name][j]) != 'nan': 
                item = self.service[name][j].replace('[','').replace(']','').\
                    replace(" '",'').replace("'",'').split(',')
                for i in item:
                    if len(self.touristExp[self.touristExp['services'] == i]\
                           ['experience'].tolist()) > 0:
                        temp.append(self.touristExp[self.touristExp['services'] == i]\
                                    ['experience'].tolist()[0])
                tourExp.append(list(set(temp)))
            else:
                tourExp.append([])    
        
        return tourExp
    
    
#%% loadReviewEmotion method

    def loadReviewEmotion(self, field):
        """
        Get the user emotion tag through the rating score of the visited hotel.         
        Parameters
        ----------
        field : TYPE
            DESCRIPTION.

        Returns
        -------
        result : object
            Hotel reviews Dataframe
        """
        df =  self.reviews if field == 'rating' else self.hotel
        result = df[field].apply(
            lambda x: self.tags['angerQuadrant']['tag'] if (x <= 3) else (
                self.tags['sadQuadrant']['tag'] if (x <= 5) else (
                    self.tags['calmQuadrant']['tag'] if (x <= 7) else 
                        self.tags['happyQuadrant']['tag']
                )
            )
        )
        return result
   

#%% getOntotoutra method  ***** PENDIENT MIGRATION CODE OF DATASET

    def getOntotoutra(self):
        """
        Get JSON documents of Booking hotels, reviews and services.
        Returns
        -------
        None.
        """

        # subject  -  predicate  -  object
        sparql = SPARQLWrapper("http://192.168.0.36:3030/ds/query")   # Dell 14R
        # sparql = SPARQLWrapper("http://192.168.0.17:3030/ds/query") # MSI
        stringQuery = """
        
        PREFIX ott: <http://tourdata.org/ontotoutra/ontotoutra.owl#>
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        
        SELECT ?hotelID ?hotelName ?hotelAddress ?cityName ?hotelDescription 
               ?hotelLat ?hotelLon ?hotelReviewCategoricalScore
               ?hotelReviewNumber ?hotelReviewScore ?hotelURL
        WHERE {
          ?hotel       rdf:type                        ott:Hotel;
                       ott:hotelID                     ?hotelID;
                       ott:hotelName                   ?hotelName;
                       ott:hotelAddress                ?hotelAddress;
                       ott:hotelDescription            ?hotelDescription;
                       ott:hotelLat                    ?hotelLat;
                       ott:hotelLon                    ?hotelLon;
                       ott:hotelReviewCategoricalScore ?hotelReviewCategoricalScore;
                       ott:hotelReviewNumber           ?hotelReviewNumber;
                       ott:hotelReviewScore            ?hotelReviewScore;
                       ott:hotelURL                    ?hotelURL;
                       ott:hasCityParent               ?city.
           ?city       rdf:type                        ott:City;
                       ott:cityID                      ?cityID;
                       ott:cityName                    ?cityName;
                       ott:hasStateParent              ?department.
           ?department rdf:type                        ott:State;
                       ott:stateName                   ?stateName.
          FILTER(?stateName = "Boyacá")
        }
        """
        sparql.setQuery(stringQuery)
        sparql.setMethod('POST')
        sparql.setReturnFormat(JSON)
        sparql.query()
        results = sparql.query().convert()
        
        hotels = []
        for result in results["results"]["bindings"]:
            hotel=[]
            for k, v in result.items():
                hotel.append(v['value'])
            hotels.append(hotel)
        
        hotels_df = pd.DataFrame(
            data = hotels,
            columns = list(results["results"]["bindings"][0].keys())
        )
        
        
        # return hotels, reviews, services
        
   
#%% hotelRatingFilter method

    def hotelRatingFilter(self):
        """
        Filter the DataFrame of hotels with score values greater than 5 and 
        with data from tourist experiences.
        Get the popularity ranks of tourist experiences in hotels.
        Returns
        -------
        None.
        """
        
        #self.getOntotoutra()
        self.getTouristExperience()
        self.hotel['emotion'] = self.loadReviewEmotion('reviewScore')
        # Filter the hotels Dataframe by score and tourist experience. (695 --- 510 --- 355)
        dftmp = self.hotel[self.hotel.reviewScore > 5]
        hotelFilter = dftmp[dftmp.astype(str).touristExperience != '[]']

        # Get the popularity ranks of tourist experiences in hotels.
        top = pd.DataFrame()
        top['ranking'] = hotelFilter['reviewNumber'].rank(method = 'first', ascending = False).astype(int)
        top['hotelID'] = hotelFilter['hotelID']
        ranking = dict(zip(top.hotelID, top.ranking))
        
        #***** Sort the hotels dataframe columns
        #hotelHead = list(hotelFilter.columns.values)

        hotelsColumn = ['hotelID', 'hotelName', 'touristExperience','hotelLon', 
                        'hotelLat','reviewScore','reviewNumber','emotion', 
                        'hotelCity','hotelAddress','hotelUrl','state',
                        'reviewCategoricalScore', 'hotelDescription']
        self.hotels = hotelFilter[hotelsColumn]         

        self.reviews['emotion'] = self.loadReviewEmotion('rating')
        # Sort the review dataframe columns 
        #reviewHead = list(self.reviews.columns.values)
        reviewColumn = ['userId','hotelID', 'rating','country','emotion','reviewDate',
                        'accommodation','accommodationDate','catRating','positiveReview',
                        'negativeReview']
        review = self.reviews[reviewColumn]         
        # Filter the hotels reviews with the hotels selected. (39114 --- 30626)
        self.rating = pd.merge(self.hotels.hotelID, review, on = 'hotelID', how ='inner')  # left or inner  
        
        # min and max ratings will be used to normalize the ratings later
        min_rating = min(self.rating["rating"])
        max_rating = max(self.rating["rating"])
        
        # Normalize the targets between 0 and 1. Makes it easy to train. - Test 2 - Y normalize
        self.rating['y'] = self.rating["rating"].apply(lambda x: (x - min_rating) / (max_rating - min_rating)).values        
        
        hotelTouristExp = self.hotels.filter(['hotelID', 'hotelName', 'touristExperience','hotelLon', 'hotelLat'], axis=1)
        hotelRating = self.rating.filter(['userId','hotelID', 'rating','country','y'], axis=1)
        
        return hotelTouristExp, hotelRating, ranking
           
#%% getConnectionMongoDB method
        
    def getConnectionMongoDB(self):
        """
        It gets connecting to MongoDB.
        """
        # Create a client instance of the MongoClient class
        self.client = MongoClient(self.mongo_client)        
        # Load database and collection instances
        self.dataBaseMongo = self.client[self.mongo_db]
        # Load myemotionband collection
        self.locationCollection = self.dataBaseMongo[self.sliced_collection]
        self.location = pd.DataFrame(list(self.locationCollection.find()))         
        self.emotionCollection = self.dataBaseMongo[self.er_collection]   
        self.emotion = pd.DataFrame(list(self.emotionCollection.find()))  
        
        return self

#%% getUserLocation method

    def getUserLocation(self):  
        """
        It gets the participants' location data into emotion slices collection.
        """
        #self.getCoordinate()
        self.location['coord'] = [', '.join(str(x) for x in y) for y in map(tuple, self.location[['latitude', 'longitude']].values)]        
        self.coord = pd.read_csv(self.coordPath, sep = ";")
        self.location['city'] = self.location.apply(lambda x: self.coord[self.coord['coord'] == x.coord]['city'].item(), axis = 1)
        self.location['state'] = self.location.apply(lambda x: self.coord[self.coord['coord'] == x.coord]['state'].item(), axis = 1)
        self.location['country'] = self.location.apply(lambda x: self.coord[self.coord['coord'] == x.coord]['country'].item(), axis = 1)
        self.location = self.location.drop(['_id'], axis =1)
        #self.location.to_csv('tourist/location.csv', index = False)
        #self.insertLocationCollection()
        
        return self

#%% insertLocationCollection method    
    
    def insertLocationCollection(self):
        """
        Create a collection with the participant's location.
        """
        # Load sliced_location_collection
        self.locCollection = self.dataBaseMongo[self.sliced_location_collection]
        data_json = json.loads(self.location.to_json(orient='records'))
        self.locCollection.insert_many(data_json) 
        self.client.close()
        
        return self

#%% getUserEmotionRecognition method

    def getUserEmotionRecognition(self):
        """
        Get the percentage of the emotion felt by each participant with the 
        emotion recognition data and the duration in minutes.
        Returns
        -------
        emotion: dataframe
            Participants' emotion dataframe.
        """
        self.getConnectionMongoDB()
        self.getUserLocation()
        df_emotion = pd.DataFrame(columns = self.emotion.columns)
        df_emotion['felt_emotion']= ""
        for i, user in enumerate(self.emotion["imei"].unique()): 
            data = self.emotion.loc[self.emotion['imei'] == user]       
            self.emotion['felt_emotion']=self.emotion.loc[data.index.values[0]:data.index.values[len(data)-1]].apply(
                lambda x: x['duration']/data["duration"].sum(), axis = 1)
            df_emotion = df_emotion.append(
                self.emotion.loc[data.index.values[0]:data.index.values[len(data)-1]])
        self.emotion.update(df_emotion)
        self.getUserCity()
        
        return self.emotion
            
#%% getUserCity method

    def getUserCity(self):
        """
        Get the participant's location city based on historical data from the emotion,
        heart rate, and location slices.        
        """            
        l1 = self.location.groupby(['imei','city','longitude','latitude','state','country'])['city'].agg(['count'])
        l2 = l1[l1.groupby(['imei'])['count'].transform(max) == l1['count']]
        l2.reset_index(level=['longitude','latitude','imei','city','state','country'], inplace=True)
        
        df_emotion = pd.DataFrame(columns = self.emotion.columns)
        df_emotion['city'] = ""
        df_emotion['state'] = "" 
        df_emotion['country'] = ""         
        df_emotion['longitude'] = ""
        df_emotion['latitude'] = ""      
        for i, user in enumerate(self.emotion["imei"].unique()):
            data = self.emotion.loc[self.emotion['imei']==user]
            data_city = l2.loc[l2['imei']==user]
            self.fieldCreate('city', data, data_city)
            self.fieldCreate('state', data, data_city)
            self.fieldCreate('country', data, data_city)            
            self.fieldCreate('longitude', data, data_city)
            self.fieldCreate('latitude', data, data_city)            
            df_emotion = df_emotion.append(self.emotion.loc[data.index.values[0]:data.index.values[len(data)-1]])
        self.emotion.update(df_emotion)
        
        return self

#%% fieldCreate method
    
    def fieldCreate(self, name, data, data_city):
        """
        Create the city, longitude and latitude field 
        Parameters
        ----------
        name : string
            Field name.
        data : dataframe
            Emotion data.
        data_city : dataframe
            Location data.
        """
        self.emotion[name] = self.emotion.loc[
        data.index.values[0]:data.index.values[len(data)-1]].apply(
        lambda x: data_city[name].to_string(index=False), axis= 1)
        
        return self

#%% getCoordinate method

    def getCoordinate(self):
        """
        Use the user's latitude and longitude parameters by sending them to the 
        OpenStreetMap server to get the city, state, and country.
        """
        # Option 1. Create the site column in the Dataframe of all the records 
        # of the location collection.
        self.location["site"] = self.location.apply(lambda x: self.city_state_country(str(x.latitude) + ', ' + str(x.longitude)), axis=1)
        # Option 2. Option 2. Due to connection problems to the OpenStreetMap server,
        # it must fragment the Dataframe. 
        self.coord = pd.DataFrame(self.location['coord'].unique(), columns = ['coord'])
        # Create the site column in the coordinate Dataframe of unique records of the location collection.
        self.coord["site"] = self.coord.apply(lambda x: self.city_state_country(x.coord), axis=1)
        # Create a fragmented 500-row Dataframe with the coordinates of the unique records in the location collection..
        # coord1 = coord.loc[:500]  # coord.loc[501:1000]... coord.loc[3001:3242]
        # coord1['site'] = coord1.apply(lambda x: city_state_country(x.coord), axis = 1)
        # coord1.to_csv('tourist/coord1.csv', index = False)
        
        return self

#%% city_state_country method  

    def city_state_country(self, coord):
        """
        Use Openstreetmap's free Geocoding service to get a participant's city,
        state, and country.
        Parameters
        ----------
        coord : string
            DESCRIPTION. Latitude and longitude of a participant
        Returns
        -------
        city : string
            DESCRIPTION. City
        state : string
            DESCRIPTION. State
        country : string
            DESCRIPTION. Country
        """
        geolocator = Nominatim(user_agent="myGeocoder", timeout=2)
        location = geolocator.reverse(coord, exactly_one=True)
        address = location.raw['address']
        city = address.get('city', '')
        state = address.get('state', '')
        country = address.get('country', '')

        return city, state, country

#%% getTouristExperienceType

    def getTouristExperienceType(self):
        """
        It gets the tourism experience category of the hotels' dataset.
        Returns
        -------
        tourismExperience : dict
           List of types of tourist experiences for each hotel.
           {'Entertainment':0, 'Heritage/Culture':1, 'Adventure':2, 'Fitness':3,
            'Ecological':4, 'Family':5, 'Romantic':6, 'Relaxation':7}
        """
        TEType = {'Adventure':0, 'Ecological':1, 'Entertainment':2, 'Family':3,
                   'Fitness':4, 'Heritage/Culture':5, 'Romantic':6, 'Relaxation':7}           
        df = pd.DataFrame()
        df['hotelID'] = self.hotels['hotelID']
        df['TEType'] = self.hotels.apply(lambda x: self.catToNumerical(TEType, x.touristExperience), axis=1)
        tourismExperience = dict(zip(df.hotelID, df.TEType))
        
        return tourismExperience

#%% catToNumerical
    
    def catToNumerical(self, TEType, te):
        """
        Get tourist experience type vectors for every hotel
        Parameters
        ----------
        TEType : dict
            List of types of tourist experiences for each hotel.
        te : list
            Tourist experience categorical list
        Returns
        -------
        result : list
            Tourist experience binary list
        """
        result = []
        for k, v in TEType.items():
            for i in range(len(te)):
                if te[i] == k:
                    temp = 1
                    break
                else:
                    temp = 0
            
            result.append(temp)
                
        return result

#%% getTELocation

    def getTELocation(self):
        """
        It gets the location of the tourist experiences of the hotels' dataset.
        Returns
        -------
        hotelTELocation : dict
           List of longitude and latitude coordinates for each hotel.
           {'hotelLong':0, 'hotelLat':1, 'hotelCity':2}
        """       
        df = pd.DataFrame()
        df['hotelID'] = self.hotels['hotelID']
        df['location'] = self.hotels.apply(lambda x: (str(x.hotelLat) + ', ' + str(x.hotelLon)), axis=1)
        
        return dict(zip(df.hotelID, df.location))

#%% getReviewsCategory
    def getReviewsCategory(self):
        """
        It gets the hotel reviews category label.
        Returns
        -------
        reviewsCategory : dict
           A category label for hotel reviews
        """       
        df = pd.DataFrame()
        df['hotelID'] = self.hotels['hotelID']
        df['reviewCategoricalScore'] = self.hotels['reviewCategoricalScore']
        
        return dict(zip(df.hotelID, df.reviewCategoricalScore))

#%% getHotelDescription
    def getHotelDescription(self):
        """
        It gets the hotel description.
        Returns
        -------
        description : dict
           Hotel description
        """       
        df = pd.DataFrame()
        df['hotelID'] = self.hotels['hotelID']
        df['hotelDescription'] = self.hotels['hotelDescription']
        
        return dict(zip(df.hotelID, df.hotelDescription))    
    
