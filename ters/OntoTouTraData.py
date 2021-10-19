#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Mon Feb 10 10:22:10 2020

@authors: 
    Juan Francisco Mendoza Moreno (jfmendozam@gmail.com)
    Luz Santamaría Granados (luzsantamariag@gmail.com)
"""

from SPARQLWrapper import SPARQLWrapper, JSON
import pandas as pd

class OntoTouTraData():
    """
    It gets the knowledge base of the OntoToutra ontology.
    """
    
    def __init__(self, endPoint = 'tourdata.org:3030/ds/query', stateName = 'Boyacá'):
        """
        Default constructor
        ----------
        """
        self._endPoint = endPoint
        self._stateName = stateName
        
#%%
    def getHotelData(self):
        """
        It gets the connection to the OntoTouTra' EndPoint server and retrieves
        the hotels' tourist experiences from a department in Colombia through a
        SPARQL query.
        Returns
        -------
        hotels_df : dataframe
            DESCRIPTION. hotels data

        """     
        sparql = SPARQLWrapper(self._endPoint)   # OntoTouTra End-Point        
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
          FILTER(?stateName = 'Boyacá')
        }
        """
        stringQuery = stringQuery.replace('Boyacá', self._stateName)
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
        
        return hotels_df

#%%
    def getHotelService(self):
        """
        It makes the connection to the OntoTouTra' EndPoint server and recovers
        the hotels' services from a department in Colombia through a SPARQL query.
        Returns
        -------
        hotels_df : dataframe
            DESCRIPTION. hotels data

        """     
        sparql = SPARQLWrapper(self._endPoint)   # OntoTouTra End-Point        
        stringQuery = """
        PREFIX ott: <http://tourdata.org/ontotoutra/ontotoutra.owl#>
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        SELECT ?hotelID ?activity ?wellness
        WHERE {
          ?hotel         rdf:type                        ott:Hotel;
                         ott:hotelID                     ?hotelID;
                         ott:hasCityParent               ?city.
          ?hotelService  rdf:type                        ott:hotelService;
                         ott:hotelID                     ?hotelID;
                         ott:activity                    ?activity;
                         ott:wellness                    ?wellness;                       
                         ott:hasHotelParent              ?hotel.
          ?city          rdf:type                        ott:City;
                         ott:cityID                      ?cityID;
                         ott:cityName                    ?cityName;
                         ott:hasStateParent              ?department.
          ?department    rdf:type                        ott:State;
                         ott:stateName                   ?stateName.
          FILTER(?stateName = 'Boyacá')        }
        }
        """
        stringQuery = stringQuery.replace('Boyacá', self._stateName)
        sparql.setQuery(stringQuery)
        sparql.setMethod('POST')
        sparql.setReturnFormat(JSON)
        sparql.query()
        results = sparql.query().convert()
        
        hotelService = []
        for result in results["results"]["bindings"]:
            service=[]
            for k, v in result.items():
                service.append(v['value'])
            hotelService.append(service)
        
        hotelService_df = pd.DataFrame(
            data = hotelService,
            columns = list(results["results"]["bindings"][0].keys())
        )
     
        return hotelService_df

#%%     
    def getHotelReview(self):
        """
        It establishes the connection to the OntoTouTra' EndPoint server and 
        regains the hotels' reviews from a department in Colombia through a 
        SPARQL query.
        Returns
        -------
        hotels_df : dataframe
            DESCRIPTION. hotels services data

        """     
        sparql = SPARQLWrapper(self._endPoint)   # OntoTouTra End-Point        
        stringQuery = """
        PREFIX ott: <http://tourdata.org/ontotoutra/ontotoutra.owl#>
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        SELECT ?hotelID ?userId ?rating ?country
        WHERE {
          ?hotel         rdf:type                        ott:Hotel;
                         ott:hotelID                     ?hotelID;
                         ott:hasCityParent               ?city.
          ?hotelReview   rdf:type                        ott:hotelReview;
                         ott:hotelID                     ?hotelID;
                         ott:userId                      ?userId;
                         ott:rating                      ?rating; 
                         ott:country                     ?country;                          
                         ott:hasHotelParent              ?hotel.
          ?city          rdf:type                        ott:City;
                         ott:cityID                      ?cityID;
                         ott:cityName                    ?cityName;
                         ott:hasStateParent              ?department.
          ?department    rdf:type                        ott:State;
                         ott:stateName                   ?stateName.
          FILTER(?stateName = 'Boyacá')        }
        """
        stringQuery = stringQuery.replace('Boyacá', self._stateName)
        sparql.setQuery(stringQuery)
        sparql.setMethod('POST')
        sparql.setReturnFormat(JSON)
        sparql.query()
        results = sparql.query().convert()
        
        hotelReview = []
        for result in results["results"]["bindings"]:
            review=[]
            for k, v in result.items():
                review.append(v['value'])
            hotelReview.append(review)
        
        hotelReview_df = pd.DataFrame(
            data = hotelReview,
            columns = list(results["results"]["bindings"][0].keys())
        )
        
        return hotelReview_df    