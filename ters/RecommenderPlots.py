#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Friday, May 1 15:32:13 2020

@authors: 
    Luz Santamaría Granados (luzsantamariag@gmail.com)
    Juan Francisco Mendoza Moreno (jfmendozam@gmail.com)
    
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class RecommenderPlots:
    
    def __init__(self, rd, results, similarity, figurePath = 'figure/'):
        """
        Default constructor
        Parameters
        ----------
        rd : object
            RecommenderData class instance
        results : dict
            Recommender metric results   
        similarity : array
            Similarity matrix            
        figurePath: string
            Figures directory path   
        Returns
        -------
        None.              
        """
        self.figurePath = figurePath  
        self.rd = rd
        self.results = results
        self.similarity = similarity

 
#%% ratingPlot method

    def ratingPlot(self):
        """
        Users rating plot
        """   
        reviews = self.rd.rating[self.rd.rating.rating >= 5.0 ]
        print('Users unique number: '+ str(reviews.userId.nunique()))
        print('Hotels unique number: '+ str(reviews.hotelID.nunique()))
        print('Rating unique number: '+ str(reviews.rating.nunique()))
                
        reviewData = reviews.groupby(by=['rating']).agg({'userId': 'count'}).reset_index()
        reviewData.columns = ['rating', 'number']
        
        fig, ax = plt.subplots(figsize=(14, 7))
        sns.set(style="whitegrid", color_codes=True)
        sns.barplot(x="rating", y="number", data=reviewData, color="#1f77b4ff") # 1f77b4ff
        
        total = reviewData['number'].sum()
        for p in ax.patches:
            percentage = '{:.1f}%'.format(100 * p.get_height()/total)
            x = p.get_x() + p.get_width()
            y = p.get_height()
            ax.annotate(percentage, (x, y),ha='right', size = 7, va='bottom')
        
        plt.title("Distribution of hotel booking ratings", fontsize=11)
        plt.xlabel('rating', fontsize=11)
        plt.ylabel('count', fontsize=11) 
        plt.xticks(fontsize = 8)
        plt.yticks(fontsize = 8)
        plt.savefig(self.figurePath + 'bar_ratings.svg', format='svg')           
        

#%% metricPlot method

    def metricPlot(self):
        """
        Comparison of all algorithms using RMSE and MAE metrics.
        """        
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
                
        fig, ax = plt.subplots(figsize = (10,5))
        ax.plot(algorithmName, rmse, label='RMSE', marker='*', linewidth=2) 
        ax.plot(algorithmName, mae, label='MAE', marker='*', linewidth=2)
        plt.title('Performance of recommendation algorithms', loc='center', fontsize=15)
        plt.xlabel('Algorithms', fontsize=15)
        plt.ylabel('metric result', fontsize=15)
        ax.legend()
        ax.grid(ls='dashed')
        plt.savefig(self.figurePath + 'metrics_MAE_RMSE.svg', format='svg') 
        
        
        fig, ax = plt.subplots(figsize = (10,5))
        ax.plot(algorithmName, mae, label='MAE', marker='*', linewidth=2)
        plt.title('Performance of recommendation algorithms', loc='center', fontsize=15)
        plt.xlabel('Algorithms', fontsize=15)
        plt.ylabel('MAE', fontsize=15)
        ax.grid(ls='dashed')
        plt.savefig(self.figurePath + 'metrics_MAE.svg', format='svg') 
        plt.show()
        
        fig, ax = plt.subplots(figsize = (10,5))
        ax.plot(algorithmName, rmse, label='RMSE', marker='*', linewidth=2, color = '#e4722e') 
        plt.title('Performance of recommendation algorithms', loc='center', fontsize=15)
        plt.xlabel('Algorithms', fontsize=15)
        plt.ylabel('RMSE', fontsize=15)
        ax.grid(ls='dashed')
        plt.savefig(self.figurePath + 'metrics_RMSE.svg', format='svg') 
        plt.show()        
        
 #%% Correlation method
        
    def plot_correlation(self): 
        """
        This method generates the dataset correlation heatmap.
        """
        
        data = pd.DataFrame(self.similarity)
        # Show heatmap of the balanced dataset variables correlation.
        correlation = data.corr().round(1)
        corr = correlation.iloc[0:20,0:20]
        
        ax = sns.heatmap(corr, 
                    xticklabels=corr.columns.values,
                    yticklabels=corr.columns.values, annot=True, fmt = ".1g",
                    cmap="Blues", rasterized=True)
        ax.set_title('Similarity matrix correlation', fontsize = 15)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        plt.setp(ax.get_yticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        bottom, top = ax.get_ylim()
        ax.set_ylim(bottom + 0.5, top - 0.5)         
