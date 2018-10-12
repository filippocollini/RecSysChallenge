import numpy as np
import pandas as pd

#top-pop recommender system
class top_pop_rec():
    def __init__(self):
        self.train = None
        self.playlist = None
        self.track = None
        self.pop_recommendations = None

    #create the RecSys
    def create(self, train, playlist, track):
        self.train = train
        self.playlist = playlist
        self.track = track

        #take as recommendation score the number of times the song is present in the playlists
        train_grouped = train.groupby([self.track]).agg({self.playlist: 'count'}).reset_index()
        train_grouped.rename(columns = {'playlist': 'score'}, inplace=True)

        #sort the tracks based on the computed score
        train_sort = train_grouped.sort_values(['count', self.track], ascending = [0,1])

        train_sort['Rank'] = train_sort['count'].rank(ascending=0, method='first')

        #get top 10
        self.pop_recommendations = train_sort.head(10)

    def recommend(self, playlist):
        playlist_recommendations = self.pop_recommendations

        #add playlist id column for which the recommendations are been generated
        playlist_recommendations['playlist'] = playlist

        # Bring user_id column to the front
        cols = playlist_recommendations.columns.tolist()
        cols = cols[-1:] + cols[:-1]
        playlist_recommendations = playlist_recommendations[cols]

        return playlist_recommendations
