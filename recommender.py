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
        train_grouped.rename(columns = {'playlist_id': 'score'}, inplace=True)

        #sort the tracks based on the computed score
        train_sort = train_grouped.sort_values(['score', self.track], ascending = [0,1])

        train_sort['Rank'] = train_sort['score'].rank(ascending=0, method='first')

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

class content_based_rec():
    def __init__(self):
        self.train_data = None
        self.playlist_id = None
        self.track_id = None
        self.cooccurence_matrix = None
        self.item_similarity_recommendations = None

    # Get unique tracks corresponding to a given playlist
    def get_playlist_songs(self, playlist):
        playlist_data = self.train_data[self.train_data[self.playlist_id] == playlist]
        playlist_tracks = list(playlist_data[self.track_id].unique())

        return playlist_tracks

    # Get unique playlists for a given track
    def get_song_playlists(self, track):
        track_data = self.train_data[self.train_data[self.track_id] == track]
        track_playlists = set(track_data[self.playlist_id].unique())

        return track_playlists

    # Get unique items (songs) in the training data
    def get_all_tracks(self):
        all_tracks = list(self.train_data[self.track_id].unique())

        return all_tracks

    # Define cooccurence matrix
    def construct_cooccurence_martix(self, playlist_tracks, all_tracks):

        # Get playlists for all tracks in playlist_tracks
        playlist_tracks_playlists = []
        for i in range(0, len(playlist_tracks)):
            playlist_tracks_playlists.append(self.get_song_playlists(playlist_tracks[i]))

        #initialize the item cooccurence matrix of size len(playlist_tracks) x len(tracks)
        cooccurence_matrix = np.matrix(np.zeros(shape=(len(playlist_tracks), len(all_tracks))), float)

        #calculate similarity between playlist tracks and all unique tracks
        for i in range(0, len(all_tracks)):
            #calculate unique listeners (playlists) of track i
            track_i_data = self.train_data[self.train_data[self.track_id] == all_tracks[i]]
            playlists_i = set(track_i_data[self.playlist_id].unique())

            for j in range(0, len(all_tracks)):
                #get unique listeners (playlists) of song j
                playlists_j = playlist_tracks_playlists[j]

                #compute intersection of listeners of song i and j
                playlists_intersection = playlists_i.intersection(playlists_j)

                #compute cooccurence matrix[i,j] as Jaccard index
                if len(playlists_intersection) != 0:
                    #compute union of listeners of songs i and j
                    playlist_union = playlists_i.union(playlists_j)
                    cooccurence_matrix[j,i] = float(len(playlists_intersection))/float(len(playlist_union))
                else:
                    cooccurence_matrix[j,i] = 0

        return cooccurence_matrix

    #use cooccurence matrix to make top recommendations
