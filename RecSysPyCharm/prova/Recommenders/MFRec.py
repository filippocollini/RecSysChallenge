import numpy as np
import scipy as sc
import pandas as pd
from scipy import sparse
from pandas import DataFrame
from tqdm import tqdm
from sklearn import feature_extraction
from time import sleep
from scipy.sparse.linalg import svds
from Builder import Builder

class MFRec(object):

    def fit(self, URM, target_playlists, target_tracks, num_playlist_to_recommend, MfRec, is_test):
        self.URM = URM
        self.target_playlists = target_playlists
        self.target_tracks = target_tracks
        self.num_playlist_to_recommend = num_playlist_to_recommend
        self.MfRec = MfRec
        self.is_test = is_test

        self.MfRec.fit()

    def recommend(self):
        # Compute the indices of the non-target playlists
        b = Builder()
        nontarget_indices = b.get_nontarget_indices(self.target_tracks)

        # Initialize the dataframe
        dataframe_list = []

        # Apply tfidf on the traspose of URM

        print('Predicting...', flush=True)
        for i in tqdm(range(0, self.num_playlist_to_recommend)):
            # Iterate over indices of target playlists
            index = b.get_target_playlist_index(self.target_playlists[i])

            # Compute the indices of the known tracks
            known_indices = np.nonzero(self.URM[index].toarray().flatten())

            # Make top-10 prediction
            URM_row_flatten = self.MfRec.predict(index).toarray().flatten()
            top_10_indices = b.get_top_10_indices(URM_row_flatten, nontarget_indices, known_indices, [])
            top_10_tracks = b.get_top_10_tracks_from_indices(top_10_indices)
            top_10_tracks_string = ' '.join([str(i) for i in top_10_tracks])

            # Create dataset
            if self.is_test:
                dataframe_list.append([self.target_playlists[i], top_10_tracks])
            else:
                dataframe_list.append([self.target_playlists[i], top_10_tracks_string])

        dataframe = pd.DataFrame(dataframe_list, columns=['playlist_id', 'track_ids'])

        return dataframe