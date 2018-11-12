import numpy as np
import scipy as sc
import pandas as pd
from tqdm import tqdm
from Builder import Builder


"""
Recommender with SVD: Singular Value Decomposition technique applied to 
the item content matrix. 

    * k: number of latent factors
    * knn: k-nearest-neighbours to evaluate similarity

If is_test is true, return a dataframe ready to be evaluated with the Evaluator class,
otherwise return a dataframe in the submission format.
"""


class SVDRec(object):

    def fit(self, URM, target_playlists, target_tracks, num_playlist_to_recommend,
            ICM, k, knn, is_test):

        self.URM = URM
        self.target_playlists = target_playlists
        self.target_tracks = target_tracks
        self.num_playlist_to_recommend = num_playlist_to_recommend
        self.ICM = ICM
        self.is_test = is_test

        self.S_ICM_SVD = Builder().get_S_ICM_SVD(self.ICM, k, knn)

    def recommend(self):
        # Compute the indices of the non-target playlists
        b = Builder()
        nontarget_indices = b.get_nontarget_indices(self.target_tracks)

        # Initialize the dataframe
        dataframe_list = []

        print('Predicting...', flush=True)
        for i in tqdm(range(0, self.num_playlist_to_recommend)):
            # Iterate over indices of target playlists
            index = b.get_target_playlist_index(self.target_playlists[i])

            # Compute the indices of the known tracks
            known_indices = np.nonzero(self.URM[index].toarray().flatten())

            # Calculate a row of the new URM
            URM_row = self.URM[index, :] * self.S_ICM_SVD

            # Make prediction
            URM_row_flatten = URM_row.toarray().flatten()
            top_5_indices = b.get_top_10_indices(URM_row_flatten, nontarget_indices, known_indices, [])
            top_5_tracks = b.get_top_10_tracks_from_indices(top_5_indices)
            top_5_tracks_string = ' '.join([str(i) for i in top_5_tracks])

            # Create dataset
            if self.is_test:
                dataframe_list.append([self.target_playlists[i], top_5_tracks])
            else:
                dataframe_list.append([self.target_playlists[i], top_5_tracks_string])

        dataframe = pd.DataFrame(dataframe_list, columns=['playlist_id', 'track_ids'])

        return dataframe
