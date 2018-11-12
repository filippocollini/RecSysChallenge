import numpy as np
import pandas as pd
import scipy.sparse as sps

from Recommenders.TopPopRec import TopPopRecommender
from Recommenders.SlimBprRec import SLIM_BPR_Recommender
from Recommenders.BasicItemKNNRec import BasicItemKNNRecommender
from Recommenders.MfBprRec import MF_BPR_Cython

# LOAD DATA FROM FILE TO DATATFRAMES

project_dir = "/home/alle/GitHub/RecSysChallenge/"

train = pd.read_csv(project_dir+"all/train.csv")
train.head()

tracks = pd.read_csv(project_dir+"all/tracks.csv")
tracks.head()

target = pd.read_csv(project_dir+"all/target_playlists.csv")
target.head()

# CREATE ARRAYS

# data from train table (playlist_id,track_id)
playlist_array = np.asarray(train['playlist_id'])
track_array = np.asarray(train['track_id'])

# data from target playlists table (playlist_id)
target_playlist_list = np.asarray(target['playlist_id']).tolist()

# data from tracks table (track_id, album_id, artist_id, duration_sec)
album_array = np.asarray(tracks['album_id'])
artist_array = np.asarray(tracks['artist_id'])
duration_array = np.asarray(tracks['duration_sec'])
all_tracks = np.asarray(tracks['track_id'])

# number of different tracks, albums, artists
num_tracks = all_tracks.size
num_albums = len(set(album_array.tolist()))
num_artists = len(set(artist_array.tolist()))


# MASKS CREATION
train_test_split = 0.80
train_mask = np.random.choice([True, False], len(playlist_array), p=[train_test_split, 1 - train_test_split])

test_mask = np.logical_not(train_mask)

# TRACK/PLAYLIST MATRIX CREATION (URM)
data = np.ones(len(playlist_array), dtype=int)

URM_all = sps.coo_matrix((data, (playlist_array, track_array)))
URM_all = URM_all.tocsr()

URM_train = sps.coo_matrix((data[train_mask], (playlist_array[train_mask], track_array[train_mask])))
URM_train = URM_train.tocsr()
print(URM_train.shape)

URM_test = sps.coo_matrix((data[test_mask], (playlist_array[test_mask], track_array[test_mask])))
URM_test = URM_test.tocsr()


def MAP(recommended_items, relevant_items):
    is_relevant = np.in1d(recommended_items, relevant_items, assume_unique=True)

    # Cumulative sum: precision at 1, at 2, at 3 ...
    p_at_k = is_relevant * np.cumsum(is_relevant, dtype=np.float32) / (1 + np.arange(is_relevant.shape[0]))

    map_score = np.sum(p_at_k) / np.min([relevant_items.shape[0], is_relevant.shape[0]])

    return map_score


def evaluate_algorithm(URM_test, recommender_object, at=10):
    cumulative_MAP = 0.0

    num_eval = 0

    for i, playlist_id in enumerate(set(playlist_array.tolist())):

        if i % 500 == 0:
            print("Playlist %d of %d" % (i, len(set(playlist_array))))

        relevant_items = URM_test[playlist_id].indices

        if len(relevant_items) > 0:
            recommended_items = recommender_object.recommend(playlist_id, at=at)
            num_eval += 1
            cumulative_MAP += MAP(recommended_items, relevant_items)

    cumulative_MAP /= num_eval

    print("Recommender performance is: MAP = {:.4f}".format(cumulative_MAP))


def generate_submission(recommender):

    target_MAP = 0.0

    submission = pd.DataFrame(columns=["playlist_id", "track_ids"])

    for i, playlist_id in enumerate(sorted(target_playlist_list)):

        if i % 500 == 0:
            print("Target playlist %d of %d" % (i, len(set(target_playlist_list))))

        recommended_items = recommender.recommend(playlist_id, at=10)
        target_MAP += MAP(recommended_items, URM_all[playlist_id].indices)
        recommendation = ' '.join(map(str, recommended_items))
        row = pd.DataFrame([[playlist_id, recommendation]], columns=["playlist_id", "track_ids"])
        submission = submission.append(row)

    target_MAP /= len(set(target_playlist_list))

    print("Recommender performance on target playlists is: MAP = {:.4f}".format(target_MAP))

    submission.to_csv(project_dir + "all/sub.csv", index=False)


def use_top_pop(generate_sub=False):

    top_pop_recommender = TopPopRecommender()
    top_pop_recommender.fit(URM_train)

    if generate_sub:
        generate_submission(top_pop_recommender)
    else:
        evaluate_algorithm(URM_test, top_pop_recommender, at=10)


def use_knn(generate_sub=False, icm="album", k=50):

    if icm != "artist" and icm != "album" and icm != "duration" and icm != "art&alb":
        return

    rec_s = BasicItemKNNRecommender(URM=URM_train, shrinkage=10.0, k=k)

    ones = np.ones(num_tracks, dtype=int)

    if icm == "artist":
        ICM_all_artist = sps.coo_matrix((ones, (all_tracks, artist_array)))
        ICM_all_artist = ICM_all_artist.tocsr()
        rec_s.fit(X=ICM_all_artist)
    elif icm == "album":
        ICM_all_album = sps.coo_matrix((ones, (all_tracks, album_array)))
        ICM_all_album = ICM_all_album.tocsr()
        rec_s.fit(X=ICM_all_album)
    elif icm == "duration":
        ICM_all_duration = sps.coo_matrix((ones, (all_tracks, duration_array)))
        ICM_all_duration = ICM_all_duration.tocsr()
        rec_s.fit(X=ICM_all_duration)
    elif icm == "art&alb":
        ICM_all_artist = sps.coo_matrix((ones, (all_tracks, artist_array)))
        ICM_all_artist = ICM_all_artist.tocsr()
        ICM_all_album = sps.coo_matrix((ones, (all_tracks, album_array)))
        ICM_all_album = ICM_all_album.tocsr()
        rec_s.fit(X=ICM_all_artist, Y=ICM_all_album)

    if generate_sub:
        generate_submission(rec_s)
    else:
        evaluate_algorithm(URM_test, rec_s)


def use_slim_brp(generate_sub=False, epochs=2):
    recommender = SLIM_BPR_Recommender(URM_train, epochs=epochs)
    recommender.fit()
    if generate_sub:
        generate_submission(recommender)
    else:
        evaluate_algorithm(URM_test, recommender)


def use_mf_bpr(gen_sub=False):
    recommender = MF_BPR_Cython(URM_train, recompile_cython=False, positive_threshold=4)

    # logFile = open("Result_log.txt", "a")

    recommender.fit(epochs=100, validate_every_N_epochs=10, start_validation_after_N_epochs=100, URM_test=URM_test,
                    batch_size=1, sgd_mode='sgd', learning_rate=1e-4, user_reg=0.01)

    if gen_sub:
        generate_submission(recommender)
    else:
        evaluate_algorithm(URM_test, recommender, at=10)


#use_knn(k=10, icm="art&alb")
#use_slim_brp(epochs=10)
use_mf_bpr()
