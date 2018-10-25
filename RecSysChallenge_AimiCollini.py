import numpy as np
import pandas as pd
import scipy.sparse as sps
import scipy


# LOAD DATA FROM FILE TO DATATFRAMES

project_dir = "/home/alle/GitHub/RecSysChallenge/"

train = pd.read_csv(project_dir + "all/train.csv")
train.head()

tracks = pd.read_csv(project_dir + "all/tracks.csv")
tracks.head()

target = pd.read_csv(project_dir + "all/target_playlists.csv")
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

URM_train = sps.coo_matrix((data[train_mask], (playlist_array[train_mask], track_array[train_mask])))
URM_train = URM_train.tocsr()
print(URM_train.shape)

URM_test = sps.coo_matrix((data[test_mask], (playlist_array[test_mask], track_array[test_mask])))
URM_test = URM_test.tocsr()
print(URM_test.shape)


def MAP(recommended_items, relevant_items):
    is_relevant = np.in1d(recommended_items, relevant_items, assume_unique=True)

    # Cumulative sum: precision at 1, at 2, at 3 ...
    p_at_k = is_relevant * np.cumsum(is_relevant, dtype=np.float32) / (1 + np.arange(is_relevant.shape[0]))

    map_score = np.sum(p_at_k) / np.min([relevant_items.shape[0], is_relevant.shape[0]])

    return map_score


def evaluate_algorithm(URM_test, recommender_object, at=10):
    cumulative_MAP = 0.0

    num_eval = 0

    for playlist_id in set(playlist_array.tolist()):

        relevant_items = URM_test[playlist_id].indices

        if len(relevant_items) > 0:
            recommended_items = recommender_object.recommend(playlist_id, at=at)
            num_eval += 1
            cumulative_MAP += MAP(recommended_items, relevant_items)

    cumulative_MAP /= num_eval

    print("Recommender performance is: MAP = {:.4f}".format(cumulative_MAP))


def generate_submission(recommender):

    submission = pd.DataFrame(columns=["playlist_id", "track_ids"])

    for playlist_id in sorted(target_playlist_list):
        recommendation = ' '.join(map(str, recommender.recommend(playlist_id, at=10)))
        row = pd.DataFrame([[playlist_id, recommendation]], columns=["playlist_id", "track_ids"])
        submission = submission.append(row)

    submission.to_csv(project_dir + "all/sub.csv", index=False)


# TOP POP RECOMMENDER IMPLEMENTATION

class TopPopRecommender(object):

    def fit(self, URM_train):

        self.URM_train = URM_train

        itemPopularity = (URM_train > 0).sum(axis=0)
        itemPopularity = np.array(itemPopularity).squeeze()

        # We are not interested in sorting the popularity value,
        # but to order the items according to it
        self.popularItems = np.flip(np.argsort(itemPopularity), axis=0)

    def recommend(self, playlist_id, at=10, remove_seen=True):

        if remove_seen:
            unseen_items_mask = np.in1d(self.popularItems,
                                        self.URM_train[playlist_id].indices,
                                        assume_unique=True,
                                        invert=True)

            unseen_items = self.popularItems[unseen_items_mask]

            recommended_items = unseen_items[0:at]

        else:
            recommended_items = self.popularItems[0:at]

        return recommended_items


def use_top_pop(generate_sub=False):

    top_pop_recommender = TopPopRecommender()
    top_pop_recommender.fit(URM_train)

    if generate_sub:
        generate_submission(top_pop_recommender)
    else:
        evaluate_algorithm(URM_test, top_pop_recommender, at=10)


# FROM HERE ON THERE IS THE BASIC ITEM KNN IMPLEMENTATION

ones = np.ones(num_tracks, dtype=int)

ICM_all_artist = sps.coo_matrix((ones, (all_tracks, artist_array)))
ICM_all_artist = ICM_all_artist.tocsr()

ICM_all_album = sps.coo_matrix((ones, (all_tracks, album_array)))
ICM_all_album = ICM_all_album.tocsr()

ICM_all_duration = sps.coo_matrix((ones, (all_tracks, duration_array)))
ICM_all_duration = ICM_all_duration.tocsr()


class BasicItemKNNRecommender(object):
    """ ItemKNN recommender with cosine similarity and no shrinkage"""

    def __init__(self, URM, k=50, shrinkage=100.0, similarity='cosine'):
        self.dataset = URM
        self.k = k
        self.shrinkage = shrinkage
        self.similarity_name = similarity
        if similarity == 'cosine':
            self.distance = Cosine(shrinkage=self.shrinkage)
        elif similarity == 'pearson':
            self.distance = Pearson(shrinkage=self.shrinkage)
        elif similarity == 'adj-cosine':
            self.distance = AdjustedCosine(shrinkage=self.shrinkage)
        else:
            raise NotImplementedError('Distance {} not implemented'.format(similarity))

    def __str__(self):
        return "ItemKNN(similarity={},k={},shrinkage={})".format(
            self.similarity_name, self.k, self.shrinkage)

    def fit(self, X):
        item_weights = self.distance.compute(X)

        item_weights = check_matrix(item_weights, 'csr')  # nearly 10 times faster
        print("Converted to csr")

        # for each column, keep only the top-k scored items
        # THIS IS THE SLOW PART, FIND A BETTER SOLUTION
        values, rows, cols = [], [], []
        nitems = self.dataset.shape[1]
        for i in range(nitems):
            if i % 10000 == 0:
                print("Item %d of %d" % (i, nitems))

            this_item_weights = item_weights[i, :].toarray()[0]
            top_k_idx = np.argsort(this_item_weights)[-self.k:]

            values.extend(this_item_weights[top_k_idx])
            rows.extend(np.arange(nitems)[top_k_idx])
            cols.extend(np.ones(self.k) * i)
        self.W_sparse = sps.csc_matrix((values, (rows, cols)), shape=(nitems, nitems), dtype=np.float32)

    def recommend(self, playlist_id, at=None, exclude_seen=True):
        # compute the scores using the dot product
        user_profile = self.dataset[playlist_id]
        scores = user_profile.dot(self.W_sparse).toarray().ravel()

        # rank items
        ranking = scores.argsort()[::-1]
        if exclude_seen:
            ranking = self._filter_seen(playlist_id, ranking)

        return ranking[:at]

    def _filter_seen(self, playlist_id, ranking):
        user_profile = self.dataset[playlist_id]
        seen = user_profile.indices
        unseen_mask = np.in1d(ranking, seen, assume_unique=True, invert=True)
        return ranking[unseen_mask]


def check_matrix(X, format='csc', dtype=np.float32):
    if format == 'csc' and not isinstance(X, sps.csc_matrix):
        return X.tocsc().astype(dtype)
    elif format == 'csr' and not isinstance(X, sps.csr_matrix):
        return X.tocsr().astype(dtype)
    elif format == 'coo' and not isinstance(X, sps.coo_matrix):
        return X.tocoo().astype(dtype)
    elif format == 'dok' and not isinstance(X, sps.dok_matrix):
        return X.todok().astype(dtype)
    elif format == 'bsr' and not isinstance(X, sps.bsr_matrix):
        return X.tobsr().astype(dtype)
    elif format == 'dia' and not isinstance(X, sps.dia_matrix):
        return X.todia().astype(dtype)
    elif format == 'lil' and not isinstance(X, sps.lil_matrix):
        return X.tolil().astype(dtype)
    else:
        return X.astype(dtype)


class ISimilarity(object):
    """Abstract interface for the similarity metrics"""

    def __init__(self, shrinkage=10.0):
        self.shrinkage = shrinkage

    def compute(self, X):
        pass


class Cosine(ISimilarity):
    def compute(self, X):
        # convert to csc matrix for faster column-wise operations
        X = check_matrix(X, 'csc', dtype=np.float32)

        # 1) normalize the columns in X
        # compute the column-wise norm
        # NOTE: this is slightly inefficient. We must copy X to compute the column norms.
        # A faster solution is to  normalize the matrix inplace with a Cython function.
        Xsq = X.copy()
        Xsq.data **= 2
        norm = np.sqrt(Xsq.sum(axis=0))
        norm = np.asarray(norm).ravel()
        norm += 1e-6
        # compute the number of non-zeros in each column
        # NOTE: this works only if X is instance of sparse.csc_matrix
        col_nnz = np.diff(X.indptr)
        # then normalize the values in each column
        X.data /= np.repeat(norm, col_nnz)
        print("Normalized")

        # 2) compute the cosine similarity using the dot-product
        dist = X * X.T
        print("Computed")

        # zero out diagonal values
        dist = dist - sps.dia_matrix((dist.diagonal()[scipy.newaxis, :], [0]), shape=dist.shape)
        print("Removed diagonal")

        # and apply the shrinkage
        if self.shrinkage > 0:
            dist = self.apply_shrinkage(X, dist)
            print("Applied shrinkage")

        return dist

    def apply_shrinkage(self, X, dist):
        # create an "indicator" version of X (i.e. replace values in X with ones)
        X_ind = X.copy()
        X_ind.data = np.ones_like(X_ind.data)
        # compute the co-rated counts
        co_counts = X_ind * X_ind.T
        # remove the diagonal
        co_counts = co_counts - sps.dia_matrix((co_counts.diagonal()[scipy.newaxis, :], [0]), shape=co_counts.shape)
        # compute the shrinkage factor as co_counts_ij / (co_counts_ij + shrinkage)
        # then multiply dist with it
        co_counts_shrink = co_counts.copy()
        co_counts_shrink.data += self.shrinkage
        co_counts.data /= co_counts_shrink.data
        dist.data *= co_counts.data
        return dist


def use_knn(generate_sub=False, icm="album"):

    if icm != "artist" and icm != "album" and icm != "duration":
        return

    rec_s = BasicItemKNNRecommender(URM=URM_train, shrinkage=10.0, k=50)

    if icm == "artist":
        rec_s.fit(ICM_all_artist)
    elif icm == "album":
        rec_s.fit(ICM_all_album)
    elif icm == "duration":
        rec_s.fit(ICM_all_duration)

    if generate_sub:
        generate_submission(rec_s)
    else:
        evaluate_algorithm(URM_test, rec_s)


use_knn(generate_sub=True)
