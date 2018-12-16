from sklearn.preprocessing import MultiLabelBinarizer

from Builder import Builder
from Evaluator import Evaluator
from KNN.ItemKNNCBFRecommender import ItemKNNCBFRecommender
from Recommenders import ItemBasedRec, CollaborativeFilteringRec, ItemUserAvgRec,\
    SlimBPRRec, SVDRec, RoundRobinRec, HybridRec, TopPopRec, MFRec
import SlimBPR, MFBPR
import os as os
from pyspark.mllib.recommendation import ALS, Rating
from pyspark import SparkContext, SparkConf
from tqdm import tqdm
import pandas as pd
import time
from Base.Recommender_utils import check_matrix, similarityMatrixTopK


os.environ["PYSPARK_PYTHON"] = "/opt/anaconda/bin/python3.6"

"""
This file is a control panel to test or make predictions with provided recommenders.
All the recommenders are gathered in the "Recommenders" folder.

Each one needs to be trained at first, with the fit() function and then can make its
prediction with the recommend() function.

If is_test is true, the dataset will be split into training set (80%) and test set (20%)
and the MAP@5 will be computed on the test set.
Otherwise, if is false, a .csv file with the prediction will be produced.
"""

import numpy as np
import scipy.sparse as sps

def train_test_holdout(URM_all, train_perc = 0.8):


    numInteractions = URM_all.nnz
    URM_all = URM_all.tocoo()
    shape = URM_all.shape


    train_mask = np.random.choice([True,False], numInteractions, [train_perc, 1-train_perc])


    URM_train = sps.coo_matrix((URM_all.data[train_mask], (URM_all.row[train_mask], URM_all.col[train_mask])), shape=shape)
    URM_train = URM_train.tocsr()

    test_mask = np.logical_not(train_mask)

    URM_test = sps.coo_matrix((URM_all.data[test_mask], (URM_all.row[test_mask], URM_all.col[test_mask])), shape=shape)
    URM_test = URM_test.tocsr()

    return URM_train, URM_test


def top_pop_rec():
    print('*** Top Popular Recommender ***')

    rec = TopPopRec.TopPopRec()

    train_df = rec.recommend()
    train_df.to_csv('TopPopular.csv', sep=',', index=False)


def item_based(is_test):
    print('*** Item Based Recommender ***')

    b = Builder()
    ev = Evaluator()
    ev.split()
    rec = ItemBasedRec.ItemBasedRec()

    S_ICM = b.build_S_ICM_knn(b.build_ICM(), 250)

    rec.fit(ev.get_URM_train(), ev.get_target_playlists(), ev.get_target_tracks(), ev.num_playlists_to_test,
            S_ICM, is_test)
    train_df = rec.recommend()

    if is_test:
        map5 = ev.map5(train_df)
        print('ItemBased MAP@5:', map5)
    else:
        print('Prediction saved!')
        train_df.to_csv('ItemBased.csv', sep=',', index=False)


def collaborative_filtering(is_test):
    print('*** Test Collaborative Filtering Recommender ***')

    b = Builder()
    ev = Evaluator(is_test=is_test)
    ev.split()
    rec = CollaborativeFilteringRec.CollaborativeFilteringRec()

    S_UCM = b.get_S_UCM_KNN(b.get_UCM(ev.get_URM_train()), 500)

    rec.fit(ev.get_URM_train(), ev.get_target_playlists(), ev.get_target_tracks(), ev.num_playlists_to_test,
            S_UCM, is_test)
    train_df = rec.recommend()

    if is_test:
        map5 = ev.map5(train_df)
        print('CollaborativeFiltering MAP@5:', map5)
    else:
        print('Prediction saved!')
        train_df.to_csv(os.path.dirname(os.path.realpath(__file__))[:-19] + "/all/subCollab.csv", sep=',', index=False)


def item_user_avg(is_test):
    print('*** Test Item User Avg Recommender ***')

    b = Builder()
    ev = Evaluator()
    ev.split()
    rec = ItemUserAvgRec.ItemUserAvgRec()

    S_ICM = b.build_S_ICM_knn(b.build_ICM(), 250)
    S_UCM = b.get_S_UCM_KNN(b.get_UCM(b.get_URM()), 500)

    rec.fit(ev.get_URM_train(), ev.get_target_playlists(), ev.get_target_tracks(), ev.num_playlists_to_test,
            S_ICM, S_UCM, True, 0.80)

    train_df = rec.recommend()

    if is_test:
        map5 = ev.map5(train_df)
        print('ItemUserAvg MAP@5:', map5)
    else:
        print('Prediction saved!')
        train_df.to_csv('ItemUserAvg.csv', sep=',', index=False)


def slim_BPR(is_test):
    print('*** Test Slim BPR Recommender ***')

    ev = Evaluator()
    ev.split()
    rec = SlimBPRRec.SlimBPRRec()

    rec.fit(ev.get_URM_train(), ev.get_target_playlists(), ev.get_target_tracks(), ev.num_playlists_to_test, 0.1, 1,
            1.0, 1.0, 1000, 1, is_test)
    train_df = rec.recommend()

    if is_test:
        map5 = ev.map5(train_df)
        print('SlimBPR MAP@5:', map5)
    else:
        print('Prediction saved!')
        train_df.to_csv('SlimBPR.csv', sep=',', index=False)


def SVD(is_test):
    print('*** Test SVD Recommender ***')

    b = Builder()
    ev = Evaluator(is_test=is_test)
    ev.split()
    rec = SVDRec.SVDRec()

    rec.fit(ev.get_URM_train(), ev.get_target_playlists(), ev.get_target_tracks(), ev.num_playlists_to_test,
            b.build_ICM(), k=10, knn=250, is_test=is_test)
    train_df = rec.recommend()

    if is_test:
        map5 = ev.map5(train_df)
        print('SlimBPR MAP@5:', map5)
    else:
        print('Prediction saved!')
        train_df.to_csv('SlimBPR.csv', sep=',', index=False)


def round_robin_rec(is_test, avg_mode):
    print('*** Test Round Robin Recommender ***')

    b = Builder()
    ev = Evaluator(is_test=is_test)
    ev.split()
    rec = RoundRobinRec.RoundRobinRec()

    S_ICM = b.build_S_ICM_knn(b.build_ICM(), 250)
    S_UCM = b.get_S_UCM_KNN(b.get_UCM(ev.get_URM_train()), 500)
    Slim =  SlimBPR.SlimBPR(ev.get_URM_train()).get_S_SLIM_BPR(500)

    rec.fit(ev.get_URM_train(), ev.get_target_playlists(), ev.get_target_tracks(), ev.num_playlists_to_test,
            S_ICM, S_UCM, Slim, is_test, mode="jump", a=3, b=1, c=1)

    if avg_mode:
        train_df = rec.recommend_avg()
    else:
        train_df = rec.recommend_rr()

    if is_test:
        map5 = ev.map5(train_df)
        print('RoundRobin MAP@5:', map5)
    else:
        print('Prediction saved!')
        train_df.to_csv(os.path.dirname(os.path.realpath(__file__))[:-19] + "/all/sub.csv", sep=',', index=False)

def hybrid_repo(is_test):
    b = Builder()
    ev = Evaluator()
    ev.split()
    ICM = b.build_ICM()

    URM_train, URM_test = train_test_holdout(b.get_URM(), train_perc=0.8)
    URM_train, URM_validation = train_test_holdout(URM_train, train_perc=0.9)

    from ParameterTuning.AbstractClassSearch import EvaluatorWrapper
    from Base.Evaluation.Evaluator import SequentialEvaluator

    evaluator_validation = SequentialEvaluator(URM_validation, cutoff_list=[5])
    evaluator_test = SequentialEvaluator(URM_test, cutoff_list=[5, 10])

    evaluator_validation = EvaluatorWrapper(evaluator_validation)
    evaluator_test = EvaluatorWrapper(evaluator_test)

    from KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
    from ParameterTuning.BayesianSearch import BayesianSearch

    recommender_class = ItemKNNCFRecommender

    parameterSearch = BayesianSearch(recommender_class,
                                     evaluator_validation=evaluator_validation,
                                     evaluator_test=evaluator_test)

    from ParameterTuning.AbstractClassSearch import DictionaryKeys

    hyperparamethers_range_dictionary = {}
    hyperparamethers_range_dictionary["topK"] = [5, 10, 20, 50, 100, 150, 200, 300, 400, 500, 600, 700, 800]
    hyperparamethers_range_dictionary["shrink"] = [0, 10, 50, 100, 200, 300, 500, 1000]
    hyperparamethers_range_dictionary["similarity"] = ["cosine"]
    hyperparamethers_range_dictionary["normalize"] = [True, False]

    recommenderDictionary = {DictionaryKeys.CONSTRUCTOR_POSITIONAL_ARGS: [URM_train],
                             DictionaryKeys.CONSTRUCTOR_KEYWORD_ARGS: {},
                             DictionaryKeys.FIT_POSITIONAL_ARGS: dict(),
                             DictionaryKeys.FIT_KEYWORD_ARGS: dict(),
                             DictionaryKeys.FIT_RANGE_KEYWORD_ARGS: hyperparamethers_range_dictionary}

    output_root_path = "result_experiments/"

    import os

    # If directory does not exist, create
    if not os.path.exists(output_root_path):
        os.makedirs(output_root_path)

    output_root_path += recommender_class.RECOMMENDER_NAME

    n_cases = 2
    metric_to_optimize = "MAP"

    best_parameters = parameterSearch.search(recommenderDictionary,
                                             n_cases=n_cases,
                                             output_root_path=output_root_path,
                                             metric=metric_to_optimize)

    itemKNNCF = ItemKNNCFRecommender(URM_train)
    itemKNNCF.fit(**best_parameters)

    from FW_Similarity.CFW_D_Similarity_Linalg import CFW_D_Similarity_Linalg

    n_cases = 2
    metric_to_optimize = "MAP"

    best_parameters_ItemKNNCBF = parameterSearch.search(recommenderDictionary,
                                                        n_cases=n_cases,
                                                        output_root_path=output_root_path,
                                                        metric=metric_to_optimize)

    itemKNNCBF = ItemKNNCBFRecommender(ICM, URM_train)
    itemKNNCBF.fit(**best_parameters_ItemKNNCBF)

    """
    #_____________________________________________________________________
    from ParameterTuning.BayesianSearch import BayesianSearch
    from ParameterTuning.AbstractClassSearch import DictionaryKeys

    from ParameterTuning.AbstractClassSearch import EvaluatorWrapper

    evaluator_validation_tuning = EvaluatorWrapper(evaluator_validation)
    evaluator_test_tuning = EvaluatorWrapper(evaluator_test)

    recommender_class = CFW_D_Similarity_Linalg

    parameterSearch = BayesianSearch(recommender_class,
                                     evaluator_validation=evaluator_validation_tuning,
                                     evaluator_test=evaluator_test_tuning)

    hyperparamethers_range_dictionary = {}
    hyperparamethers_range_dictionary["topK"] = [5, 10, 20, 50, 100, 150, 200, 300, 400, 500, 600, 700, 800]
    hyperparamethers_range_dictionary["add_zeros_quota"] = range(0, 1)
    hyperparamethers_range_dictionary["normalize_similarity"] = [True, False]

    recommenderDictionary = {DictionaryKeys.CONSTRUCTOR_POSITIONAL_ARGS: [URM_train, ICM, itemKNNCF.W_sparse],
                             DictionaryKeys.CONSTRUCTOR_KEYWORD_ARGS: {},
                             DictionaryKeys.FIT_POSITIONAL_ARGS: dict(),
                             DictionaryKeys.FIT_KEYWORD_ARGS: dict(),
                             DictionaryKeys.FIT_RANGE_KEYWORD_ARGS: hyperparamethers_range_dictionary}

    output_root_path = "result_experiments/"

    import os

    # If directory does not exist, create
    if not os.path.exists(output_root_path):
        os.makedirs(output_root_path)

    output_root_path += recommender_class.RECOMMENDER_NAME

    n_cases = 2
    metric_to_optimize = "MAP"

    best_parameters_CFW_D = parameterSearch.search(recommenderDictionary,
                                                   n_cases=n_cases,
                                                   output_root_path=output_root_path,
                                                   metric=metric_to_optimize)

    CFW_weithing = CFW_D_Similarity_Linalg(URM_train, ICM, itemKNNCF.W_sparse)
    CFW_weithing.fit(**best_parameters_CFW_D)
    #___________________________________________________________________________________________-

    """

    from GraphBased.P3alphaRecommender import P3alphaRecommender

    P3alpha = P3alphaRecommender(URM_train)
    P3alpha.fit()

    from MatrixFactorization.PureSVD import PureSVDRecommender

    #pureSVD = PureSVDRecommender(URM_train)
    #pureSVD.fit()

    rec = HybridRec.HybridRec()

    S_UCM = b.get_S_UCM_KNN(b.get_UCM(ev.get_URM_train()), 600)
    S_ICM = b.build_S_ICM_knn(b.build_ICM(), 250)

    rec.fit(ev.get_URM_train(), ev.get_target_playlists(), ev.get_target_tracks(), ev.num_playlists_to_test,
            itemKNNCBF.W_sparse, itemKNNCF.W_sparse, P3alpha.W_sparse, is_test=True, alfa=0.7, avg=0.3)

    train_df = rec.recommend()

    if is_test:
        map5 = ev.map5(train_df)
        print('Hybrid MAP@10:', map5)
        return map5
    else:
        print('Prediction saved!')
        train_df.to_csv(os.path.dirname(os.path.realpath(__file__))[:-19] + "/all/sub.csv", sep=',', index=False)
        return 0


    #hybridrecommender = ItemKNNSimilarityHybridRecommender(URM_train, itemKNNCF.W_sparse, P3alpha.W_sparse)
    #hybridrecommender.fit(alpha=0.5)

    #print(evaluator_validation.evaluateRecommender(hybridrecommender))

    """
    n_cases = 2
    metric_to_optimize = "MAP"

    best_parameters_ItemKNNCBF = parameterSearch.search(recommenderDictionary,
                                                        n_cases=n_cases,
                                                        output_root_path=output_root_path,
                                                        metric=metric_to_optimize)

    itemKNNCBF = ItemKNNCBFRecommender(b.build_ICM(), URM_train, True)
    itemKNNCBF.fit(**best_parameters_ItemKNNCBF)
    
    #hybridrecommender = ItemKNNScoresHybridRecommender(URM_train, itemKNNCF, pureSVD)
    #hybridrecommender.fit(alpha=0.5)

    #print(evaluator_validation.evaluateRecommender(hybridrecommender))

    profile_length = np.ediff1d(URM_train.indptr)
    block_size = int(len(profile_length) * 0.05)
    sorted_users = np.argsort(profile_length)

    MAP_itemKNNCF_per_group = []
    MAP_itemKNNCBF_per_group = []
    MAP_pureSVD_per_group = []
    cutoff = 10

    for group_id in range(0, 10):
        start_pos = group_id * block_size
        end_pos = min((group_id + 1) * block_size, len(profile_length))

        users_in_group = sorted_users[start_pos:end_pos]

        users_in_group_p_len = profile_length[users_in_group]

        print("Group {}, average p.len {:.2f}, min {}, max {}".format(group_id,
                                                                      users_in_group_p_len.mean(),
                                                                      users_in_group_p_len.min(),
                                                                      users_in_group_p_len.max()))

        users_not_in_group_flag = np.isin(sorted_users, users_in_group, invert=True)
        users_not_in_group = sorted_users[users_not_in_group_flag]

        evaluator_test = SequentialEvaluator(URM_test, cutoff_list=[cutoff], ignore_users=users_not_in_group)

        results, _ = evaluator_test.evaluateRecommender(itemKNNCF)
        MAP_itemKNNCF_per_group.append(results[cutoff]["MAP"])

        results, _ = evaluator_test.evaluateRecommender(pureSVD)
        MAP_pureSVD_per_group.append(results[cutoff]["MAP"])

        results, _ = evaluator_test.evaluateRecommender(itemKNNCBF)
        MAP_itemKNNCBF_per_group.append(results[cutoff]["MAP"])

        import matplotlib.pyplot as pyplot

        pyplot.plot(MAP_itemKNNCF_per_group, label="itemKNNCF")
        pyplot.plot(MAP_itemKNNCBF_per_group, label="itemKNNCBF")
        pyplot.plot(MAP_pureSVD_per_group, label="pureSVD")
        pyplot.ylabel('MAP')
        pyplot.xlabel('User Group')
        pyplot.legend()
        pyplot.show()
"""

def hybrid_rec(is_test):
    print('*** Test Hybrid Recommender ***')

    b = Builder()
    ev = Evaluator(is_test=is_test)
    ev.split()
    rec = HybridRec.HybridRec()

    S_UCM = b.get_S_UCM_KNN(b.get_UCM(ev.get_URM_train()), 600)
    S_ICM = b.build_S_ICM_knn(b.build_ICM(), 250)
    Slim = SlimBPR.SlimBPR(ev.get_URM_train(),
                           epochs=1,
                           learning_rate=0.01,
                           positive_item_regularization=1,
                           negative_item_regularization=1
                           ).get_S_SLIM_BPR(500)

    rec.fit(ev.get_URM_train(), ev.get_target_playlists(), ev.get_target_tracks(), ev.num_playlists_to_test,
            S_ICM, S_UCM, Slim, is_test, alfa=0.3, avg=0.3)

    """
    0.30, 0.30
    alfa*((1-avg)*collab + avg*content) + (1-alfa)*slimBPR
    
    only collab     con knn=500 0.09080017548893707
                        knn=600 0.09085745115462485
    
    only content        knn=250 0.05537121844924659
                        knn=300 0.055101704695727706
                        
    only slim       con lr=0.01     epoch=1 0.09087007071213243
                        lr=0.001    epoch=8 0.09346656108877179
                        
    content+collab  con avg=0.20 0.
                        avg=0.30 0.09762916809334841
                                    
    all together    con alfa=0.40 0.10715025718387602
                        alfa=0.30 0.1082252839472891
    """

    train_df = rec.recommend()

    if is_test:
        map5 = ev.map5(train_df)
        print('Hybrid MAP@10:', map5)
        return map5
    else:
        print('Prediction saved!')
        train_df.to_csv(os.path.dirname(os.path.realpath(__file__))[:-19] + "/all/sub.csv", sep=',', index=False)
        return 0


def mf_bpr_rec(is_test):
    print('*** Test MF-BPR Recommender ***')

    ev = Evaluator(is_test=is_test)
    ev.split()
    rec = MFRec.MFRec()

    MfRec = MFBPR.MFBPR(ev.get_URM_train(),
                        nnz=0.1,
                        n_factors=20,
                        learning_rate=0.01,
                        epochs=100,
                        user_regularization=0.1,
                        positive_item_regularization=0.1,
                        negative_item_regularization=0.1)

    rec.fit(ev.get_URM_train(), ev.get_target_playlists(), ev.get_target_tracks(), ev.num_playlists_to_test, MfRec,
            is_test)

    train_df = rec.recommend()

    if is_test:
        map5 = ev.map5(train_df)
        print('Hybrid MAP@10:', map5)
        return map5
    else:
        print('Prediction saved!')
        train_df.to_csv(os.path.dirname(os.path.realpath(__file__))[:-19] + "/all/sub.csv", sep=',', index=False)
        return 0


def mf_als_rec(is_test):
    print('*** Test MF-ALS Recommender ***')

    conf = SparkConf().setAppName("MF-ALS Rec").setMaster("local")
    sc = SparkContext(conf=conf)

    b = Builder()
    ev = Evaluator(is_test=is_test)
    ev.split()

    UCM = b.get_UCM(ev.get_URM_train())

    target_playlists = ev.get_target_playlists()
    urm_train_indices = ev.get_URM_train().nonzero()
    ratings_list = []

    print('Creating RDD of tuples')
    for index in tqdm(range(0, urm_train_indices[0].size)):
        ratings_list.append(Rating(urm_train_indices[0][index], urm_train_indices[1][index], 1))

    ratings = sc.parallelize(ratings_list)

    model = ALS.trainImplicit(ratings, rank=10, iterations=5, alpha=0.01)

    dataframe_list = []

    print('Predicting...', flush=True)

    all_predictions = model.recommendProductsForUsers(10).filter(lambda r: r[0] in target_playlists)\
                                                         .collect()

    for u in tqdm(all_predictions):
        prediction = []
        for i in u[1]:
            prediction.append(i.product)
        dataframe_list.append([u[0], prediction])

    def get_id(e):
        return e[0]

    dataframe_list.sort(key=get_id)

    train_df = pd.DataFrame(dataframe_list, columns=['playlist_id', 'track_ids'])

    if is_test:
        map5 = ev.map5(train_df)
        print('Hybrid MAP@10:', map5)
        return map5
    else:
        print('Prediction saved!')
        train_df.to_csv(os.path.dirname(os.path.realpath(__file__))[:-19] + "/all/sub.csv", sep=',', index=False)
        return 0

from Base.Recommender import Recommender

class ItemKNNScoresHybridRecommender(Recommender):
    """ ItemKNNScoresHybridRecommender
    Hybrid of two prediction scores R = R1*alpha + R2*(1-alpha)

    """

    RECOMMENDER_NAME = "ItemKNNScoresHybridRecommender"

    def __init__(self, URM_train, Recommender_1, Recommender_2):
        super(ItemKNNScoresHybridRecommender, self).__init__()

        self.URM_train = check_matrix(URM_train.copy(), 'csr')
        self.Recommender_1 = Recommender_1
        self.Recommender_2 = Recommender_2

        self.compute_item_score = self.compute_score_hybrid

    def fit(self, alpha=0.5):
        self.alpha = alpha

    def compute_score_hybrid(self, user_id_array):
        item_weights_1 = self.Recommender_1.compute_item_score(user_id_array)
        item_weights_2 = self.Recommender_2.compute_item_score(user_id_array)

        item_weights = item_weights_1 * self.alpha + item_weights_2 * (1 - self.alpha)

        return item_weights

from Base.Recommender import Recommender
from Base.Recommender_utils import check_matrix, similarityMatrixTopK
from Base.SimilarityMatrixRecommender import SimilarityMatrixRecommender


class ItemKNNSimilarityHybridRecommender(SimilarityMatrixRecommender, Recommender):
    """ ItemKNNSimilarityHybridRecommender
    Hybrid of two similarities S = S1*alpha + S2*(1-alpha)

    """

    RECOMMENDER_NAME = "ItemKNNSimilarityHybridRecommender"


    def __init__(self, URM_train, Similarity_1, Similarity_2, sparse_weights=True):
        super(ItemKNNSimilarityHybridRecommender, self).__init__()

        if Similarity_1.shape != Similarity_2.shape:
            raise ValueError("ItemKNNSimilarityHybridRecommender: similarities have different size, S1 is {}, S2 is {}".format(
                Similarity_1.shape, Similarity_2.shape
            ))

        # CSR is faster during evaluation
        self.Similarity_1 = check_matrix(Similarity_1.copy(), 'csr')
        self.Similarity_2 = check_matrix(Similarity_2.copy(), 'csr')

        self.URM_train = check_matrix(URM_train.copy(), 'csr')

        self.sparse_weights = sparse_weights


    def fit(self, topK=100, alpha = 0.5):

        self.topK = topK
        self.alpha = alpha

        W = self.Similarity_1*self.alpha + self.Similarity_2*(1-self.alpha)

        if self.sparse_weights:
            self.W_sparse = similarityMatrixTopK(W, forceSparseOutput=True, k=self.topK)
        else:
            self.W = similarityMatrixTopK(W, forceSparseOutput=False, k=self.topK)
