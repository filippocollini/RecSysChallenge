from Builder import Builder
from Evaluator import Evaluator
from Recommenders import ItemBasedRec, CollaborativeFilteringRec, ItemUserAvgRec,\
    SlimBPRRec, SVDRec, RoundRobinRec, HybridRec, TopPopRec, MFRec
import SlimBPR, MFBPR
import os as os
from pyspark.mllib.recommendation import ALS, Rating
from pyspark import SparkContext, SparkConf
from tqdm import tqdm
import pandas as pd
import time


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

