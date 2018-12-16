import Recsys as rs

"""
From here you can make prediction or test each recommender.
Just uncomment one of the following line and run.

Hyperparameters for each recommender can be changed in the
Recsys.py file.

"""
n = 1
avg_map = 0
for i in range(n):
    avg_map += rs.hybrid_repo(False)
#print('Average MAP@10:', avg_map/n)


#rs.hybrid_repo()

"""
    only collab     con knn=500 0.09554863107234404
                        knn=600 0.09585788899429891
    only content        knn=250 0.
                        knn=300 0.050498162386506486
                        
    content+collab  con avg=0.20 0.09957571895841204
                        avg=0.30 0.09984266811095445
                        avg=0.40 0.09877481446868795

    only slim       con lr=0.01     epoch=1 0.0874042765419505
                        lr=0.001    epoch=4 0.089788462052809
                                    epoch=8 0.09238170363133094

    all together    con alfa=0.60 0.10745100746215795
                        alfa=0.50 0.10748996628419637
    """
# rs.mf_bpr_rec(is_test=True)
# rs.top_pop_rec()
# rs.item_based(is_test=True)
# rs.round_robin_rec(is_test=True, avg_mode=False)
# rs.round_robin_rec(is_test=True, avg_mode=True)
# rs.item_based(is_test=True)
# rs.SVD(is_test=True)
# rs.item_user_avg(is_test=True)
# rs.collaborative_filtering(is_test=True)
# rs.mf_als_rec(is_test=True)



