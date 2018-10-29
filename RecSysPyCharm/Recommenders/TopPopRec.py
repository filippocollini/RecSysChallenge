import numpy as np


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
