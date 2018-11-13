#cython: boundscheck=False
#cython: wraparound=False
#cython: initializedcheck=False
#cython: language_level=3
#cython: nonecheck=False
#cython: cdivision=True
#cython: unpack_method_calls=True
#cython: overflowcheck=False

#defining NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

cimport numpy as np
import numpy as np

cdef class SlimBPRCythonEpoch:

    cdef int n_users, n_items, nnz

    cdef float learning_rate, positive_item_regularization, negative_item_regularization

    cdef double[:,:] similarity_matrix

    def __init__(self, URM,
                 nnz = 1,
                 learning_rate = 0.01,
                 positive_item_regularization = 1.0,
                 negative_item_regularization = 1.0):
        self.nnz = nnz
        self.learning_rate = learning_rate
        self.positive_item_regularization = positive_item_regularization
        self.negative_item_regularization = negative_item_regularization
        self.n_users = URM.shape[0]
        self.n_items = URM.shape[1]
        self.URM = URM

    def sampleTriplet(self):

        # By randomly selecting a user in this way we could end up
        # with a user with no interactions
        # user_id = np.random.randint(0, self.n_users)

        user_id = np.random.choice(self.n_users)

        # Get user seen items and choose one
        userSeenItems = self.URM[user_id, :].indices
        pos_item_id = np.random.choice(userSeenItems)

        negItemSelected = False

        # It's faster to just try again then to build a mapping of the non-seen items
        while (not negItemSelected):
            neg_item_id = np.random.randint(0, self.n_items)

            if (neg_item_id not in userSeenItems):
                negItemSelected = True

        return user_id, pos_item_id, neg_item_id

    def epochIteration(self):

        # Get number of available interactions
        numPositiveIteractions = int(self.URM.nnz * self.nnz)

        # Uniform user sampling without replacement
        for num_sample in tqdm(range(numPositiveIteractions)):

            # Sample
            user_id, positive_item_id, negative_item_id = self.sampleTriplet()

            userSeenItems = self.URM[user_id, :].indices

            # Prediction
            x_i = self.similarity_matrix[positive_item_id, userSeenItems].sum()
            x_j = self.similarity_matrix[negative_item_id, userSeenItems].sum()

            # Gradient
            x_ij = x_i - x_j

            gradient = 1 / (1 + np.exp(x_ij))

            for i in userSeenItems:
                dp = gradient - self.positive_item_regularization * x_i
                self.similarity_matrix[positive_item_id, i] = self.similarity_matrix[positive_item_id, i] +\
                    self.learning_rate * dp
                dn = gradient - self.negative_item_regularization * x_j
                self.similarity_matrix[negative_item_id, i] = self.similarity_matrix[negative_item_id, i] -\
                    self.learning_rate * dn

            self.similarity_matrix[positive_item_id, positive_item_id] = 0
            self.similarity_matrix[negative_item_id, negative_item_id] = 0