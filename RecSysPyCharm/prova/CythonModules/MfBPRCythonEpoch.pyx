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

cdef struct Triplet:
    long user
    long pos_item
    long neg_item

cdef class MfBPRCythonEpoch:

    cdef int n_users, n_items, URM_nnz, n_factors

    cdef float learning_rate, positive_item_regularization, negative_item_regularization, user_regularization, nnz

    cdef int[:] userSeenItems

    cdef int[:] URM_mask_indices, URM_mask_indptr

    cdef double[:,:] user_factors, item_factors

    def __init__(self, URM,
                 nnz=0.1,
                 n_factors=10,
                 learning_rate=0.1,
                 user_regularization=0.1,
                 positive_item_regularization=0.1,
                 negative_item_regularization=0.1):

        self.n_factors = n_factors
        self.learning_rate = learning_rate

        self.user_regularization = user_regularization
        self.positive_item_regularization = positive_item_regularization
        self.negative_item_regularization = negative_item_regularization
        self.nnz = nnz

        self.n_users = URM.shape[0]
        self.n_items = URM.shape[1]
        self.URM_nnz = int(URM.nnz)
        self.URM_mask_indices = URM.indices
        self.URM_mask_indptr = URM.indptr

        self.user_factors = np.random.random_sample((self.n_users, n_factors))
        self.item_factors = np.random.random_sample((self.n_items, n_factors))

    def epochIteration(self):

        cdef Triplet triplet

        cdef long u, i, j

        cdef double z, x, d

        # Get number of available interactions
        cdef int numPositiveIteractions = int(self.URM_nnz * self.nnz)

        # Uniform user sampling without replacement
        for it in range(numPositiveIteractions):

            # Sample
            triplet = self.sampleTriplet()
            u = triplet.user
            i = triplet.pos_item
            j = triplet.neg_item

            # Apply SGD update

            x = 0.0
            for index in range(self.n_factors):
                x += self.user_factors[u, index] * (self.item_factors[i, index] - self.item_factors[j, index])

            z = 1.0 / (1.0 + np.exp(x))

            for index in range(self.n_factors):
                self.user_factors[u, index] += self.learning_rate * \
                                           ((self.item_factors[i, index] - self.item_factors[j, index]) * z
                                            - self.user_regularization * self.user_factors[u, index])

                self.item_factors[i, index] += self.learning_rate * \
                                               (self.user_factors[u, index] * z - self.positive_item_regularization
                                                * self.item_factors[i, index])

                self.item_factors[j, index] += self.learning_rate * \
                                               (-self.user_factors[u, index] * z - self.negative_item_regularization
                                                * self.item_factors[j, index])

    cdef Triplet sampleTriplet(self):

        cdef Triplet triplet = Triplet()

        # By randomly selecting a user in this way we could end up
        # with a user with no interactions
        # user_id = np.random.randint(0, self.n_users)

        triplet.user = np.random.choice(self.n_users)

        # Get user seen items and choose one
        self.userSeenItems = self.getSeenItems(triplet.user)
        triplet.pos_item = np.random.choice(self.userSeenItems)

        negItemSelected = False

        # It's faster to just try again then to build a mapping of the non-seen items
        while not negItemSelected:
            triplet.neg_item = np.random.randint(0, self.n_items)

            if triplet.neg_item not in self.userSeenItems:
                negItemSelected = True

        return triplet

    cdef int[:] getSeenItems(self, long index):
        return self.URM_mask_indices[self.URM_mask_indptr[index]:self.URM_mask_indptr[index + 1]]

    def get_prediction(self, u):
        return np.dot(self.user_factors[u], self.item_factors.T)
