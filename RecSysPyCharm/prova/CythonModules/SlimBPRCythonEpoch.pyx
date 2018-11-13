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
import tqdm

cdef struct Triplet:
    long user
    long pos_item
    long neg_item

cdef class SlimBPRCythonEpoch:

    cdef int n_users, n_items, nnz, URM_nnz

    cdef float learning_rate, positive_item_regularization, negative_item_regularization

    cdef int[:] userSeenItems

    cdef int[:] URM_mask_indices, URM_mask_indptr

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
        self.URM_nnz = int(URM.nnz)
        self.URM_mask_indices = URM.indices
        self.URM_mask_indptr = URM.indptr

        self.similarity_matrix = np.zeros((self.n_items, self.n_items))

    cdef int[:] getSeenItems(self, long index):
        return self.URM_mask_indices[self.URM_mask_indptr[index]:self.URM_mask_indptr[index + 1]]

    def get_similarity_matrix(self):

        return np.array(self.similarity_matrix)

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

    def epochIteration(self):

        cdef Triplet triplet

        cdef long user_id, positive_item_id, negative_item_id

        cdef double gradient, x_ij, dp, dn, x_i, x_j

        # Get number of available interactions
        cdef int numPositiveIteractions = int(self.URM_nnz * self.nnz)

        # Uniform user sampling without replacement
        for num_sample in range(numPositiveIteractions):

            # Sample
            triplet = self.sampleTriplet()
            user_id = triplet.user
            positive_item_id = triplet.pos_item
            negative_item_id = triplet.neg_item

            userSeenItems = self.getSeenItems(triplet.user)

            x_i = 0
            x_j = 0

            # Prediction
            for index in userSeenItems:
                x_i += self.similarity_matrix[positive_item_id, index]
                x_j += self.similarity_matrix[negative_item_id, index]

            # Gradient
            x_ij = x_i - x_j

            gradient = 1 / (1 + np.exp(x_ij))

            for index in userSeenItems:
                dp = gradient - self.positive_item_regularization * x_i
                self.similarity_matrix[positive_item_id, index] += self.learning_rate * dp
                dn = gradient - self.negative_item_regularization * x_j
                self.similarity_matrix[negative_item_id, index] -= self.learning_rate * dn

            self.similarity_matrix[positive_item_id, positive_item_id] = 0
            self.similarity_matrix[negative_item_id, negative_item_id] = 0