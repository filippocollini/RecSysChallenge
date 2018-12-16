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

# VERY VERY SLOW AND USES A LOT OF RAM :(

cdef class KnnRecCython:

    cdef long[:] user_profile
    cdef float[:,:] W_sparse
    cdef double[:] scores
    cdef long[:] ranking
    cdef int[:] seen
    cdef int[:] unseen_mask

    def __init__(self, W_sparse):
        self.W_sparse = W_sparse

    def recommend(self, user_profile, at=None, exclude_seen=True):
        # compute the scores using the dot product
        self.user_profile = user_profile.ravel()
        self.scores = np.dot(self.user_profile, self.W_sparse).ravel()

        if exclude_seen:
            self.scores = self._filter_seen()

        # rank items
        self.ranking = np.argsort(self.scores)[::-1]

        return self.ranking[:at]

    cdef double[:] _filter_seen(self):

        return np.delete(self.scores, np.nonzero(self.user_profile))
