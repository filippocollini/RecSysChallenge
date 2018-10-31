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

cdef class KnnRecCython:

    cdef long[:,:] user_profile
    cdef double[:] scores
    cdef int[:] seen
    cdef int[:] unseen_mask
    cdef int[:,:] dataset
    cdef float[:,:] W_sparse

    def __init__(self, W_sparse):
        self.W_sparse = W_sparse.todense()

    def recommend(self, user_profile, at=None, exclude_seen=True):
        # compute the scores using the dot product
        self.user_profile = user_profile
        self.scores = np.dot(self.user_profile, self.W_sparse).ravel()

        # rank items
        ranking = np.argsort(self.scores)[::-1]
        if exclude_seen:
            ranking = self._filter_seen(ranking, user_profile)

        return ranking[:at]

    cdef long[:] _filter_seen(self, ranking, user_profile):

        self.seen = np.array(user_profile[0]).indices
        self.unseen_mask = np.in1d(ranking, self.seen, assume_unique=True, invert=True)

        return ranking[self.unseen_mask]
