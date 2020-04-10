# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False

# Authors: Gilles Louppe <g.louppe@gmail.com>
#          Peter Prettenhofer <peter.prettenhofer@gmail.com>
#          Brian Holt <bdholt1@gmail.com>
#          Noel Dawe <noel@dawe.me>
#          Satrajit Gosh <satrajit.ghosh@gmail.com>
#          Lars Buitinck <L.J.Buitinck@uva.nl>
#          Arnaud Joly <arnaud.v.joly@gmail.com>
#          Joel Nothman <joel.nothman@gmail.com>
#          Fares Hedayati <fares.hedayati@gmail.com>
#          Jacob Schreiber <jmschreiber91@gmail.com>
#          Ken Jung <Ken Jung >
#
# Licence: BSD 3 clause

from ._criterion cimport Criterion

from libc.stdlib cimport free
from libc.stdlib cimport qsort
from libc.string cimport memcpy
from libc.string cimport memset
from libc.stdio cimport printf

import numpy as np
cimport numpy as np
np.import_array()

from scipy.sparse import csc_matrix

from ._utils cimport log
from ._utils cimport rand_int
from ._utils cimport rand_uniform
from ._utils cimport RAND_R_MAX
from ._utils cimport safe_realloc
from ._criterion cimport PowersCriterion
from ._criterion cimport VarianceCriterion

cdef double INFINITY = np.inf

# Mitigate precision differences between 32 bit and 64 bit
cdef DTYPE_t FEATURE_THRESHOLD = 1e-7

# For Powers method, we don't bother with splits that lead to objective gains less than this.
cdef DTYPE_t MIN_OBJECTIVE_IMPROVEMENT = 1e-7

# Constant to switch between algorithm non zero value extract algorithm
# in SparseSplitter
cdef DTYPE_t EXTRACT_NNZ_SWITCH = 0.1

cdef inline void _init_split(SplitRecord* self, SIZE_t start_pos) nogil:
    self.impurity_left = INFINITY
    self.impurity_right = INFINITY
    self.pos = start_pos
    self.feature = 0
    self.threshold = 0.
    self.improvement = -INFINITY

cdef class Splitter:
    """Abstract splitter class.
    Splitters are called by tree builders to find the best splits on both
    sparse and dense data, one split at a time.
    """

    def __cinit__(self, Criterion criterion, SIZE_t max_features,
                  SIZE_t min_samples_leaf, double min_weight_leaf,
                  object random_state, bint presort):
        """
        Parameters
        ----------
        criterion: Criterion
            The criterion to measure the quality of a split.
        max_features: SIZE_t
            The maximal number of randomly selected features which can be
            considered for a split.
        min_samples_leaf: SIZE_t
            The minimal number of samples each leaf can have, where splits
            which would result in having less samples in a leaf are not
            considered.
        min_weight_leaf: double
            The minimal weight each leaf can have, where the weight is the sum
            of the weights of each sample in it.
        random_state: object
            The user inputted random state to be used for pseudo-randomness
        """

        self.criterion = criterion

        self.samples = NULL
        self.n_samples = 0
        self.features = NULL
        self.n_features = 0
        self.feature_values = NULL

        self.y = NULL
        self.y_stride = 0
        self.sample_weight = NULL

        self.max_features = max_features
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_leaf = min_weight_leaf
        self.random_state = random_state
        self.presort = presort

    def __dealloc__(self):
        """Destructor."""

        free(self.samples)
        free(self.features)
        free(self.constant_features)
        free(self.feature_values)

    def __getstate__(self):
        return {}

    def __setstate__(self, d):
        pass

    cdef void init(self,
                   object X,
                   np.ndarray[DOUBLE_t, ndim=2, mode="c"] y,
                   DOUBLE_t* sample_weight,
                   np.ndarray X_idx_sorted=None) except *:
        """Initialize the splitter.
        Take in the input data X, the target Y, and optional sample weights.
        Parameters
        ----------
        X: object
            This contains the inputs. Usually it is a 2d numpy array.
        y: numpy.ndarray, dtype=DOUBLE_t
            This is the vector of targets, or true labels, for the samples
        sample_weight: numpy.ndarray, dtype=DOUBLE_t (optional)
            The weights of the samples, where higher weighted samples are fit
            closer than lower weight samples. If not provided, all samples
            are assumed to have uniform weight.
        """

        self.rand_r_state = self.random_state.randint(0, RAND_R_MAX)
        cdef SIZE_t n_samples = X.shape[0]

        # Create a new array which will be used to store nonzero
        # samples from the feature of interest
        cdef SIZE_t* samples = safe_realloc(&self.samples, n_samples)

        cdef SIZE_t i, j
        cdef double weighted_n_samples = 0.0
        j = 0

        for i in range(n_samples):
            # Only work with positively weighted samples
            if sample_weight == NULL or sample_weight[i] != 0.0:
                samples[j] = i
                j += 1

            if sample_weight != NULL:
                weighted_n_samples += sample_weight[i]
            else:
                weighted_n_samples += 1.0

        # Number of samples is number of positively weighted samples
        self.n_samples = j
        self.weighted_n_samples = weighted_n_samples

        cdef SIZE_t n_features = X.shape[1]
        cdef SIZE_t* features = safe_realloc(&self.features, n_features)

        for i in range(n_features):
            features[i] = i

        self.n_features = n_features

        safe_realloc(&self.feature_values, n_samples)
        safe_realloc(&self.constant_features, n_features)

        self.y = <DOUBLE_t*> y.data
        self.y_stride = <SIZE_t> y.strides[0] / <SIZE_t> y.itemsize

        self.sample_weight = sample_weight

    cdef void node_reset(self, SIZE_t start, SIZE_t end,
                         double* weighted_n_node_samples) nogil:
        """Reset splitter on node samples[start:end].
        Parameters
        ----------
        start: SIZE_t
            The index of the first sample to consider
        end: SIZE_t
            The index of the last sample to consider
        weighted_n_node_samples: numpy.ndarray, dtype=double pointer
            The total weight of those samples
        """

        self.start = start
        self.end = end

        self.criterion.init(self.y,
                            self.y_stride,
                            self.sample_weight,
                            self.weighted_n_samples,
                            self.samples,
                            start,
                            end)

        weighted_n_node_samples[0] = self.criterion.weighted_n_node_samples

    cdef void node_split(self, double impurity, SplitRecord* split,
                         SIZE_t* n_constant_features) nogil:
        """Find the best split on node samples[start:end].
        This is a placeholder method. The majority of computation will be done
        here.
        """

        pass

    cdef void node_value(self, double* dest) nogil:
        """Copy the value of node samples[start:end] into dest."""

        self.criterion.node_value(dest)

    cdef double node_impurity(self) nogil:
        """Return the impurity of the current node."""

        return self.criterion.node_impurity()


cdef class PowersSplitter:
    """Splitter class for using Scott Powers' split criterion.
    Splitters are called by tree builders to find the best splits on both
    sparse and dense data, one split at a time.
    """
    def __reduce__(self):
        return (PowersSplitter, (self.criterion,
                                 self.max_features,
                                 self.min_samples_leaf,
                                 self.min_weight_leaf,
                                 self.random_state,
                                 self.presort), self.__getstate__())

    def __cinit__(self, PowersCriterion criterion, SIZE_t max_features,
                  SIZE_t min_samples_leaf, double min_weight_leaf,
                  object random_state, bint presort):
        """
        Parameters
        ----------
        criterion: Criterion
            The criterion to measure the quality of a split.
        max_features: SIZE_t
            The maximal number of randomly selected features which can be
            considered for a split.
        min_samples_leaf: SIZE_t
            The minimal number of samples each leaf can have, where splits
            which would result in having less samples in a leaf are not
            considered.
        min_weight_leaf: double
            The minimal weight each leaf can have, where the weight is the sum
            of the weights of each sample in it.
        random_state: object
            The user inputted random state to be used for pseudo-randomness
        """

        self.criterion = criterion

        self.samples = NULL
        self.n_samples = 0
        self.features = NULL
        self.n_features = 0
        self.feature_values = NULL

        self.y = NULL
        self.y_stride = 0
        self.w = NULL
        self.w_stride = 0
        self.sample_weight = NULL
        self.X_sample_stride = 0
        self.X_feature_stride = 0

        self.max_features = max_features
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_leaf = min_weight_leaf
        self.random_state = random_state
        self.presort = presort

    def __dealloc__(self):
        """Destructor."""

        free(self.samples)
        free(self.features)
        free(self.constant_features)
        free(self.feature_values)

    def __getstate__(self):
        return {}

    def __setstate__(self, d):
        pass

    cdef void init(self,
                   object X,
                   np.ndarray[DOUBLE_t, ndim=2, mode="c"] y,
                   np.ndarray[DOUBLE_t, ndim=2, mode="c"] w,
                   DOUBLE_t* sample_weight,
                   np.ndarray X_idx_sorted=None) except *:
        """Initialize the splitter.
        Take in the input data X, the outcomes Y, the treatment indicators w,
        and optional sample weights.
        Parameters
        ----------
        X: object
            This contains the inputs. Usually it is a 2d numpy array.
        y: numpy.ndarray, dtype=DOUBLE_t
            This is the vector of outcomes for the samples
        w: numpy.ndarray, dtype=DOUBLE_t
            This is the vector of treatment indicators for the samples

        sample_weight: numpy.ndarray, dtype=DOUBLE_t (optional)
            The weights of the samples, where higher weighted samples are fit
            closer than lower weight samples. If not provided, all samples
            are assumed to have uniform weight.
        """

        self.rand_r_state = self.random_state.randint(0, RAND_R_MAX)
        cdef SIZE_t n_samples = X.shape[0]

        # Create a new array which will be used to store nonzero
        # samples from the feature of interest
        cdef SIZE_t* samples = safe_realloc(&self.samples, n_samples)

        cdef SIZE_t i, j
        cdef double weighted_n_samples = 0.0
        j = 0

        for i in range(n_samples):
            # Only work with positively weighted samples
            if sample_weight == NULL or sample_weight[i] != 0.0:
                samples[j] = i
                j += 1

            if sample_weight != NULL:
                weighted_n_samples += sample_weight[i]
            else:
                weighted_n_samples += 1.0

        # Number of samples is number of positively weighted samples
        self.n_samples = j
        self.weighted_n_samples = weighted_n_samples

        cdef SIZE_t n_features = X.shape[1]
        cdef SIZE_t* features = safe_realloc(&self.features, n_features)

        for i in range(n_features):
            features[i] = i

        self.n_features = n_features

        safe_realloc(&self.feature_values, n_samples)
        safe_realloc(&self.constant_features, n_features)

        self.y = <DOUBLE_t*> y.data
        self.y_stride = <SIZE_t> y.strides[0] / <SIZE_t> y.itemsize
        self.w = <DOUBLE_t*> w.data
        self.w_stride = <SIZE_t> w.strides[0] / <SIZE_t> w.itemsize

        self.sample_weight = sample_weight

        # Initialize X
        cdef np.ndarray X_ndarray = X

        self.X = <DTYPE_t*> X_ndarray.data
        self.X_sample_stride = <SIZE_t> X.strides[0] / <SIZE_t> X.itemsize
        self.X_feature_stride = <SIZE_t> X.strides[1] / <SIZE_t> X.itemsize

        if self.presort == 1:
            self.X_idx_sorted = X_idx_sorted
            self.X_idx_sorted_ptr = <INT32_t*> self.X_idx_sorted.data
            self.X_idx_sorted_stride = (<SIZE_t> self.X_idx_sorted.strides[1] /
                                        <SIZE_t> self.X_idx_sorted.itemsize)

            self.n_total_samples = X.shape[0]
            safe_realloc(&self.sample_mask, self.n_total_samples)
            memset(self.sample_mask, 0, self.n_total_samples*sizeof(SIZE_t))


    cdef void node_reset(self, SIZE_t start, SIZE_t end,
                         double* weighted_n_node_samples) nogil:
        """Reset splitter on node samples[start:end].
        Parameters
        ----------
        start: SIZE_t
            The index of the first sample to consider
        end: SIZE_t
            The index of the last sample to consider
        weighted_n_node_samples: numpy.ndarray, dtype=double pointer
            The total weight of those samples
        """
        self.start = start
        self.end = end
        self.criterion.init(self.y,
                            self.y_stride,
                            self.w,
                            self.w_stride,
                            self.sample_weight,
                            self.weighted_n_samples,
                            self.samples,
                            start,
                            end)
        weighted_n_node_samples[0] = self.criterion.weighted_n_node_samples

    cdef INT32_t min_class_count(self, SIZE_t start, SIZE_t end) nogil:
        """Count the min number of samples in the range [start:end[,
           with respect to the treatment w (assume binary tx).   """
        cdef INT32_t count0 = 0
        cdef INT32_t count1 = 0
        cdef SIZE_t i
        for i in range(start, end):
            if (self.w[self.samples[i]] == 0):
                count0 += 1
            else:
                count1 += 1
        if count0 < count1:
            return count0
        else:
            return count1

    cdef void node_split(self, SplitRecord* split,
                         SIZE_t* n_constant_features) nogil:
        """Find the best split on node samples[start:end]."""
        # Find the best split
        cdef SIZE_t* samples = self.samples
        cdef SIZE_t start = self.start
        cdef SIZE_t end = self.end

        cdef SIZE_t* features = self.features
        cdef SIZE_t* constant_features = self.constant_features
        cdef SIZE_t n_features = self.n_features

        cdef DTYPE_t* X = self.X
        cdef DTYPE_t* Xf = self.feature_values
        cdef SIZE_t X_sample_stride = self.X_sample_stride
        cdef SIZE_t X_feature_stride = self.X_feature_stride
        cdef SIZE_t max_features = self.max_features
        cdef SIZE_t min_samples_leaf = self.min_samples_leaf
        cdef double min_weight_leaf = self.min_weight_leaf
        cdef UINT32_t* random_state = &self.rand_r_state

        cdef INT32_t* X_idx_sorted = self.X_idx_sorted_ptr
        cdef SIZE_t* sample_mask = self.sample_mask

        cdef SplitRecord best, current
        cdef double current_objective_improvement = -INFINITY
        cdef double best_objective_improvement = -INFINITY

        cdef SIZE_t f_i = n_features
        cdef SIZE_t f_j
        cdef SIZE_t tmp
        cdef SIZE_t p
        cdef SIZE_t feature_idx_offset
        cdef SIZE_t feature_offset
        cdef SIZE_t i
        cdef SIZE_t j

        cdef SIZE_t n_visited_features = 0
        # Number of features discovered to be constant during the split search
        cdef SIZE_t n_found_constants = 0
        # Number of features known to be constant and drawn without replacement
        cdef SIZE_t n_drawn_constants = 0
        cdef SIZE_t n_known_constants = n_constant_features[0]
        # n_total_constants = n_known_constants + n_found_constants
        cdef SIZE_t n_total_constants = n_known_constants
        cdef DTYPE_t current_feature_value
        cdef SIZE_t partition_end
        cdef INT32_t min_num_in_class

        _init_split(&best, end)

        if self.presort == 1:
            for p in range(start, end):
                sample_mask[samples[p]] = 1  # sample_mask indicates what samples
                # to use; uses canonical ordering, not samples[] ordering.

        # Sample up to max_features without replacement using a
        # Fisher-Yates-based algorithm (using the local variables `f_i` and
        # `f_j` to compute a permutation of the `features` array).
        #
        # Skip the CPU intensive evaluation of the impurity criterion for
        # features that were already detected as constant (hence not suitable
        # for good splitting) by ancestor nodes and save the information on
        # newly discovered constant features to spare computation on descendant
        # nodes.

        while (f_i > n_total_constants and  # Stop early if remaining features
                                            # are constant
                (n_visited_features < max_features or
                 # At least one drawn features must be non constant
                 n_visited_features <= n_found_constants + n_drawn_constants)):

            n_visited_features += 1

            # Loop invariant: elements of features in
            # - [:n_drawn_constant[ holds drawn and known constant features;
            # - [n_drawn_constant:n_known_constant[ holds known constant
            #   features that haven't been drawn yet;
            # - [n_known_constant:n_total_constant[ holds newly found constant
            #   features;
            # - [n_total_constant:f_i[ holds features that haven't been drawn
            #   yet and aren't constant apriori.
            # - [f_i:n_features[ holds features that have been drawn
            #   and aren't constant.

            # Draw a feature at random
            f_j = rand_int(n_drawn_constants, f_i - n_found_constants,
                           random_state)
            if f_j < n_known_constants:
                # f_j in the interval [n_drawn_constants, n_known_constants[
                tmp = features[f_j]
                features[f_j] = features[n_drawn_constants]
                features[n_drawn_constants] = tmp

                n_drawn_constants += 1

            else:
                # f_j in the interval [n_known_constants, f_i - n_found_constants[
                f_j += n_found_constants
                # f_j in the interval [n_total_constants, f_i[
                current.feature = features[f_j]
                feature_offset = self.X_feature_stride * current.feature

                # Sort samples along that feature; either by utilizing
                # presorting, or by copying the values into an array and
                # sorting the array in a manner which utilizes the cache more
                # effectively.
                if self.presort == 1:
                    p = start
                    feature_idx_offset = self.X_idx_sorted_stride * current.feature

                    for i in range(self.n_total_samples):
                        j = X_idx_sorted[i + feature_idx_offset]
                        if sample_mask[j] == 1:
                            samples[p] = j
                            Xf[p] = X[self.X_sample_stride * j + feature_offset]
                            p += 1
                else:
                    for i in range(start, end):
                        Xf[i] = X[self.X_sample_stride * samples[i] + feature_offset]
                    sort(Xf + start, samples + start, end - start)

                if Xf[end - 1] <= Xf[start] + FEATURE_THRESHOLD:
                    features[f_j] = features[n_total_constants]
                    features[n_total_constants] = current.feature

                    n_found_constants += 1
                    n_total_constants += 1

                else:
                    f_i -= 1
                    features[f_i], features[f_j] = features[f_j], features[f_i]

                    # Evaluate all splits; this is largely a hack to decrease computation
                    # by doing min necessary work to evaluate each split in turn.  It all
                    # relies on having a sorted vector of feature values so we can march along
                    # it in order to evaluate different split points...
                    self.criterion.reset()
                    p = start

                    # We have a sorted vector of feature values; here we iterate through them, checking
                    # each as a potential split point.
                    while p < end:
                        # It is possible to have a run of identical feature values...
                        while (p + 1 < end and
                               Xf[p + 1] <= Xf[p] + FEATURE_THRESHOLD):
                            p += 1

                        # (p + 1 >= end) or (X[samples[p + 1], current.feature] >
                        #                    X[samples[p], current.feature])
                        p += 1
                        # (p >= end) or (X[samples[p], current.feature] >
                        #                X[samples[p - 1], current.feature])

                        if p < end:
                            current.pos = p

                            # Reject if min_samples_leaf is not guaranteed
                            # if (((current.pos - start) < min_samples_leaf) or
                            #         ((end - current.pos) < min_samples_leaf)):
                            #     continue
                            min_num_in_class = self.min_class_count(start, current.pos)
                            if min_num_in_class <= min_samples_leaf :
                                continue
                            min_num_in_class = self.min_class_count(current.pos, end)
                            if min_num_in_class <= min_samples_leaf :
                                continue

                            # Calculate improvement using current.pos for threshold.
                            # This calculates sufficient statistics (sum of outcomes per
                            # treatment group, plus sum of squared outcomes per treatment
                            # group, along with cardinality) in each of the children, using
                            # current.pos as the split point.
                            self.criterion.update(current.pos)

                            # Reject if min_weight_leaf is not satisfied
                            # if ((self.criterion.weighted_n_left < min_weight_leaf) or
                            #         (self.criterion.weighted_n_right < min_weight_leaf)):
                            #     continue
                            current_objective_improvement = self.criterion.objective_improvement()
                            if current_objective_improvement <= MIN_OBJECTIVE_IMPROVEMENT :
                                continue

                            if current_objective_improvement > best_objective_improvement:
                                best_objective_improvement = current_objective_improvement
                                current.threshold = (Xf[p - 1] + Xf[p]) / 2.0
                                current.improvement = current_objective_improvement
                                if current.threshold == Xf[p]:
                                    current.threshold = Xf[p - 1]
                                best = current  # copy

        # Reorganize into samples[start:best.pos] + samples[best.pos:end]
        if best.pos < end:
            # Only do this if we found a good split...
            feature_offset = X_feature_stride * best.feature
            partition_end = end
            p = start

            while p < partition_end:
                if X[X_sample_stride * samples[p] + feature_offset] <= best.threshold:
                    p += 1

                else:
                    partition_end -= 1

                    tmp = samples[partition_end]
                    samples[partition_end] = samples[p]
                    samples[p] = tmp
            # Don't think any of this is necessary...
            # self.criterion.reset()
            # self.criterion.update(best.pos)
            # best.improvement = self.criterion.impurity_improvement(impurity)
            # self.criterion.children_impurity(&best.impurity_left,
            #                                  &best.impurity_right)

        # Reset sample mask
        if self.presort == 1:
            for p in range(start, end):
                sample_mask[samples[p]] = 0

        # Respect invariant for constant features: the original order of
        # element in features[:n_known_constants] must be preserved for sibling
        # and child nodes
        memcpy(features, constant_features, sizeof(SIZE_t) * n_known_constants)

        # Copy newly found constant features
        memcpy(constant_features + n_known_constants,
               features + n_known_constants,
               sizeof(SIZE_t) * n_found_constants)

        # Return values
        split[0] = best
        n_constant_features[0] = n_total_constants

    cdef void node_value(self, double* dest) nogil:
        """Copy the value of node samples[start:end] into dest."""
        self.criterion.node_value(dest)

    cdef double node_impurity(self) nogil:
        """Return the impurity of the current node.  Not at all clear what this should do for Scott's algorithm...
           TODO: Remove any dependencies on this...
        """
        return self.criterion.objective_improvement()




cdef class VarianceSplitter:
    """Splitter class for using variance split criterion used in fitting
       Wager & Athey double sample trees.
    Splitters are called by tree builders to find the best splits on both
    sparse and dense data, one split at a time.
    """
    def __reduce__(self):
        return (VarianceSplitter, (self.criterion,
                                   self.max_features,
                                   self.min_samples_leaf,
                                   self.min_weight_leaf,
                                   self.random_state,
                                   self.presort), self.__getstate__())

    def __cinit__(self, VarianceCriterion criterion, SIZE_t max_features,
                  SIZE_t min_samples_leaf, double min_weight_leaf,
                  object random_state, bint presort):
        """
        Parameters
        ----------
        criterion: Criterion
            The criterion to measure the quality of a split.
        max_features: SIZE_t
            The maximal number of randomly selected features which can be
            considered for a split.
        min_samples_leaf: SIZE_t
            The minimal number of samples each leaf can have, where splits
            which would result in having less samples in a leaf are not
            considered.
        min_weight_leaf: double
            The minimal weight each leaf can have, where the weight is the sum
            of the weights of each sample in it.
        random_state: object
            The user inputted random state to be used for pseudo-randomness
        """

        self.criterion = criterion

        self.samples = NULL
        self.n_samples = 0
        self.n_treated = 0
        self.n_control = 0
        self.features = NULL
        self.n_features = 0
        self.feature_values = NULL
        self.split_indices = NULL

        self.y = NULL
        self.y_stride = 0
        self.w = NULL
        self.w_stride = 0
        self.sample_weight = NULL
        self.X_sample_stride = 0
        self.X_feature_stride = 0

        self.sum_tau = 0
        self.sum_tau_sq = 0
        self.variance_tau = 0

        self.max_features = max_features
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_leaf = min_weight_leaf
        self.random_state = random_state
        self.presort = presort

    def __dealloc__(self):
        """Destructor."""

        free(self.samples)
        free(self.features)
        free(self.constant_features)
        free(self.feature_values)

    def __getstate__(self):
        return {}

    def __setstate__(self, d):
        pass

    cdef void init(self,
                   object X,
                   np.ndarray[DOUBLE_t, ndim=2, mode="c"] y,
                   np.ndarray[DOUBLE_t, ndim=2, mode="c"] w,
                   DOUBLE_t* sample_weight,
                   SIZE_t* split_indices,
                   np.ndarray X_idx_sorted=None) except *:
        """Initialize the splitter.
        Take in the input data X, the outcomes Y, the treatment indicators w,
        and optional sample weights.
        Parameters
        ----------
        X: object
            This contains the inputs. Usually it is a 2d numpy array.
        y: numpy.ndarray, dtype=DOUBLE_t
            This is the vector of outcomes for the samples
        w: numpy.ndarray, dtype=DOUBLE_t
            This is the vector of treatment indicators for the samples

        sample_weight: numpy.ndarray, dtype=DOUBLE_t (optional)
            The weights of the samples, where higher weighted samples are fit
            closer than lower weight samples. If not provided, all samples
            are assumed to have uniform weight.
        """

        self.rand_r_state = self.random_state.randint(0, RAND_R_MAX)
        cdef SIZE_t n_samples = X.shape[0]
        self.split_indices = split_indices

        # Create a new array which will be used to store nonzero
        # samples from the feature of interest
        cdef SIZE_t* samples = safe_realloc(&self.samples, n_samples)

        cdef SIZE_t i, j
        cdef double weighted_n_samples = 0.0
        j = 0

        for i in range(n_samples):
            # Only work with positively weighted samples
            if sample_weight == NULL or sample_weight[i] != 0.0:
                samples[j] = i
                j += 1

            if sample_weight != NULL:
                weighted_n_samples += sample_weight[i]
            else:
                weighted_n_samples += 1.0

        # Number of samples is number of positively weighted samples
        self.n_samples = j
        self.weighted_n_samples = weighted_n_samples

        cdef SIZE_t n_features = X.shape[1]
        cdef SIZE_t* features = safe_realloc(&self.features, n_features)

        for i in range(n_features):
            features[i] = i

        self.n_features = n_features

        safe_realloc(&self.feature_values, n_samples)
        safe_realloc(&self.constant_features, n_features)

        self.y = <DOUBLE_t*> y.data
        self.y_stride = <SIZE_t> y.strides[0] / <SIZE_t> y.itemsize
        self.w = <DOUBLE_t*> w.data
        self.w_stride = <SIZE_t> w.strides[0] / <SIZE_t> w.itemsize

        self.sample_weight = sample_weight

        # Initialize initial tau, tau_sq, sum_tau, sum_tau_sq, and variance_tau
        cdef DOUBLE_t sum_y_treated = 0
        cdef DOUBLE_t sum_y_control = 0
        self.n_treated = 0
        self.n_control = 0
        for i in range(n_samples) :
            if self.split_indices[i] == 1 :
                if self.w[i] == 0 :
                    sum_y_control += self.y[i]
                    self.n_control += 1
                else :
                    sum_y_treated += self.y[i]
                    self.n_treated += 1

        # Override self.n_samples here; we only want to count samples used to estimate splits right now...
        self.n_samples = self.n_control + self.n_treated

        # Calculate initial sum_tau, sum_tau_sq, and variance_tau...
        cdef DOUBLE_t init_tau = (sum_y_treated / self.n_treated) - (sum_y_control / self.n_control)
        self.sum_tau = (self.n_treated + self.n_control) * init_tau
        self.sum_tau_sq = (self.n_treated + self.n_control) * init_tau*init_tau
        self.variance_tau = (self.sum_tau_sq / (self.n_treated + self.n_control)) - (init_tau * init_tau)

        # Initialize X
        cdef np.ndarray X_ndarray = X

        self.X = <DTYPE_t*> X_ndarray.data
        self.X_sample_stride = <SIZE_t> X.strides[0] / <SIZE_t> X.itemsize
        self.X_feature_stride = <SIZE_t> X.strides[1] / <SIZE_t> X.itemsize

        if self.presort == 1:
            self.X_idx_sorted = X_idx_sorted
            self.X_idx_sorted_ptr = <INT32_t*> self.X_idx_sorted.data
            self.X_idx_sorted_stride = (<SIZE_t> self.X_idx_sorted.strides[1] /
                                        <SIZE_t> self.X_idx_sorted.itemsize)

            self.n_total_samples = X.shape[0]
            safe_realloc(&self.sample_mask, self.n_total_samples)
            memset(self.sample_mask, 0, self.n_total_samples*sizeof(SIZE_t))


    cdef void node_reset(self, SIZE_t start, SIZE_t end,
                         double* weighted_n_node_samples) nogil:
        """Reset splitter on node samples[start:end].
        Parameters
        ----------
        start: SIZE_t
            The index of the first sample to consider
        end: SIZE_t
            The index of the last sample to consider
        weighted_n_node_samples: numpy.ndarray, dtype=double pointer
            The total weight of those samples
        """
        self.start = start
        self.end = end
        self.criterion.init(self.y,
                            self.y_stride,
                            self.w,
                            self.w_stride,
                            self.sample_weight,
                            self.weighted_n_samples,
                            self.samples,
                            start,
                            end,
                            self.split_indices)
        weighted_n_node_samples[0] = self.criterion.weighted_n_node_samples

    cdef INT32_t min_class_count(self, SIZE_t start, SIZE_t end) nogil:
        """Count the min number of samples in the range [start:end[,
           with respect to the treatment w (assume binary tx).  Note -
           we only count samples used for effect estimation, i.e.,
           with self.split_indices == 0 """
        cdef INT32_t count0 = 0
        cdef INT32_t count1 = 0
        cdef SIZE_t i
        for i in range(start, end):
            if self.split_indices[ self.samples[i] ] == 0 :
                if (self.w[self.samples[i]] == 0):
                    count0 += 1
                else:
                    count1 += 1
        if count0 < count1:
            return count0
        else:
            return count1

    cdef void node_split(self, SplitRecord* split,
                         SIZE_t* n_constant_features) nogil:
        """Find the best split on node samples[start:end]."""
        # Find the best split
        cdef SIZE_t* samples = self.samples
        cdef SIZE_t start = self.start
        cdef SIZE_t end = self.end

        cdef SIZE_t* features = self.features
        cdef SIZE_t* constant_features = self.constant_features
        cdef SIZE_t n_features = self.n_features

        cdef DTYPE_t* X = self.X
        cdef DTYPE_t* Xf = self.feature_values
        cdef SIZE_t X_sample_stride = self.X_sample_stride
        cdef SIZE_t X_feature_stride = self.X_feature_stride
        cdef SIZE_t max_features = self.max_features
        cdef SIZE_t min_samples_leaf = self.min_samples_leaf
        cdef double min_weight_leaf = self.min_weight_leaf
        cdef UINT32_t* random_state = &self.rand_r_state

        cdef INT32_t* X_idx_sorted = self.X_idx_sorted_ptr
        cdef SIZE_t* sample_mask = self.sample_mask

        cdef SplitRecord best, current
        cdef double current_objective_improvement = -INFINITY
        cdef double best_objective_improvement = -INFINITY
        cdef double current_sum_tau = -INFINITY
        cdef double current_sum_tau_sq = -INFINITY
        cdef double best_sum_tau = -INFINITY
        cdef double best_sum_tau_sq = -INFINITY
        cdef double mean_tau

        cdef SIZE_t f_i = n_features
        cdef SIZE_t f_j
        cdef SIZE_t tmp
        cdef SIZE_t p
        cdef SIZE_t feature_idx_offset
        cdef SIZE_t feature_offset
        cdef SIZE_t i
        cdef SIZE_t j

        cdef SIZE_t n_visited_features = 0
        # Number of features discovered to be constant during the split search
        cdef SIZE_t n_found_constants = 0
        # Number of features known to be constant and drawn without replacement
        cdef SIZE_t n_drawn_constants = 0
        cdef SIZE_t n_known_constants = n_constant_features[0]
        # n_total_constants = n_known_constants + n_found_constants
        cdef SIZE_t n_total_constants = n_known_constants
        cdef DTYPE_t current_feature_value
        cdef SIZE_t partition_end
        cdef INT32_t min_num_in_class

        _init_split(&best, end)

        if self.presort == 1:
            for p in range(start, end):
                sample_mask[samples[p]] = 1  # sample_mask indicates what samples
                # to use; uses canonical ordering, not samples[] ordering.

        # Sample up to max_features without replacement using a
        # Fisher-Yates-based algorithm (using the local variables `f_i` and
        # `f_j` to compute a permutation of the `features` array).
        #
        # Skip the CPU intensive evaluation of the impurity criterion for
        # features that were already detected as constant (hence not suitable
        # for good splitting) by ancestor nodes and save the information on
        # newly discovered constant features to spare computation on descendant
        # nodes.

        # Iterate over some features...
        while (f_i > n_total_constants and  # Stop early if remaining features
                                            # are constant
                (n_visited_features < max_features or
                 # At least one drawn features must be non constant
                 n_visited_features <= n_found_constants + n_drawn_constants)):

            n_visited_features += 1

            # Loop invariant: elements of features in
            # - [:n_drawn_constant[ holds drawn and known constant features;
            # - [n_drawn_constant:n_known_constant[ holds known constant
            #   features that haven't been drawn yet;
            # - [n_known_constant:n_total_constant[ holds newly found constant
            #   features;
            # - [n_total_constant:f_i[ holds features that haven't been drawn
            #   yet and aren't constant apriori.
            # - [f_i:n_features[ holds features that have been drawn
            #   and aren't constant.

            # Draw a feature at random
            f_j = rand_int(n_drawn_constants, f_i - n_found_constants,
                           random_state)
            if f_j < n_known_constants:
                # f_j in the interval [n_drawn_constants, n_known_constants[
                tmp = features[f_j]
                features[f_j] = features[n_drawn_constants]
                features[n_drawn_constants] = tmp

                n_drawn_constants += 1

            else:
                # f_j in the interval [n_known_constants, f_i - n_found_constants[
                f_j += n_found_constants
                # f_j in the interval [n_total_constants, f_i[
                current.feature = features[f_j]
                feature_offset = self.X_feature_stride * current.feature

                # Sort samples along that feature; either by utilizing
                # presorting, or by copying the values into an array and
                # sorting the array in a manner which utilizes the cache more
                # effectively.
                if self.presort == 1:
                    p = start
                    feature_idx_offset = self.X_idx_sorted_stride * current.feature

                    for i in range(self.n_total_samples):
                        j = X_idx_sorted[i + feature_idx_offset]
                        if sample_mask[j] == 1:
                            samples[p] = j
                            Xf[p] = X[self.X_sample_stride * j + feature_offset]
                            p += 1
                else:
                    for i in range(start, end):
                        Xf[i] = X[self.X_sample_stride * samples[i] + feature_offset]
                    sort(Xf + start, samples + start, end - start)

                if Xf[end - 1] <= Xf[start] + FEATURE_THRESHOLD:
                    features[f_j] = features[n_total_constants]
                    features[n_total_constants] = current.feature

                    n_found_constants += 1
                    n_total_constants += 1

                else:
                    f_i -= 1
                    features[f_i], features[f_j] = features[f_j], features[f_i]

                    # Evaluate all splits; this is largely a hack to decrease computation
                    # by doing min necessary work to evaluate each split in turn.  It all
                    # relies on having a sorted vector of feature values so we can march along
                    # it in order to evaluate different split points...
                    self.criterion.reset()
                    p = start

                    # We have a sorted vector of feature values; here we iterate through them, checking
                    # each as a potential split point.
                    while p < end:
                        # It is possible to have a run of identical feature values...
                        while (p + 1 < end and
                               Xf[p + 1] <= Xf[p] + FEATURE_THRESHOLD):
                            p += 1

                        # (p + 1 >= end) or (X[samples[p + 1], current.feature] >
                        #                    X[samples[p], current.feature])
                        p += 1
                        # (p >= end) or (X[samples[p], current.feature] >
                        #                X[samples[p - 1], current.feature])

                        if p < end:
                            current.pos = p

                            # Reject if min_samples_leaf is not guaranteed
                            # if (((current.pos - start) < min_samples_leaf) or
                            #         ((end - current.pos) < min_samples_leaf)):
                            #     continue
                            min_num_in_class = self.min_class_count(start, current.pos)
                            if min_num_in_class <= min_samples_leaf :
                                continue
                            min_num_in_class = self.min_class_count(current.pos, end)
                            if min_num_in_class <= min_samples_leaf :
                                continue

                            # Calculate improvement using current.pos for threshold.
                            # This calculates sufficient statistics (sum of outcomes per
                            # treatment group, plus sum of squared outcomes per treatment
                            # group, along with cardinality) in each of the children, using
                            # current.pos as the split point.
                            self.criterion.update(current.pos)

                            # Reject if min_weight_leaf is not satisfied
                            # if ((self.criterion.weighted_n_left < min_weight_leaf) or
                            #         (self.criterion.weighted_n_right < min_weight_leaf)):
                            #     continue
                            current_sum_tau = self.sum_tau
                            current_sum_tau_sq = self.sum_tau_sq
                            current_objective_improvement = self.criterion.objective_improvement(self.variance_tau,
                                                                                                 &current_sum_tau,
                                                                                                 &current_sum_tau_sq,
                                                                                                 self.n_samples)
                            if current_objective_improvement <= MIN_OBJECTIVE_IMPROVEMENT :
                                continue

                            if current_objective_improvement > best_objective_improvement:
                                best_objective_improvement = current_objective_improvement
                                best_sum_tau = current_sum_tau
                                best_sum_tau_sq = current_sum_tau_sq
                                current.threshold = (Xf[p - 1] + Xf[p]) / 2.0
                                current.improvement = current_objective_improvement
                                if current.threshold == Xf[p]:
                                    current.threshold = Xf[p - 1]
                                best = current  # copy

        # Reorganize into samples[start:best.pos] + samples[best.pos:end]
        # Only do this if we found a good split...
        if best.pos < end:
            # Update variance_tau, sum_tau, and sum_tau_sq
            mean_tau = best_sum_tau / self.n_samples
            self.variance_tau = (best_sum_tau_sq / self.n_samples) - (mean_tau*mean_tau)
            self.sum_tau = best_sum_tau
            self.sum_tau_sq = best_sum_tau_sq

            # Reorganize samples...
            feature_offset = X_feature_stride * best.feature
            partition_end = end
            p = start

            while p < partition_end:
                if X[X_sample_stride * samples[p] + feature_offset] <= best.threshold:
                    p += 1

                else:
                    partition_end -= 1

                    tmp = samples[partition_end]
                    samples[partition_end] = samples[p]
                    samples[p] = tmp

        # Reset sample mask
        if self.presort == 1:
            for p in range(start, end):
                sample_mask[samples[p]] = 0

        # Respect invariant for constant features: the original order of
        # element in features[:n_known_constants] must be preserved for sibling
        # and child nodes
        memcpy(features, constant_features, sizeof(SIZE_t) * n_known_constants)

        # Copy newly found constant features
        memcpy(constant_features + n_known_constants,
               features + n_known_constants,
               sizeof(SIZE_t) * n_found_constants)

        # Return values
        split[0] = best
        n_constant_features[0] = n_total_constants

    cdef void node_value(self, double* dest) nogil:
        """Copy the value of node samples[start:end] into dest."""
        self.criterion.node_value(dest)

    cdef double node_impurity(self) nogil:
        """Return the impurity of the current node.  Not at all clear what this should do for Scott's algorithm...
           TODO: Remove any dependencies on this...
        """
        #return self.criterion.objective_improvement()
        return 0