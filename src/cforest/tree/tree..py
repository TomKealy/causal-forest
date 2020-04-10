from sklearn.base import ClassifierMixin
from sklearn.tree import BaseDecisionTree


class PropensityTree(BaseDecisionTree, ClassifierMixin):
    """A propensity tree classifier.  This is a base model for Wager & Athey's
    causal forests algorithm 2.
    Read more in the :ref:`User Guide <tree>`.
    Parameters
    ----------
    criterion : string, optional (default="gini")
        The function to measure the quality of a split. Supported criteria are
        "gini" for the Gini impurity and "entropy" for the information gain.
    splitter : string, optional (default="best")
        The strategy used to choose the split at each node. Supported
        strategies are "best" to choose the best split and "random" to choose
        the best random split.
    max_features : int, float, string or None, optional (default=None)
        The number of features to consider when looking for the best split:
            - If int, then consider `max_features` features at each split.
            - If float, then `max_features` is a percentage and
              `int(max_features * n_features)` features are considered at each
              split.
            - If "auto", then `max_features=sqrt(n_features)`.
            - If "sqrt", then `max_features=sqrt(n_features)`.
            - If "log2", then `max_features=log2(n_features)`.
            - If None, then `max_features=n_features`.
        Note: the search for a split does not stop until at least one
        valid partition of the node samples is found, even if it requires to
        effectively inspect more than ``max_features`` features.
    max_depth : int or None, optional (default=None)
        The maximum depth of the tree. If None, then nodes are expanded until
        all leaves are pure or until all leaves contain less than
        min_samples_split samples.
        Ignored if ``max_leaf_nodes`` is not None.
    min_samples_split : int, float, optional (default=2)
        The minimum number of samples required to split an internal node:
        - If int, then consider `min_samples_split` as the minimum number.
        - If float, then `min_samples_split` is a percentage and
          `ceil(min_samples_split * n_samples)` are the minimum
          number of samples for each split.
    min_samples_leaf : int, float, optional (default=1)
        The minimum number of samples of each treatment class required to
        be at a leaf node:
        - If int, then consider `min_samples_leaf` as the minimum number.
        - If float, then `min_samples_leaf` is a percentage and
          `ceil(min_samples_leaf * n_samples)` are the minimum
          number of samples for each node.
    min_weight_fraction_leaf : float, optional (default=0.)
        The minimum weighted fraction of the input samples required to be at a
        leaf node.
    max_leaf_nodes : int or None, optional (default=None)
        Grow a tree with ``max_leaf_nodes`` in best-first fashion.
        Best nodes are defined as relative reduction in impurity.
        If None then unlimited number of leaf nodes.
        If not None then ``max_depth`` will be ignored.
    class_weight : dict, list of dicts, "balanced" or None, optional (default=None)
        Weights associated with classes in the form ``{class_label: weight}``.
        If not given, all classes are supposed to have weight one. For
        multi-output problems, a list of dicts can be provided in the same
        order as the columns of y.
        The "balanced" mode uses the values of y to automatically adjust
        weights inversely proportional to class frequencies in the input data
        as ``n_samples / (n_classes * np.bincount(y))``
        For multi-output, the weights of each column of y will be multiplied.
        Note that these weights will be multiplied with sample_weight (passed
        through the fit method) if sample_weight is specified.
    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.
    presort : bool, optional (default=False)
        Whether to presort the data to speed up the finding of best splits in
        fitting. For the default settings of a decision tree on large
        datasets, setting this to true may slow down the training process.
        When using either a smaller dataset or a restricted depth, this may
        speed up the training.
    Attributes
    ----------
    classes_ : array of shape = [n_classes] or a list of such arrays
        The classes labels (single output problem),
        or a list of arrays of class labels (multi-output problem).
    feature_importances_ : array of shape = [n_features]
        The feature importances. The higher, the more important the
        feature. The importance of a feature is computed as the (normalized)
        total reduction of the criterion brought by that feature.  It is also
        known as the Gini importance [4]_.
    max_features_ : int,
        The inferred value of max_features.
    n_classes_ : int or list
        The number of classes (for single output problems),
        or a list containing the number of classes for each
        output (for multi-output problems).
    n_features_ : int
        The number of features when ``fit`` is performed.
    n_outputs_ : int
        The number of outputs when ``fit`` is performed.
    tree_ : Tree object
        The underlying Tree object.
    See also
    --------
    References
    ----------
    .. [1] http://en.wikipedia.org/wiki/Decision_tree_learning
    .. [2] L. Breiman, J. Friedman, R. Olshen, and C. Stone, "Classification
           and Regression Trees", Wadsworth, Belmont, CA, 1984.
    .. [3] T. Hastie, R. Tibshirani and J. Friedman. "Elements of Statistical
           Learning", Springer, 2009.
    .. [4] L. Breiman, and A. Cutler, "Random Forests",
           http://www.stat.berkeley.edu/~breiman/RandomForests/cc_home.htm
    Examples
    --------
    >>> from sklearn.datasets import load_iris
    >>> from sklearn.model_selection import cross_val_score
    >>> from sklearn.tree import DecisionTreeClassifier
    >>> clf = DecisionTreeClassifier(random_state=0)
    >>> iris = load_iris()
    >>> cross_val_score(clf, iris.data, iris.target, cv=10)
    ...                             # doctest: +SKIP
    ...
    array([ 1.     ,  0.93...,  0.86...,  0.93...,  0.93...,
            0.93...,  0.93...,  1.     ,  0.93...,  1.      ])
    """

    def __init__(self,
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=3,
                 min_weight_fraction_leaf=0.,
                 max_features=None,
                 random_state=None,
                 max_leaf_nodes=None,
                 class_weight=None,
                 presort=False):
        super(PropensityTree, self).__init__(
            criterion="propensity",
            splitter="propensity",
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            max_leaf_nodes=max_leaf_nodes,
            class_weight=class_weight,
            random_state=random_state,
            presort=presort)

    # NB - the fit method here overrides the fit method from the base class, BaseDecisionTree.
    def fit(self, X, y, w, sample_weight=None, check_input=True,
            X_idx_sorted=None):
        """Build a propensity tree from the training set (X, y, w).
        This is algorithm 2 from Wager & Athey.  We fit a tree that
        uses the treatment variable w to determine splits, but predict
        using the outcome y.
        Parameters
        ----------
        X : array-like or sparse matrix, shape = [n_samples, n_features]
            The training input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csc_matrix``.
        y : array-like, shape = [n_samples]
            Array of outcome indicators.
        w : array-lie, shape = [n_samples]
            Array of treatment indicators.
        sample_weight : array-like, shape = [n_samples] or None
            Sample weights. If None, then samples are equally weighted. Splits
            that would create child nodes with net zero or negative weight are
            ignored while searching for a split in each node. In the case of
            classification, splits are also ignored if they would result in any
            single class carrying a negative weight in either child node.
        check_input : boolean, (default=True)
            Allow to bypass several input checking.
            Don't use this parameter unless you know what you do.
        X_idx_sorted : array-like, shape = [n_samples, n_features], optional
            The indexes of the sorted training input samples. If many tree
            are grown on the same dataset, this allows the ordering to be
            cached between trees. If None, the data will be sorted here.
            Don't use this parameter unless you know what to do.
        Returns
        -------
        self : object
            Returns self.
        """

        is_classification = True

        random_state = check_random_state(self.random_state)
        if check_input:
            X = check_array(X, dtype=DTYPE, accept_sparse="csc")
            y = check_array(y, ensure_2d=False, dtype=None)
            w = check_array(w, ensure_2d=False, dtype=None)
            if issparse(X):
                X.sort_indices()

                if X.indices.dtype != np.intc or X.indptr.dtype != np.intc:
                    raise ValueError("No support for np.int64 index based "
                                     "sparse matrices")

        # Determine output settings
        n_samples, self.n_features_ = X.shape

        y = np.atleast_1d(y)
        w = np.atleast_1d(w)
        expanded_class_weight = None

        if y.ndim == 1:
            # reshape is necessary to preserve the data contiguity against vs
            # [:, np.newaxis] that does not.
            y = np.reshape(y, (-1, 1))
        if w.ndim == 1:
            w = np.reshape(w, (-1, 1))

        # We will hard assign b/c multi-class doesn't make sense in this context.
        self.n_outputs_ = 1

        self.classes_ = []
        self.n_classes_ = []

        if self.class_weight is not None:
            w_original = np.copy(w)

        w_encoded = np.zeros(w.shape, dtype=np.int)
        for k in range(self.n_outputs_):
            classes_k, w_encoded[:, k] = np.unique(w[:, k],
                                                   return_inverse=True)
            self.classes_.append(classes_k)
            self.n_classes_.append(classes_k.shape[0])
            w = w_encoded

            if self.class_weight is not None:
                expanded_class_weight = compute_sample_weight(
                    self.class_weight, w_original)

        self.n_classes_ = np.array(self.n_classes_, dtype=np.intp)

        if getattr(w, "dtype", None) != DOUBLE or not w.flags.contiguous:
            w = np.ascontiguousarray(w, dtype=DOUBLE)
        if getattr(y, "dtype", None) != DOUBLE or not y.flags.contiguous:
            y = np.ascontiguousarray(y, dtype=DOUBLE)

        # Check parameters
        max_depth = ((2 ** 31) - 1 if self.max_depth is None
                     else self.max_depth)
        max_leaf_nodes = (-1 if self.max_leaf_nodes is None
                          else self.max_leaf_nodes)

        if isinstance(self.min_samples_leaf, (numbers.Integral, np.integer)):
            min_samples_leaf = self.min_samples_leaf
        else:  # float
            min_samples_leaf = int(ceil(self.min_samples_leaf * n_samples))

        if isinstance(self.min_samples_split, (numbers.Integral, np.integer)):
            min_samples_split = self.min_samples_split
        else:  # float
            min_samples_split = int(ceil(self.min_samples_split * n_samples))
            min_samples_split = max(2, min_samples_split)

        min_samples_split = max(min_samples_split, 2 * min_samples_leaf)

        if isinstance(self.max_features, six.string_types):
            if self.max_features == "auto":
                if is_classification:
                    max_features = max(1, int(np.sqrt(self.n_features_)))
                else:
                    max_features = self.n_features_
            elif self.max_features == "sqrt":
                max_features = max(1, int(np.sqrt(self.n_features_)))
            elif self.max_features == "log2":
                max_features = max(1, int(np.log2(self.n_features_)))
            else:
                raise ValueError(
                    'Invalid value for max_features. Allowed string '
                    'values are "auto", "sqrt" or "log2".')
        elif self.max_features is None:
            max_features = self.n_features_
        elif isinstance(self.max_features, (numbers.Integral, np.integer)):
            max_features = self.max_features
        else:  # float
            if self.max_features > 0.0:
                max_features = max(1,
                                   int(self.max_features * self.n_features_))
            else:
                max_features = 0

        self.max_features_ = max_features

        if len(y) != n_samples:
            raise ValueError("Number of outcomees=%d does not match "
                             "number of samples=%d" % (len(y), n_samples))
        if len(w) != n_samples:
            raise ValueError("Number of treatments=%d does not match "
                             "number of samples=%d" % (len(w), n_samples))
        if not (0. < self.min_samples_split <= 1. or
                2 <= self.min_samples_split):
            raise ValueError("min_samples_split must be in at least 2"
                             " or in (0, 1], got %s" % min_samples_split)
        if not (0. < self.min_samples_leaf <= 0.5 or
                1 <= self.min_samples_leaf):
            raise ValueError("min_samples_leaf must be at least than 1 "
                             "or in (0, 0.5], got %s" % min_samples_leaf)

        if not 0 <= self.min_weight_fraction_leaf <= 0.5:
            raise ValueError("min_weight_fraction_leaf must in [0, 0.5]")
        if max_depth <= 0:
            raise ValueError("max_depth must be greater than zero. ")
        if not (0 < max_features <= self.n_features_):
            raise ValueError("max_features must be in (0, n_features]")
        if not isinstance(max_leaf_nodes, (numbers.Integral, np.integer)):
            raise ValueError("max_leaf_nodes must be integral number but was "
                             "%r" % max_leaf_nodes)
        if -1 < max_leaf_nodes < 2:
            raise ValueError(("max_leaf_nodes {0} must be either smaller than "
                              "0 or larger than 1").format(max_leaf_nodes))

        if sample_weight is not None:
            if (getattr(sample_weight, "dtype", None) != DOUBLE or
                    not sample_weight.flags.contiguous):
                sample_weight = np.ascontiguousarray(
                    sample_weight, dtype=DOUBLE)
            if len(sample_weight.shape) > 1:
                raise ValueError("Sample weights array has more "
                                 "than one dimension: %d" %
                                 len(sample_weight.shape))
            if len(sample_weight) != n_samples:
                raise ValueError("Number of weights=%d does not match "
                                 "number of samples=%d" %
                                 (len(sample_weight), n_samples))

        if expanded_class_weight is not None:
            if sample_weight is not None:
                sample_weight = sample_weight * expanded_class_weight
            else:
                sample_weight = expanded_class_weight

        # Set min_weight_leaf from min_weight_fraction_leaf
        if self.min_weight_fraction_leaf != 0. and sample_weight is not None:
            min_weight_leaf = (self.min_weight_fraction_leaf *
                               np.sum(sample_weight))
        else:
            min_weight_leaf = 0.

        presort = self.presort
        # Allow presort to be 'auto', which means True if the dataset is dense,
        # otherwise it will be False.
        if self.presort == 'auto' and issparse(X):
            presort = False
        elif self.presort == 'auto':
            presort = True

        if presort is True and issparse(X):
            raise ValueError("Presorting is not supported for sparse "
                             "matrices.")

        # If multiple trees are built on the same dataset, we only want to
        # presort once. Splitters now can accept presorted indices if desired,
        # but do not handle any presorting themselves. Ensemble algorithms
        # which desire presorting must do presorting themselves and pass that
        # matrix into each tree.
        if X_idx_sorted is None and presort:
            X_idx_sorted = np.asfortranarray(np.argsort(X, axis=0),
                                             dtype=np.int32)

        if presort and X_idx_sorted.shape != X.shape:
            raise ValueError("The shape of X (X.shape = {}) doesn't match "
                             "the shape of X_idx_sorted (X_idx_sorted"
                             ".shape = {})".format(X.shape,
                                                   X_idx_sorted.shape))

        # Build tree
        # criterion = PropensityCriterion(self.n_outputs_,
        #                                 self.n_classes_)
        criterion = Gini(self.n_outputs_, self.n_classes_)
        splitter = PropensitySplitter(criterion,
                                      self.max_features_,
                                      min_samples_leaf,
                                      min_weight_leaf,
                                      random_state,
                                      self.presort)

        self.tree_ = Tree(self.n_features_, self.n_classes_, self.n_outputs_)

        # Use version of DepthFirstTreeBuilder...
        builder = PropensityTreeBuilder(splitter,
                                        min_samples_split,
                                        min_samples_leaf,
                                        min_weight_leaf,
                                        max_depth)
        builder.build(self.tree_, X, w, sample_weight, X_idx_sorted)

        # Now apply to training data and get the leaf node assignments; keep them in an instance var.
        self.y = np.copy(y)
        self.w = np.copy(w)
        self.training_sample_nodes = self.tree_.apply(X)

        # How do we do this?  We want to average the outcome in each node, conditioned on w.
        # Then we just subtract...
        num_bins = np.max(self.training_sample_nodes) + 1
        self.treated_y = np.zeros(num_bins)
        self.control_y = np.zeros(num_bins)
        self.treated_n = np.zeros(num_bins)
        self.control_n = np.zeros(num_bins)
        for i in range(0, len(self.training_sample_nodes)):
            node_idx = self.training_sample_nodes[i]
            if (self.w[i] == 0):
                self.control_y[node_idx] += self.y[i]
                self.control_n[node_idx] += 1
            else:
                self.treated_y[node_idx] += self.y[i]
                self.treated_n[node_idx] += 1
        for i in range(num_bins):
            if self.control_n[i] == 0:
                self.control_n[i] = -1
            if self.treated_n[i] == 0:
                self.treated_n[i] = -1

        self.treated_mean_y = self.treated_y / self.treated_n
        self.control_mean_y = self.control_y / self.control_n
        self.effect_estimates = self.treated_mean_y - self.control_mean_y

        if self.n_outputs_ == 1:
            self.n_classes_ = self.n_classes_[0]
            self.classes_ = self.classes_[0]

        return self

    def predict_effect(self, X):
        """Predict the treatment effect for each sample in X.
        Returns
        -------
        y : array of shape = [n_samples]
        The estimated treatment effect for each sample in X.
        """
        nodes = self.apply(X)
        return np.take(self.effect_estimates, nodes)

    def predict_outcomes(self, X):
        """Predict the outcomes for treated and untreated for each sample in X;
           return as 2 column array."""
        nodes = self.apply(X)
        tx_outcomes = np.take(self.treated_mean_y, nodes)
        ctl_outcomes = np.take(self.control_mean_y, nodes)
        return np.transpose(np.matrix([tx_outcomes, ctl_outcomes]))

    # NB - the predict method here overrides the predict method from the base class, BaseDecisionTree.
    # Basically, we just use apply and get the (precomputed) tx effects for each of the leaf nodes.
    def predict(self, X, check_input=True):
        """Predict class or regression value for X.
        For a classification model, the predicted class for each sample in X is
        returned. For a regression model, the predicted value based on X is
        returned.
        Parameters
        ----------
        X : array-like or sparse matrix of shape = [n_samples, n_features]
            The input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csr_matrix``.
        check_input : boolean, (default=True)
            Allow to bypass several input checking.
            Don't use this parameter unless you know what you do.
        Returns
        -------
        y : array of shape = [n_samples] or [n_samples, n_outputs]
            The predicted classes, or the predict values.
        """

        X = self._validate_X_predict(X, check_input)
        proba = self.tree_.predict(X)
        n_samples = X.shape[0]

        # Classification
        if isinstance(self, ClassifierMixin):
            if self.n_outputs_ == 1:
                return self.classes_.take(np.argmax(proba, axis=1), axis=0)

            else:
                predictions = np.zeros((n_samples, self.n_outputs_))

                for k in range(self.n_outputs_):
                    predictions[:, k] = self.classes_[k].take(
                        np.argmax(proba[:, k], axis=1),
                        axis=0)

                return predictions

        # Regression
        else:
            if self.n_outputs_ == 1:
                return proba[:, 0]

            else:
                return proba[:, :, 0]

    # NB - the apply method here overrides the apply method from the base class, BaseDecisionTree.
    def apply(self, X, check_input=True):
        """
        Returns the index of the leaf that each sample is predicted as.
        .. versionadded:: 0.17
        Parameters
        ----------
        X : array_like or sparse matrix, shape = [n_samples, n_features]
            The input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csr_matrix``.
        check_input : boolean, (default=True)
            Allow to bypass several input checking.
            Don't use this parameter unless you know what you do.
        Returns
        -------
        X_leaves : array_like, shape = [n_samples,]
            For each datapoint x in X, return the index of the leaf x
            ends up in. Leaves are numbered within
            ``[0; self.tree_.node_count)``, possibly with gaps in the
            numbering.
        """
        X = self._validate_X_predict(X, check_input)
        return self.tree_.apply(X)

    def predict_proba(self, X, check_input=True):
        """Predict class probabilities of the input samples X.
        The predicted class probability is the fraction of samples of the same
        class in a leaf.
        check_input : boolean, (default=True)
            Allow to bypass several input checking.
            Don't use this parameter unless you know what you do.
        Parameters
        ----------
        X : array-like or sparse matrix of shape = [n_samples, n_features]
            The input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csr_matrix``.
        Returns
        -------
        p : array of shape = [n_samples, n_classes], or a list of n_outputs
            such arrays if n_outputs > 1.
            The class probabilities of the input samples. The order of the
            classes corresponds to that in the attribute `classes_`.
        """
        X = self._validate_X_predict(X, check_input)
        proba = self.tree_.predict(X)

        if self.n_outputs_ == 1:
            proba = proba[:, :self.n_classes_]
            normalizer = proba.sum(axis=1)[:, np.newaxis]
            normalizer[normalizer == 0.0] = 1.0
            proba /= normalizer

            return proba

        else:
            all_proba = []

            for k in range(self.n_outputs_):
                proba_k = proba[:, k, :self.n_classes_[k]]
                normalizer = proba_k.sum(axis=1)[:, np.newaxis]
                normalizer[normalizer == 0.0] = 1.0
                proba_k /= normalizer
                all_proba.append(proba_k)

            return all_proba

    def predict_log_proba(self, X):
        """Predict class log-probabilities of the input samples X.
        Parameters
        ----------
        X : array-like or sparse matrix of shape = [n_samples, n_features]
            The input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csr_matrix``.
        Returns
        -------
        p : array of shape = [n_samples, n_classes], or a list of n_outputs
            such arrays if n_outputs > 1.
            The class log-probabilities of the input samples. The order of the
            classes corresponds to that in the attribute `classes_`.
        """
        proba = self.predict_proba(X)

        if self.n_outputs_ == 1:
            return np.log(proba)

        else:
            for k in range(self.n_outputs_):
                proba[k] = np.log(proba[k])

            return proba


class DoubleSampleTree(BaseDecisionTree, ClassifierMixin):
    """A double sample tree.  This is a base model for Wager & Athey's
    causal forests algorithm 1 (double sample forests).
    Read more in the :ref:`User Guide <tree>`.
    Parameters
    ----------
    criterion : string, optional (default="gini")
        The function to measure the quality of a split. Supported criteria are
        "gini" for the Gini impurity and "entropy" for the information gain.
    splitter : string, optional (default="best")
        The strategy used to choose the split at each node. Supported
        strategies are "best" to choose the best split and "random" to choose
        the best random split.
    max_features : int, float, string or None, optional (default=None)
        The number of features to consider when looking for the best split:
            - If int, then consider `max_features` features at each split.
            - If float, then `max_features` is a percentage and
              `int(max_features * n_features)` features are considered at each
              split.
            - If "auto", then `max_features=sqrt(n_features)`.
            - If "sqrt", then `max_features=sqrt(n_features)`.
            - If "log2", then `max_features=log2(n_features)`.
            - If None, then `max_features=n_features`.
        Note: the search for a split does not stop until at least one
        valid partition of the node samples is found, even if it requires to
        effectively inspect more than ``max_features`` features.
    max_depth : int or None, optional (default=None)
        The maximum depth of the tree. If None, then nodes are expanded until
        all leaves are pure or until all leaves contain less than
        min_samples_split samples.
        Ignored if ``max_leaf_nodes`` is not None.
    min_samples_split : int, float, optional (default=2)
        The minimum number of samples required to split an internal node:
        - If int, then consider `min_samples_split` as the minimum number.
        - If float, then `min_samples_split` is a percentage and
          `ceil(min_samples_split * n_samples)` are the minimum
          number of samples for each split.
    min_samples_leaf : int, float, optional (default=1)
        The minimum number of samples of each treatment class required to
        be at a leaf node:
        - If int, then consider `min_samples_leaf` as the minimum number.
        - If float, then `min_samples_leaf` is a percentage and
          `ceil(min_samples_leaf * n_samples)` are the minimum
          number of samples for each node.
    min_weight_fraction_leaf : float, optional (default=0.)
        The minimum weighted fraction of the input samples required to be at a
        leaf node.
    max_leaf_nodes : int or None, optional (default=None)
        Grow a tree with ``max_leaf_nodes`` in best-first fashion.
        Best nodes are defined as relative reduction in impurity.
        If None then unlimited number of leaf nodes.
        If not None then ``max_depth`` will be ignored.
    class_weight : dict, list of dicts, "balanced" or None, optional (default=None)
        Weights associated with classes in the form ``{class_label: weight}``.
        If not given, all classes are supposed to have weight one. For
        multi-output problems, a list of dicts can be provided in the same
        order as the columns of y.
        The "balanced" mode uses the values of y to automatically adjust
        weights inversely proportional to class frequencies in the input data
        as ``n_samples / (n_classes * np.bincount(y))``
        For multi-output, the weights of each column of y will be multiplied.
        Note that these weights will be multiplied with sample_weight (passed
        through the fit method) if sample_weight is specified.
    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.
    presort : bool, optional (default=False)
        Whether to presort the data to speed up the finding of best splits in
        fitting. For the default settings of a decision tree on large
        datasets, setting this to true may slow down the training process.
        When using either a smaller dataset or a restricted depth, this may
        speed up the training.
    Attributes
    ----------
    classes_ : array of shape = [n_classes] or a list of such arrays
        The classes labels (single output problem),
        or a list of arrays of class labels (multi-output problem).
    feature_importances_ : array of shape = [n_features]
        The feature importances. The higher, the more important the
        feature. The importance of a feature is computed as the (normalized)
        total reduction of the criterion brought by that feature.  It is also
        known as the Gini importance [4]_.
    max_features_ : int,
        The inferred value of max_features.
    n_classes_ : int or list
        The number of classes (for single output problems),
        or a list containing the number of classes for each
        output (for multi-output problems).
    n_features_ : int
        The number of features when ``fit`` is performed.
    n_outputs_ : int
        The number of outputs when ``fit`` is performed.
    tree_ : Tree object
        The underlying Tree object.
    See also
    --------
    References
    ----------
    .. [1] http://en.wikipedia.org/wiki/Decision_tree_learning
    .. [2] L. Breiman, J. Friedman, R. Olshen, and C. Stone, "Classification
           and Regression Trees", Wadsworth, Belmont, CA, 1984.
    .. [3] T. Hastie, R. Tibshirani and J. Friedman. "Elements of Statistical
           Learning", Springer, 2009.
    .. [4] L. Breiman, and A. Cutler, "Random Forests",
           http://www.stat.berkeley.edu/~breiman/RandomForests/cc_home.htm
    Examples
    --------
    >>> from sklearn.datasets import load_iris
    >>> from sklearn.model_selection import cross_val_score
    >>> from sklearn.tree import DecisionTreeClassifier
    >>> clf = DecisionTreeClassifier(random_state=0)
    >>> iris = load_iris()
    >>> cross_val_score(clf, iris.data, iris.target, cv=10)
    ...                             # doctest: +SKIP
    ...
    array([ 1.     ,  0.93...,  0.86...,  0.93...,  0.93...,
            0.93...,  0.93...,  1.     ,  0.93...,  1.      ])
    """

    def __init__(self,
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=3,
                 min_weight_fraction_leaf=0.,
                 max_features=None,
                 random_state=None,
                 max_leaf_nodes=None,
                 class_weight=None,
                 presort=False):
        super(DoubleSampleTree, self).__init__(
            criterion="variance",
            splitter="variance",
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            max_leaf_nodes=max_leaf_nodes,
            class_weight=class_weight,
            random_state=random_state,
            presort=presort)

    # NB - the fit method here overrides the fit method from the base class, BaseDecisionTree.
    def fit(self, X, y, w, sample_weight=None, check_input=True, X_idx_sorted=None):
        """Build a double sample tree from the training set (X, y, w).
        This is algorithm 1 from Wager & Athey.  We fit a tree using
        half the data to find split points (using a variance maximizing
        criterion) and use the rest of the data to estimate treatment
        effects in the leaves.
        Parameters
        ----------
        X : array-like or sparse matrix, shape = [n_samples, n_features]
            The training input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csc_matrix``.
        y : array-like, shape = [n_samples]
            Array of outcome indicators.
        w : array-lie, shape = [n_samples]
            Array of treatment indicators.
        sample_weight : array-like, shape = [n_samples] or None
            Sample weights. If None, then samples are equally weighted. Splits
            that would create child nodes with net zero or negative weight are
            ignored while searching for a split in each node. In the case of
            classification, splits are also ignored if they would result in any
            single class carrying a negative weight in either child node.
        check_input : boolean, (default=True)
            Allow to bypass several input checking.
            Don't use this parameter unless you know what you do.
        X_idx_sorted : array-like, shape = [n_samples, n_features], optional
            The indexes of the sorted training input samples. If many tree
            are grown on the same dataset, this allows the ordering to be
            cached between trees. If None, the data will be sorted here.
            Don't use this parameter unless you know what to do.
        Returns
        -------
        self : object
            Returns self.
        """

        is_classification = True

        # Determine output settings
        n_samples, self.n_features_ = X.shape

        # self.split_indices is used to partition samples into two sets -
        # split_indices == 0 - used for estimation of treatment effects
        # split_indices == 1 - used to find splits.
        n_split = int(floor(n_samples / 2))
        self.split_indices = np.zeros(n_samples, dtype=np.int)
        self.split_indices[np.random.choice(n_samples, size=n_split, replace=False)] = 1
        # dtype is int64, which is int, and is cdef'ed to SIZE_t in most pxd files.

        n_samples, self.n_features_ = X.shape

        # And enforce this
        X_idx_sorted = None

        random_state = check_random_state(self.random_state)
        if check_input:
            X = check_array(X, dtype=DTYPE, accept_sparse="csc")
            y = check_array(y, ensure_2d=False, dtype=None)
            w = check_array(w, ensure_2d=False, dtype=None)
            if issparse(X):
                X.sort_indices()

                if X.indices.dtype != np.intc or X.indptr.dtype != np.intc:
                    raise ValueError("No support for np.int64 index based "
                                     "sparse matrices")

        y = np.atleast_1d(y)
        w = np.atleast_1d(w)
        expanded_class_weight = None

        if y.ndim == 1:
            # reshape is necessary to preserve the data contiguity against vs
            # [:, np.newaxis] that does not.
            y = np.reshape(y, (-1, 1))
        if w.ndim == 1:
            w = np.reshape(w, (-1, 1))

        # We will hard assign b/c multi-class doesn't make sense in this context.
        self.n_outputs_ = 1

        self.classes_ = []
        self.n_classes_ = []

        if self.class_weight is not None:
            w_original = np.copy(w)

        w_encoded = np.zeros(w.shape, dtype=np.int)
        for k in range(self.n_outputs_):
            classes_k, w_encoded[:, k] = np.unique(w[:, k],
                                                   return_inverse=True)
            self.classes_.append(classes_k)
            self.n_classes_.append(classes_k.shape[0])
            w = w_encoded

            if self.class_weight is not None:
                expanded_class_weight = compute_sample_weight(
                    self.class_weight, w_original)

        self.n_classes_ = np.array(self.n_classes_, dtype=np.intp)

        if getattr(w, "dtype", None) != DOUBLE or not w.flags.contiguous:
            w = np.ascontiguousarray(w, dtype=DOUBLE)
        if getattr(y, "dtype", None) != DOUBLE or not y.flags.contiguous:
            y = np.ascontiguousarray(y, dtype=DOUBLE)

        # Check parameters
        max_depth = ((2 ** 31) - 1 if self.max_depth is None
                     else self.max_depth)
        max_leaf_nodes = (-1 if self.max_leaf_nodes is None
                          else self.max_leaf_nodes)

        if isinstance(self.min_samples_leaf, (numbers.Integral, np.integer)):
            min_samples_leaf = self.min_samples_leaf
        else:  # float
            min_samples_leaf = int(ceil(self.min_samples_leaf * n_samples))

        if isinstance(self.min_samples_split, (numbers.Integral, np.integer)):
            min_samples_split = self.min_samples_split
        else:  # float
            min_samples_split = int(ceil(self.min_samples_split * n_samples))
            min_samples_split = max(2, min_samples_split)

        min_samples_split = max(min_samples_split, 2 * min_samples_leaf)

        if isinstance(self.max_features, six.string_types):
            if self.max_features == "auto":
                if is_classification:
                    max_features = max(1, int(np.sqrt(self.n_features_)))
                else:
                    max_features = self.n_features_
            elif self.max_features == "sqrt":
                max_features = max(1, int(np.sqrt(self.n_features_)))
            elif self.max_features == "log2":
                max_features = max(1, int(np.log2(self.n_features_)))
            else:
                raise ValueError(
                    'Invalid value for max_features. Allowed string '
                    'values are "auto", "sqrt" or "log2".')
        elif self.max_features is None:
            max_features = self.n_features_
        elif isinstance(self.max_features, (numbers.Integral, np.integer)):
            max_features = self.max_features
        else:  # float
            if self.max_features > 0.0:
                max_features = max(1,
                                   int(self.max_features * self.n_features_))
            else:
                max_features = 0

        self.max_features_ = max_features

        if len(y) != n_samples:
            raise ValueError("Number of outcomees=%d does not match "
                             "number of samples=%d" % (len(y), n_samples))
        if len(w) != n_samples:
            raise ValueError("Number of treatments=%d does not match "
                             "number of samples=%d" % (len(w), n_samples))
        if not (0. < self.min_samples_split <= 1. or
                2 <= self.min_samples_split):
            raise ValueError("min_samples_split must be in at least 2"
                             " or in (0, 1], got %s" % min_samples_split)
        if not (0. < self.min_samples_leaf <= 0.5 or
                1 <= self.min_samples_leaf):
            raise ValueError("min_samples_leaf must be at least than 1 "
                             "or in (0, 0.5], got %s" % min_samples_leaf)

        if not 0 <= self.min_weight_fraction_leaf <= 0.5:
            raise ValueError("min_weight_fraction_leaf must in [0, 0.5]")
        if max_depth <= 0:
            raise ValueError("max_depth must be greater than zero. ")
        if not (0 < max_features <= self.n_features_):
            raise ValueError("max_features must be in (0, n_features]")
        if not isinstance(max_leaf_nodes, (numbers.Integral, np.integer)):
            raise ValueError("max_leaf_nodes must be integral number but was "
                             "%r" % max_leaf_nodes)
        if -1 < max_leaf_nodes < 2:
            raise ValueError(("max_leaf_nodes {0} must be either smaller than "
                              "0 or larger than 1").format(max_leaf_nodes))

        if sample_weight is not None:
            if (getattr(sample_weight, "dtype", None) != DOUBLE or
                    not sample_weight.flags.contiguous):
                sample_weight = np.ascontiguousarray(
                    sample_weight, dtype=DOUBLE)
            if len(sample_weight.shape) > 1:
                raise ValueError("Sample weights array has more "
                                 "than one dimension: %d" %
                                 len(sample_weight.shape))
            # if len(sample_weight) != n_samples:
            #     raise ValueError("Number of weights=%d does not match "
            #                      "number of samples=%d" %
            #                      (len(sample_weight), n_samples))

        if expanded_class_weight is not None:
            if sample_weight is not None:
                sample_weight = sample_weight * expanded_class_weight
            else:
                sample_weight = expanded_class_weight

        # Set min_weight_leaf from min_weight_fraction_leaf
        if self.min_weight_fraction_leaf != 0. and sample_weight is not None:
            min_weight_leaf = (self.min_weight_fraction_leaf *
                               np.sum(sample_weight))
        else:
            min_weight_leaf = 0.

        presort = self.presort
        # Allow presort to be 'auto', which means True if the dataset is dense,
        # otherwise it will be False.
        if self.presort == 'auto' and issparse(X):
            presort = False
        elif self.presort == 'auto':
            presort = True

        if presort is True and issparse(X):
            raise ValueError("Presorting is not supported for sparse "
                             "matrices.")

        # If multiple trees are built on the same dataset, we only want to
        # presort once. Splitters now can accept presorted indices if desired,
        # but do not handle any presorting themselves. Ensemble algorithms
        # which desire presorting must do presorting themselves and pass that
        # matrix into each tree.
        if X_idx_sorted is None and presort:
            X_idx_sorted = np.asfortranarray(np.argsort(X, axis=0),
                                             dtype=np.int32)

        if presort and X_idx_sorted.shape != X.shape:
            raise ValueError("The shape of X (X.shape = {}) doesn't match "
                             "the shape of X_idx_sorted (X_idx_sorted"
                             ".shape = {})".format(X.shape,
                                                   X_idx_sorted.shape))

        # Build tree
        criterion = VarianceCriterion(self.n_outputs_,
                                      self.n_classes_)
        splitter = VarianceSplitter(criterion,
                                    self.max_features_,
                                    min_samples_leaf,
                                    min_weight_leaf,
                                    random_state,
                                    self.presort)

        self.tree_ = Tree(self.n_features_, self.n_classes_, self.n_outputs_)

        # Use version of DepthFirstTreeBuilder...
        builder = DoubleSampleTreeBuilder(splitter,
                                          min_samples_split,
                                          min_samples_leaf,
                                          min_weight_leaf,
                                          max_depth)
        builder.build(self.tree_, X, y, w, self.split_indices, sample_weight, X_idx_sorted)

        # Now apply to estimation data and get the leaf node assignments; keep them in an instance var.
        # The problem is - what if some leaf nodes don't have any samples from which to estimate the
        # treatment effect?  And what about the requirement for a min number of samples in each leaf node?
        # What do we do then!...
        self.training_sample_nodes = self.tree_.apply(X)

        # Calculate effect estimates on each.
        # How do we do this?  We want to average the outcome in each node, conditioned on w.
        # num_bins is number of nodes in the tree (not just leaf nodes).
        # We keep arrays treated_y, control_y, etc, that keep track of quantities in
        # each node, but only for leaf nodes in practice.
        num_bins = np.max(self.training_sample_nodes) + 1
        self.treated_y = np.zeros(num_bins)
        self.control_y = np.zeros(num_bins)
        self.treated_n = np.zeros(num_bins)
        self.control_n = np.zeros(num_bins)

        # For each sample, if it wasn't used to find the splits (split_indices[i] == 0)
        # then use it for effect estimation.
        for i in range(0, len(self.training_sample_nodes)):
            if self.split_indices[i] == 0:
                node_idx = self.training_sample_nodes[i]
                if (w[i] == 0):
                    self.control_y[node_idx] += y[i]
                    self.control_n[node_idx] += 1
                else:
                    self.treated_y[node_idx] += y[i]
                    self.treated_n[node_idx] += 1

        # Guard against divide by zero; should execute this...
        # presumably, all non-leaf nodes should have -1 as the value.
        for i in range(num_bins):
            if self.control_n[i] == 0:
                self.control_n[i] = -1
            if self.treated_n[i] == 0:
                self.treated_n[i] = -1

        # Calculate means in each leaf node, then subtract to get effect estimates.
        # note - this does not use any guards against 0 or 1 probabilities...
        self.treated_mean_y = self.treated_y / self.treated_n
        self.control_mean_y = self.control_y / self.control_n
        self.effect_estimates = self.treated_mean_y - self.control_mean_y

        if self.n_outputs_ == 1:
            self.n_classes_ = self.n_classes_[0]
            self.classes_ = self.classes_[0]

        return self

    def predict_effect(self, X):
        """Predict the treatment effect for each sample in X.
        Returns
        -------
        y : array of shape = [n_samples]
        The estimated treatment effect for each sample in X.
        """
        nodes = self.apply(X)
        return np.take(self.effect_estimates, nodes)

    def predict_outcomes(self, X):
        """Predict the outcomes for treated and untreated for each sample in X;
           return as 2 column array."""
        nodes = self.apply(X)
        tx_outcomes = np.take(self.treated_mean_y, nodes)
        ctl_outcomes = np.take(self.control_mean_y, nodes)
        return np.transpose(np.matrix([tx_outcomes, ctl_outcomes]))

    # NB - the predict method here overrides the predict method from the base class, BaseDecisionTree.
    # Basically, we just use apply and get the (precomputed) tx effects for each of the leaf nodes.
    def predict(self, X, check_input=True):
        """Predict class or regression value for X.
        For a classification model, the predicted class for each sample in X is
        returned. For a regression model, the predicted value based on X is
        returned.
        Parameters
        ----------
        X : array-like or sparse matrix of shape = [n_samples, n_features]
            The input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csr_matrix``.
        check_input : boolean, (default=True)
            Allow to bypass several input checking.
            Don't use this parameter unless you know what you do.
        Returns
        -------
        y : array of shape = [n_samples] or [n_samples, n_outputs]
            The predicted classes, or the predict values.
        """

        X = self._validate_X_predict(X, check_input)
        proba = self.tree_.predict(X)
        n_samples = X.shape[0]

        # Classification
        if isinstance(self, ClassifierMixin):
            if self.n_outputs_ == 1:
                return self.classes_.take(np.argmax(proba, axis=1), axis=0)

            else:
                predictions = np.zeros((n_samples, self.n_outputs_))

                for k in range(self.n_outputs_):
                    predictions[:, k] = self.classes_[k].take(
                        np.argmax(proba[:, k], axis=1),
                        axis=0)

                return predictions

        # Regression
        else:
            if self.n_outputs_ == 1:
                return proba[:, 0]

            else:
                return proba[:, :, 0]

    # NB - the apply method here overrides the apply method from the base class, BaseDecisionTree.
    def apply(self, X, check_input=True):
        """
        Returns the index of the leaf that each sample is predicted as.
        .. versionadded:: 0.17
        Parameters
        ----------
        X : array_like or sparse matrix, shape = [n_samples, n_features]
            The input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csr_matrix``.
        check_input : boolean, (default=True)
            Allow to bypass several input checking.
            Don't use this parameter unless you know what you do.
        Returns
        -------
        X_leaves : array_like, shape = [n_samples,]
            For each datapoint x in X, return the index of the leaf x
            ends up in. Leaves are numbered within
            ``[0; self.tree_.node_count)``, possibly with gaps in the
            numbering.
        """
        X = self._validate_X_predict(X, check_input)
        return self.tree_.apply(X)

    def predict_proba(self, X, check_input=True):
        """Predict class probabilities of the input samples X.
        The predicted class probability is the fraction of samples of the same
        class in a leaf.
        check_input : boolean, (default=True)
            Allow to bypass several input checking.
            Don't use this parameter unless you know what you do.
        Parameters
        ----------
        X : array-like or sparse matrix of shape = [n_samples, n_features]
            The input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csr_matrix``.
        Returns
        -------
        p : array of shape = [n_samples, n_classes], or a list of n_outputs
            such arrays if n_outputs > 1.
            The class probabilities of the input samples. The order of the
            classes corresponds to that in the attribute `classes_`.
        """
        X = self._validate_X_predict(X, check_input)
        proba = self.tree_.predict(X)

        if self.n_outputs_ == 1:
            proba = proba[:, :self.n_classes_]
            normalizer = proba.sum(axis=1)[:, np.newaxis]
            normalizer[normalizer == 0.0] = 1.0
            proba /= normalizer

            return proba

        else:
            all_proba = []

            for k in range(self.n_outputs_):
                proba_k = proba[:, k, :self.n_classes_[k]]
                normalizer = proba_k.sum(axis=1)[:, np.newaxis]
                normalizer[normalizer == 0.0] = 1.0
                proba_k /= normalizer
                all_proba.append(proba_k)

            return all_proba

    def predict_log_proba(self, X):
        """Predict class log-probabilities of the input samples X.
        Parameters
        ----------
        X : array-like or sparse matrix of shape = [n_samples, n_features]
            The input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csr_matrix``.
        Returns
        -------
        p : array of shape = [n_samples, n_classes], or a list of n_outputs
            such arrays if n_outputs > 1.
            The class log-probabilities of the input samples. The order of the
            classes corresponds to that in the attribute `classes_`.
        """
        proba = self.predict_proba(X)

        if self.n_outputs_ == 1:
            return np.log(proba)

        else:
            for k in range(self.n_outputs_):
                proba[k] = np.log(proba[k])

            return proba


class PowersTree(BaseDecisionTree, ClassifierMixin):
    """A tree for estimating causal effects using Scott Power's split criterion.
    Parameters
    ----------
    criterion : string, optional (default="gini")
        The function to measure the quality of a split. Supported criteria are
        "gini" for the Gini impurity and "entropy" for the information gain.
    splitter : string, optional (default="best")
        The strategy used to choose the split at each node. Supported
        strategies are "best" to choose the best split and "random" to choose
        the best random split.
    max_features : int, float, string or None, optional (default=None)
        The number of features to consider when looking for the best split:
            - If int, then consider `max_features` features at each split.
            - If float, then `max_features` is a percentage and
              `int(max_features * n_features)` features are considered at each
              split.
            - If "auto", then `max_features=sqrt(n_features)`.
            - If "sqrt", then `max_features=sqrt(n_features)`.
            - If "log2", then `max_features=log2(n_features)`.
            - If None, then `max_features=n_features`.
        Note: the search for a split does not stop until at least one
        valid partition of the node samples is found, even if it requires to
        effectively inspect more than ``max_features`` features.
    max_depth : int or None, optional (default=None)
        The maximum depth of the tree. If None, then nodes are expanded until
        all leaves are pure or until all leaves contain less than
        min_samples_split samples.
        Ignored if ``max_leaf_nodes`` is not None.
    min_samples_split : int, float, optional (default=2)
        The minimum number of samples required to split an internal node:
        - If int, then consider `min_samples_split` as the minimum number.
        - If float, then `min_samples_split` is a percentage and
          `ceil(min_samples_split * n_samples)` are the minimum
          number of samples for each split.
    min_samples_leaf : int, float, optional (default=1)
        The minimum number of samples of each treatment class required to
        be at a leaf node:
        - If int, then consider `min_samples_leaf` as the minimum number.
        - If float, then `min_samples_leaf` is a percentage and
          `ceil(min_samples_leaf * n_samples)` are the minimum
          number of samples for each node.
    min_weight_fraction_leaf : float, optional (default=0.)
        The minimum weighted fraction of the input samples required to be at a
        leaf node.
    max_leaf_nodes : int or None, optional (default=None)
        Grow a tree with ``max_leaf_nodes`` in best-first fashion.
        Best nodes are defined as relative reduction in impurity.
        If None then unlimited number of leaf nodes.
        If not None then ``max_depth`` will be ignored.
    class_weight : dict, list of dicts, "balanced" or None, optional (default=None)
        Weights associated with classes in the form ``{class_label: weight}``.
        If not given, all classes are supposed to have weight one. For
        multi-output problems, a list of dicts can be provided in the same
        order as the columns of y.
        The "balanced" mode uses the values of y to automatically adjust
        weights inversely proportional to class frequencies in the input data
        as ``n_samples / (n_classes * np.bincount(y))``
        For multi-output, the weights of each column of y will be multiplied.
        Note that these weights will be multiplied with sample_weight (passed
        through the fit method) if sample_weight is specified.
    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.
    presort : bool, optional (default=False)
        Whether to presort the data to speed up the finding of best splits in
        fitting. For the default settings of a decision tree on large
        datasets, setting this to true may slow down the training process.
        When using either a smaller dataset or a restricted depth, this may
        speed up the training.
    Attributes
    ----------
    classes_ : array of shape = [n_classes] or a list of such arrays
        The classes labels (single output problem),
        or a list of arrays of class labels (multi-output problem).
    feature_importances_ : array of shape = [n_features]
        The feature importances. The higher, the more important the
        feature. The importance of a feature is computed as the (normalized)
        total reduction of the criterion brought by that feature.  It is also
        known as the Gini importance [4]_.
    max_features_ : int,
        The inferred value of max_features.
    n_classes_ : int or list
        The number of classes (for single output problems),
        or a list containing the number of classes for each
        output (for multi-output problems).
    n_features_ : int
        The number of features when ``fit`` is performed.
    n_outputs_ : int
        The number of outputs when ``fit`` is performed.
    tree_ : Tree object
        The underlying Tree object.
    See also
    --------
    References
    ----------
    .. [1] http://en.wikipedia.org/wiki/Decision_tree_learning
    .. [2] L. Breiman, J. Friedman, R. Olshen, and C. Stone, "Classification
           and Regression Trees", Wadsworth, Belmont, CA, 1984.
    .. [3] T. Hastie, R. Tibshirani and J. Friedman. "Elements of Statistical
           Learning", Springer, 2009.
    .. [4] L. Breiman, and A. Cutler, "Random Forests",
           http://www.stat.berkeley.edu/~breiman/RandomForests/cc_home.htm
    Examples
    --------
    >>> from sklearn.datasets import load_iris
    >>> from sklearn.model_selection import cross_val_score
    >>> from sklearn.tree import DecisionTreeClassifier
    >>> clf = DecisionTreeClassifier(random_state=0)
    >>> iris = load_iris()
    >>> cross_val_score(clf, iris.data, iris.target, cv=10)
    ...                             # doctest: +SKIP
    ...
    array([ 1.     ,  0.93...,  0.86...,  0.93...,  0.93...,
            0.93...,  0.93...,  1.     ,  0.93...,  1.      ])
    """

    def __init__(self,
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=3,
                 min_weight_fraction_leaf=0.,
                 max_features=None,
                 random_state=None,
                 max_leaf_nodes=None,
                 class_weight=None,
                 presort=False):
        super(PowersTree, self).__init__(
            criterion="PowersCriterion",
            splitter="PowersSplitter",
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            max_leaf_nodes=max_leaf_nodes,
            class_weight=class_weight,
            random_state=random_state,
            presort=presort)

    # NB - the fit method here overrides the fit method from the base class, BaseDecisionTree.
    def fit(self, X, y, w, sample_weight=None, check_input=True,
            X_idx_sorted=None):
        """Build a causal tree from the training set (X, y, w).
           using Scott Powers' split criterion.
        Parameters
        ----------
        X : array-like or sparse matrix, shape = [n_samples, n_features]
            The training input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csc_matrix``.
        y : array-like, shape = [n_samples]
            Array of outcome indicators.
        w : array-lie, shape = [n_samples]
            Array of treatment indicators.
        sample_weight : array-like, shape = [n_samples] or None
            Sample weights. If None, then samples are equally weighted. Splits
            that would create child nodes with net zero or negative weight are
            ignored while searching for a split in each node. In the case of
            classification, splits are also ignored if they would result in any
            single class carrying a negative weight in either child node.
        check_input : boolean, (default=True)
            Allow to bypass several input checking.
            Don't use this parameter unless you know what you do.
        X_idx_sorted : array-like, shape = [n_samples, n_features], optional
            The indexes of the sorted training input samples. If many tree
            are grown on the same dataset, this allows the ordering to be
            cached between trees. If None, the data will be sorted here.
            Don't use this parameter unless you know what to do.
        Returns
        -------
        self : object
            Returns self.
        """

        is_classification = True

        random_state = check_random_state(self.random_state)
        if check_input:
            X = check_array(X, dtype=DTYPE, accept_sparse="csc")
            y = check_array(y, ensure_2d=False, dtype=None)
            w = check_array(w, ensure_2d=False, dtype=None)
            if issparse(X):
                X.sort_indices()

                if X.indices.dtype != np.intc or X.indptr.dtype != np.intc:
                    raise ValueError("No support for np.int64 index based "
                                     "sparse matrices")

        # Determine output settings
        n_samples, self.n_features_ = X.shape

        y = np.atleast_1d(y)
        w = np.atleast_1d(w)
        expanded_class_weight = None

        if y.ndim == 1:
            # reshape is necessary to preserve the data contiguity against vs
            # [:, np.newaxis] that does not.
            y = np.reshape(y, (-1, 1))
        if w.ndim == 1:
            w = np.reshape(w, (-1, 1))

        # We will hard assign b/c multi-class doesn't make sense in this context.
        self.n_outputs_ = 1

        self.classes_ = []
        self.n_classes_ = []

        if self.class_weight is not None:
            w_original = np.copy(w)

        w_encoded = np.zeros(w.shape, dtype=np.int)
        for k in range(self.n_outputs_):
            classes_k, w_encoded[:, k] = np.unique(w[:, k],
                                                   return_inverse=True)
            self.classes_.append(classes_k)
            self.n_classes_.append(classes_k.shape[0])
            w = w_encoded

            if self.class_weight is not None:
                expanded_class_weight = compute_sample_weight(
                    self.class_weight, w_original)

        self.n_classes_ = np.array(self.n_classes_, dtype=np.intp)

        if getattr(w, "dtype", None) != DOUBLE or not w.flags.contiguous:
            w = np.ascontiguousarray(w, dtype=DOUBLE)
        if getattr(y, "dtype", None) != DOUBLE or not y.flags.contiguous:
            y = np.ascontiguousarray(y, dtype=DOUBLE)

        # Check parameters
        max_depth = ((2 ** 31) - 1 if self.max_depth is None
                     else self.max_depth)
        max_leaf_nodes = (-1 if self.max_leaf_nodes is None
                          else self.max_leaf_nodes)

        if isinstance(self.min_samples_leaf, (numbers.Integral, np.integer)):
            min_samples_leaf = self.min_samples_leaf
        else:  # float
            min_samples_leaf = int(ceil(self.min_samples_leaf * n_samples))

        if isinstance(self.min_samples_split, (numbers.Integral, np.integer)):
            min_samples_split = self.min_samples_split
        else:  # float
            min_samples_split = int(ceil(self.min_samples_split * n_samples))
            min_samples_split = max(2, min_samples_split)

        min_samples_split = max(min_samples_split, 2 * min_samples_leaf)

        if isinstance(self.max_features, six.string_types):
            if self.max_features == "auto":
                if is_classification:
                    max_features = max(1, int(np.sqrt(self.n_features_)))
                else:
                    max_features = self.n_features_
            elif self.max_features == "sqrt":
                max_features = max(1, int(np.sqrt(self.n_features_)))
            elif self.max_features == "log2":
                max_features = max(1, int(np.log2(self.n_features_)))
            else:
                raise ValueError(
                    'Invalid value for max_features. Allowed string '
                    'values are "auto", "sqrt" or "log2".')
        elif self.max_features is None:
            max_features = self.n_features_
        elif isinstance(self.max_features, (numbers.Integral, np.integer)):
            max_features = self.max_features
        else:  # float
            if self.max_features > 0.0:
                max_features = max(1,
                                   int(self.max_features * self.n_features_))
            else:
                max_features = 0

        self.max_features_ = max_features

        if len(y) != n_samples:
            raise ValueError("Number of outcomees=%d does not match "
                             "number of samples=%d" % (len(y), n_samples))
        if len(w) != n_samples:
            raise ValueError("Number of treatments=%d does not match "
                             "number of samples=%d" % (len(w), n_samples))
        if not (0. < self.min_samples_split <= 1. or
                2 <= self.min_samples_split):
            raise ValueError("min_samples_split must be in at least 2"
                             " or in (0, 1], got %s" % min_samples_split)
        if not (0. < self.min_samples_leaf <= 0.5 or
                1 <= self.min_samples_leaf):
            raise ValueError("min_samples_leaf must be at least than 1 "
                             "or in (0, 0.5], got %s" % min_samples_leaf)

        if not 0 <= self.min_weight_fraction_leaf <= 0.5:
            raise ValueError("min_weight_fraction_leaf must in [0, 0.5]")
        if max_depth <= 0:
            raise ValueError("max_depth must be greater than zero. ")
        if not (0 < max_features <= self.n_features_):
            raise ValueError("max_features must be in (0, n_features]")
        if not isinstance(max_leaf_nodes, (numbers.Integral, np.integer)):
            raise ValueError("max_leaf_nodes must be integral number but was "
                             "%r" % max_leaf_nodes)
        if -1 < max_leaf_nodes < 2:
            raise ValueError(("max_leaf_nodes {0} must be either smaller than "
                              "0 or larger than 1").format(max_leaf_nodes))

        if sample_weight is not None:
            if (getattr(sample_weight, "dtype", None) != DOUBLE or
                    not sample_weight.flags.contiguous):
                sample_weight = np.ascontiguousarray(
                    sample_weight, dtype=DOUBLE)
            if len(sample_weight.shape) > 1:
                raise ValueError("Sample weights array has more "
                                 "than one dimension: %d" %
                                 len(sample_weight.shape))
            if len(sample_weight) != n_samples:
                raise ValueError("Number of weights=%d does not match "
                                 "number of samples=%d" %
                                 (len(sample_weight), n_samples))

        if expanded_class_weight is not None:
            if sample_weight is not None:
                sample_weight = sample_weight * expanded_class_weight
            else:
                sample_weight = expanded_class_weight

        # Set min_weight_leaf from min_weight_fraction_leaf
        if self.min_weight_fraction_leaf != 0. and sample_weight is not None:
            min_weight_leaf = (self.min_weight_fraction_leaf *
                               np.sum(sample_weight))
        else:
            min_weight_leaf = 0.

        presort = self.presort
        # Allow presort to be 'auto', which means True if the dataset is dense,
        # otherwise it will be False.
        if self.presort == 'auto' and issparse(X):
            presort = False
        elif self.presort == 'auto':
            presort = True

        if presort is True and issparse(X):
            raise ValueError("Presorting is not supported for sparse "
                             "matrices.")

        # If multiple trees are built on the same dataset, we only want to
        # presort once. Splitters now can accept presorted indices if desired,
        # but do not handle any presorting themselves. Ensemble algorithms
        # which desire presorting must do presorting themselves and pass that
        # matrix into each tree.
        if X_idx_sorted is None and presort:
            X_idx_sorted = np.asfortranarray(np.argsort(X, axis=0),
                                             dtype=np.int32)

        if presort and X_idx_sorted.shape != X.shape:
            raise ValueError("The shape of X (X.shape = {}) doesn't match "
                             "the shape of X_idx_sorted (X_idx_sorted"
                             ".shape = {})".format(X.shape,
                                                   X_idx_sorted.shape))

        # Build tree
        criterion = PowersCriterion()
        if np.unique(y).shape[0] == 2:
            criterion.set_binary_outcome(1)
        else:
            criterion.set_binary_outcome(0)
        splitter = PowersSplitter(criterion,
                                  self.max_features_,
                                  min_samples_leaf,
                                  min_weight_leaf,
                                  random_state,
                                  self.presort)

        self.tree_ = Tree(self.n_features_, self.n_classes_, self.n_outputs_)

        # Use version of DepthFirstTreeBuilder...
        builder = PowersTreeBuilder(splitter,
                                    min_samples_split,
                                    min_samples_leaf,
                                    min_weight_leaf,
                                    max_depth)
        builder.build(self.tree_, X, y, w, sample_weight, X_idx_sorted)

        # Now apply to training data and get the leaf node assignments; keep them in an instance var.
        self.y = np.copy(y)
        self.w = np.copy(w)
        self.training_sample_nodes = self.tree_.apply(X)

        # TODO: Calculate effect estimates on each.
        # How do we do this?  We want to average the outcome in each node, conditioned on w.
        # Then we just subtract...
        num_bins = np.max(self.training_sample_nodes) + 1
        self.treated_y = np.zeros(num_bins)
        self.control_y = np.zeros(num_bins)
        self.treated_n = np.zeros(num_bins)
        self.control_n = np.zeros(num_bins)
        for i in range(0, len(self.training_sample_nodes)):
            node_idx = self.training_sample_nodes[i]
            if (self.w[i] == 0):
                self.control_y[node_idx] += self.y[i]
                self.control_n[node_idx] += 1
            else:
                self.treated_y[node_idx] += self.y[i]
                self.treated_n[node_idx] += 1

        # Note - should never have -1 counts because of the min number constraint...
        for i in range(num_bins):
            if self.control_n[i] == 0:
                self.control_n[i] = -1
            if self.treated_n[i] == 0:
                self.treated_n[i] = -1

        self.treated_mean_y = self.treated_y / self.treated_n
        self.control_mean_y = self.control_y / self.control_n
        self.effect_estimates = self.treated_mean_y - self.control_mean_y

        if self.n_outputs_ == 1:
            self.n_classes_ = self.n_classes_[0]
            self.classes_ = self.classes_[0]

        return self

    def predict_effect(self, X):
        """Predict the treatment effect for each sample in X.
        Returns
        -------
        y : array of shape = [n_samples]
        The estimated treatment effect for each sample in X.
        """
        nodes = self.apply(X)
        return np.take(self.effect_estimates, nodes)

    def predict_outcomes(self, X):
        """Predict the outcomes for treated and untreated for each sample in X;
           return as 2 column array."""
        nodes = self.apply(X)
        tx_outcomes = np.take(self.treated_mean_y, nodes)
        ctl_outcomes = np.take(self.control_mean_y, nodes)
        return np.transpose(np.matrix([tx_outcomes, ctl_outcomes]))

    # NB - the predict method here overrides the predict method from the base class, BaseDecisionTree.
    # Basically, we just use apply and get the (precomputed) tx effects for each of the leaf nodes.
    def predict(self, X, check_input=True):
        """Predict class or regression value for X.
        For a classification model, the predicted class for each sample in X is
        returned. For a regression model, the predicted value based on X is
        returned.
        Parameters
        ----------
        X : array-like or sparse matrix of shape = [n_samples, n_features]
            The input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csr_matrix``.
        check_input : boolean, (default=True)
            Allow to bypass several input checking.
            Don't use this parameter unless you know what you do.
        Returns
        -------
        y : array of shape = [n_samples] or [n_samples, n_outputs]
            The predicted classes, or the predict values.
        """

        X = self._validate_X_predict(X, check_input)
        proba = self.tree_.predict(X)
        n_samples = X.shape[0]

        # Classification
        if isinstance(self, ClassifierMixin):
            if self.n_outputs_ == 1:
                return self.classes_.take(np.argmax(proba, axis=1), axis=0)

            else:
                predictions = np.zeros((n_samples, self.n_outputs_))

                for k in range(self.n_outputs_):
                    predictions[:, k] = self.classes_[k].take(
                        np.argmax(proba[:, k], axis=1),
                        axis=0)

                return predictions

        # Regression
        else:
            if self.n_outputs_ == 1:
                return proba[:, 0]

            else:
                return proba[:, :, 0]

    # NB - the apply method here overrides the apply method from the base class, BaseDecisionTree.
    def apply(self, X, check_input=True):
        """
        Returns the index of the leaf that each sample is predicted as.
        .. versionadded:: 0.17
        Parameters
        ----------
        X : array_like or sparse matrix, shape = [n_samples, n_features]
            The input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csr_matrix``.
        check_input : boolean, (default=True)
            Allow to bypass several input checking.
            Don't use this parameter unless you know what you do.
        Returns
        -------
        X_leaves : array_like, shape = [n_samples,]
            For each datapoint x in X, return the index of the leaf x
            ends up in. Leaves are numbered within
            ``[0; self.tree_.node_count)``, possibly with gaps in the
            numbering.
        """
        X = self._validate_X_predict(X, check_input)
        return self.tree_.apply(X)

    def predict_proba(self, X, check_input=True):
        """Predict class probabilities of the input samples X.
        The predicted class probability is the fraction of samples of the same
        class in a leaf.
        check_input : boolean, (default=True)
            Allow to bypass several input checking.
            Don't use this parameter unless you know what you do.
        Parameters
        ----------
        X : array-like or sparse matrix of shape = [n_samples, n_features]
            The input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csr_matrix``.
        Returns
        -------
        p : array of shape = [n_samples, n_classes], or a list of n_outputs
            such arrays if n_outputs > 1.
            The class probabilities of the input samples. The order of the
            classes corresponds to that in the attribute `classes_`.
        """
        X = self._validate_X_predict(X, check_input)
        proba = self.tree_.predict(X)

        if self.n_outputs_ == 1:
            proba = proba[:, :self.n_classes_]
            normalizer = proba.sum(axis=1)[:, np.newaxis]
            normalizer[normalizer == 0.0] = 1.0
            proba /= normalizer

            return proba

        else:
            all_proba = []

            for k in range(self.n_outputs_):
                proba_k = proba[:, k, :self.n_classes_[k]]
                normalizer = proba_k.sum(axis=1)[:, np.newaxis]
                normalizer[normalizer == 0.0] = 1.0
                proba_k /= normalizer
                all_proba.append(proba_k)

            return all_proba

    def predict_log_proba(self, X):
        """Predict class log-probabilities of the input samples X.
        Parameters
        ----------
        X : array-like or sparse matrix of shape = [n_samples, n_features]
            The input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csr_matrix``.
        Returns
        -------
        p : array of shape = [n_samples, n_classes], or a list of n_outputs
            such arrays if n_outputs > 1.
            The class log-probabilities of the input samples. The order of the
            classes corresponds to that in the attribute `classes_`.
        """
        proba = self.predict_proba(X)

        if self.n_outputs_ == 1:
            return np.log(proba)

        else:
            for k in range(self.n_outputs_):
                proba[k] = np.log(proba[k])

            return proba