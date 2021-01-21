#! /usr/bin/env python
# -*- coding:utf-8 -*-


import numpy as np
import sklearn

from finsurf_forest import ForestClassifier, ForestRegressor, _generate_unsampled_indices, _generate_sample_indices
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier, _tree
from distutils.version import LooseVersion
import sklearn.utils

import sklearn.pipeline

import itertools as itt


# Calculation of feature importance :
# - the one from the original package (linear model, pred_score = SUM(Fc) + bias
# - the one from Epifanio et al 2017: "intervention in decision"


# NOTE: on joint_contribution
# In joint contribution : a unique "feature set" (corresponding to the begining
# of the path of multiple leaves) can be stored in a list, mapping to its
# associated feature contributions.
# Then for a given leave, we just need to report the indices of all the feature
# sets that are encountered in the path to this leave.
# This is a faster and more memory-aware approach than calculating for each
# leaf its associated feature-sets mapped to feature contributions.
#
#
# eg: leaves 1 and 3 share the nodes A, B, C (then D is specific to leaf 1,
# while E is specific to leaf 3.) ; we can thus calculate the feature
# contributions for feature sets {A,}, {A,B}, {A,B,C}, and store them in a list
# ; in parallel, a dictionary map leaves to the list of feature-sets they are
# composed of.


# NOTE: on feature intervention (I. Epifanio, 2017)
# For a given path associated to a leave, 4 nodes are encountered:
# with features F1, F3, F4, F1. (total features = 5).
# This means that in this path, the number of times they are used are:
# F1:2 ; F2:0 ; F3:1 ; F4:1 ; F5:0
# Division by the number of nodes in the path gives the "feature intervention:
# F1:0.5 ; F2:0 ; F3:0.25 ; F4:0.25 ; F5:0


def _get_tree_paths(tree, node_id, depth=0):
    """ Returns all paths through the tree as list of node_ids

    Recursive function to go through each possible path in a tree and
    accumulate the node ids of a current <node_id> into a list.

    Starting with node_id=0 allows to explore all paths going from the root
    node to the leaves.

    In:
        tree : sklearn tree model (decision tree regressor or classifier)
        node_id (int): 

    """
    if node_id == _tree.TREE_LEAF:
        raise ValueError("Invalid node_id %s" % _tree.TREE_LEAF)

    left_child = tree.children_left[node_id]
    right_child = tree.children_right[node_id]

    if left_child != _tree.TREE_LEAF:
        left_paths = _get_tree_paths(tree, left_child, depth=depth + 1)
        right_paths = _get_tree_paths(tree, right_child, depth=depth + 1)

        for path in left_paths:
            path.append(node_id)

        for path in right_paths:
            path.append(node_id)
        paths = left_paths + right_paths
    else:
        paths = [[node_id]]
    return paths


class Interpreter(object):
    """
    """
    # PART : INIT
    # -----------
    def __init__(self,
                 model,
                 X=None,
                 y=None,
                 train_dataset=False,
                 joint_contribution=False,
                 ):

        if train_dataset == True and y is None:
            raise ValueError("y is needed when working on the train_dataset.")

        self.y = y
        self.joint_contribution=joint_contribution
        self.train_dataset = train_dataset

        # NOTE: might not be needed if working with a tree only.
        self._set_full_training_indices = None
        self._sampled_idx_per_tree = None
        self._unsampled_idx_per_tree = None
        self.model_FC = None

        # Prepare the dataset and get the prediction model from a potential
        # pipeline.
        if isinstance(model, sklearn.pipeline.Pipeline):
            self.preprocessing_steps = sklearn.pipeline.Pipeline(model.steps[:-1])
            self.model = model.steps[-1][1]

            sklearn.utils.validation.check_is_fitted(self.model, "feature_importances_")

            if X is not None:
                self.set_X(self.preprocessing_steps.transform(X))
            else:
                self.X = None

        else:
            self.model = model

            sklearn.utils.validation.check_is_fitted(self.model, "feature_importances_")

            self.preprocessing_steps = None

            if X is not None:
                self.set_X(X)
            else:
                self.X = None

        if self.model.n_outputs_ > 1:
            raise ValueError("Multilabel classification trees not supported.")


    def set_X(self, X):
        """ Cast X to np.array ; check if len(y) == X.shape[0] if train_dataset
        """
        if self.train_dataset:
            print("CAUTION: make sure you are indeed inputing the train dataset.")
            if not len(self.y)==X.shape[0]:
                raise ValueError("Wrong shape for X when compared against y.")

        if not isinstance(X, np.ndarray):
            # Likely that the pipeline returns a dataframe.
            self.X = X.values

        else:
            self.X = X

    def _generate_sample_indices(self, i, random_state):
        if i>0 and i%50==0:
            print("\tTree {} - get sampled idx".format(i))

        return _generate_sample_indices(random_state, self.y)

    def _subtract_sets_tolist(self, i, subtract_set):
        if i>0 and i%50==0:
            print("\tTree {} - get unsampled idx".format(i))

        return sorted(self._full_training_indices_set - set(subtract_set))

    def _generate_unsampled_idx(self):
        """ Create list of unsampled indices per tree for a rdf model.

        During training of a RandomForest model, each tree is grown on a
        bootstrap of the full X matrix ; by default the same number is taken,
        resulting in 2/3 of unique sampled elements.

        Here are retrieved the 1/3 unseen elements, for each of the tree.

        The procedure is as follow:
        - gather for each tree the set of training indices
        - merge into the full set of training indices.
        - then for each tree, define the unsampled indices as 
          fullSet - sampledIndices_t

        """
        print("Generating unsampled indices per tree..")
        if self.y is None:
            raise ValueError("y is None ; cannot generate sample idx without it.")

        # Merge into a single set of unique sampled indices.
        full_training_indices = np.unique(
                            list(itt.chain(*sampled_idx_per_tree)))

        self._sampled_idx_per_tree = sampled_idx_per_tree

        self._full_training_indices_set = set(full_training_indices)


        unsampled_idx_per_tree = []
        for i, sampled_idx in enumerate(sampled_idx_per_tree):
            # Old version : was just getting the unsampled indices => samples
            # that were not used during training of the forest were included.
            # New version : get from the full list the indices that were not
            # used in the tree.

            #unsamp_idx = self._generate_unsampled_indices(tree.random_state, self.y)
            unsamp_idx = self._full_training_indices_set - set(sampled_idx)
            unsampled_idx_per_tree.append(sorted(list(unsamp_idx)))

        self._unsampled_idx_per_tree = unsampled_idx_per_tree 

        return

    def _generate_unsampled_idx(self):
        """ Create list of unsampled indices per tree for a rdf model.

        During training of a RandomForest model, each tree is grown on a
        bootstrap of the full X matrix ; by default the same number is taken,
        resulting in 2/3 of unique sampled elements.

        Here are retrieved the 1/3 unseen elements, for each of the tree.

        The procedure is as follow:
        - gather for each tree the set of training indices
        - merge into the full set of training indices.
        - then for each tree, define the unsampled indices as 
          fullSet - sampledIndices_t

        """
        print("Generating unsampled indices per tree..")
        if self.y is None:
            raise ValueError("y is None ; cannot generate sample idx without it.")

        print("\tGenerate sampled indices")

        sampled_idx_per_tree = []
        for i, tree in enumerate(self.model.estimators_):
            sampled_idx = self._generate_sample_indices(i, tree.random_state)

            sampled_idx_per_tree.append(sampled_idx)

        print("\n")


        self._sampled_idx_per_tree = sampled_idx_per_tree
        
        print("\tCreate full set of sampled indices")
        # Merge into a single set of unique sampled indices.
        full_training_indices = np.unique(list(itt.chain(*sampled_idx_per_tree))) 

        self._full_training_indices_set = set(full_training_indices)

        # And now generate the unsampled indices per tree, by subtracting for
        # each tree its sampled idx from the full set.

        # Old version : was just getting the unsampled indices => samples
        # that were not used during training of the forest were included.
        # New version : get from the full list the indices that were not
        # used in the tree.
        print("\tGenerate unsampled indices")
        unsampled_idx_per_tree = []
        for i, sampled_idx in enumerate(sampled_idx_per_tree):
            #unsamp_idx = self._generate_unsampled_indices(tree.random_state, self.y)
            unsamp_idx = self._full_training_indices_set - set(sampled_idx)
            unsampled_idx_per_tree.append(sorted(list(unsamp_idx)))


        self._unsampled_idx_per_tree = unsampled_idx_per_tree
        return


    # PART : FC struct
    # ----------------

    def build_FC(self):
        """ Get the feature contribution structures from the model.

        These structures correspond to:
        - predictions per leaf
        - bias at root per path
        - feature contribution vector per path

        Each path being associated to a leaf, in a given tree.

        In a fully grown tree, the number of leaf is at maximum the number of
        samples used during training.
        """
        if (isinstance(self.model, DecisionTreeClassifier) or
            isinstance(self.model, DecisionTreeRegressor)):
            res = self._get_feature_contribs_tree(self.model)
            self.model_FC = [res] # Store as a single-element list. 
            return
            

        elif (isinstance(self.model, ForestClassifier) or
              isinstance(self.model, ForestRegressor)):

            res = self._get_feature_contribs_forest()
            self.model_FC = res 
            return 

        else:
            raise ValueError("Wrong model type. Base learner needs to be a "
                             "DecisionTreeClassifier or DecisionTreeRegressor.")


    def _get_feature_contribs_tree(self, tree_model, i=0):
        """ Create the FC structures from all paths in the tree_model

        Calculate only once the feature contributions in each path of the
        tree_model, by exploring each path and calculating for each feature the
        impurity reduction between the parent node where it is used and the
        child nodes, for each of the target values.

        For a RegressionTree, the predicted value at a leaf is equal to the
        initial value in the root + the sum of all differences between child
        value and parent value (representing the feature contribution of the
        feature used at the node.)

        For a DecisionTree, the predicted value at a leaf is a vector of
        probabilities for each target class ; a given probability is the
        initial proportion of samples in this class in the root + the sum of
        all 

        Feature Contributions can be calculated either in a joint fashion, or
        independently:
          * if joint contribution : for a path, a set of iteratively combined
          features will be created by going deeper into the tree ; for each
          combination (of size N_joint_feat = depth_node), the feature
          contribution corresponds to the difference between the child node and
          the current node.
          * else : for each path in the tree, the feature at node_i has its
          contribution calculated from the difference between the child node and
          the current node.

        """
        if i>0 and i%50==0:
            print("Feature contrib - tree {}".format(i))

        # FIRST : CREATE VARIABLES
        # ========================

        # Get for each node the feature index.
        feature_index = list(tree_model.tree_.feature)

        paths = _get_tree_paths(tree_model.tree_, 0)
        for path in paths:
            path.reverse()
        
        # Map leaves to paths
        leaf_to_path = {path[-1]:path for path in paths}

        # Get class-counts from all nodes ; remove the single-dimensional inner arrays
        values = tree_model.tree_.value.squeeze()

        # reshape if squeezed into a single float
        if len(values.shape) == 0:
            values = np.array([values])
            # Unclear what situation might lead here. Kept from the original # lib.
            raise ValueError("No node in tree??")

        # Cast into python list, accessing values will be faster
        values_list = list(values)

        # Leaves predictions
        direct_pred = {leaf:values[leaf] for leaf in leaf_to_path.keys()} 

        # Get Biases and shapes of feature contrib vector.
        # line_shape is used to create the empty matrix of FC.
        if isinstance(tree_model, DecisionTreeRegressor):
            biases = values[paths[0][0]]
            line_shape = len(tree_model.feature_importances_)

        elif isinstance(tree_model, DecisionTreeClassifier):
            # scikit stores category counts, we turn them into proportions
            normalizer = values.sum(axis=1)[:, np.newaxis]
            normalizer[normalizer == 0.0] = 1.0
            values /= normalizer

            biases = values[paths[0][0]]

            line_shape = (len(tree_model.feature_importances_),
                          tree_model.n_classes_)

        # NEXT : CREATE FEATURE CONTRIBS
        # ==============================

        path_contribs = {} 
        for path in paths:
            leaf = path[-1]

            if self.joint_contribution:
                # Feature contribution is calculated at each node as the
                # sum of differences in proportions per class between child node
                # and current node (where a feature is used), and associated to
                # the set of features encountered so far in the path.
                path_features = set()

                path_contribs[leaf] = {}

                for i in range(len(path) - 1):
                    path_features.add(feature_index[path[i]])

                    # Here the key is represented as the sorted set of features
                    # in the path so far. NOTE : maybe keep unsorted?
                    key_pathfeats = tuple(sorted(path_features))
                    contrib = values_list[path[i+1]] - values_list[path[i]]

                    path_contribs[leaf][key_pathfeats] = \
                            path_contribs[leaf].get(key_pathfeats, 0) + contrib

            else:
                # Feature contribution is calculated for each feature as the
                # sum of differences in proportions per class between child node
                # and parent node (where a feature is used).
                contribs = np.zeros(line_shape)

                for i in range(len(path) - 1):
                    contrib = values_list[path[i+1]] - values_list[path[i]]
                    contribs[feature_index[path[i]]] += contrib

                path_contribs[leaf] = contribs

        # FINALLY : RETURN STRUCTURE
        # ==========================

        if self.joint_contribution:
            #raise NotImplementedError("Not implemented yet...")
            return direct_pred, biases, path_contribs

        else:
            # Here : merge into a (N_feat + 2) x N_class matrix.
            # (where N_class = 1 when doing regression).
            full_fc_mat = {}
            for leaf in path_contribs.keys():
                full_fc_mat[leaf] = np.concatenate((
                                        np.stack((direct_pred[leaf],
                                                  biases)),
                                        path_contribs[leaf]
                                        ))

            # Can be merged into a single dataframe, adding a column for leaf
            # index. 
            #pd.concat([pd.DataFrame(v).T.assign(index=k).set_index('index')
            #            for k, v in full_fc_mat.items()]
            #         ).reset_index()
            return full_fc_mat 


    def _get_feature_contribs_forest(self):
        """ Create FC vectors from paths in each tree of the forest.
        
        No particular thing to do here, beside call the
        _get_feature_contribs_tree function on each of the estimators.

        """
        

        if self.joint_contribution:
            raise NotImplementedError("Not implemented yet...")

        else:
            args = ((tree, i) for i, tree in enumerate(self.model.estimators_))
            # This creates a list of dictionaries, one for each tree,
            # mapping leaves to feature contribution tables.
            results = [self._get_feature_contribs_tree(*v) for v in args]
            print("\tDone.")

        return results


    # PART : FC ON SAMPLES
    # --------------------

    def _predict_forest(self):
        """
            If you need to sum the matrices with np.nan (rather than
            modify in place):
            res = np.where(np.isnan(a+b), np.where(np.isnan(a), b, a), a+b)
        """
        if self.joint_contribution:
            raise NotImplementedError("Not implemented yet...")

        else:

            for tree_idx, tree in enumerate(self.model.estimators_):
                if self.train_dataset:
                    sample_idx_tree = self._unsampled_idx_per_tree[tree_idx]

                else:
                    sample_idx_tree = np.arange(self.X.shape[0])

                # This function will modify the "self.finale_FC_mat" in place.
                self._predict_tree(tree, tree_idx, sample_idx_tree)


            self.finale_FC_mat = self.finale_FC_mat / self.count_trees_FC[:,np.newaxis,np.newaxis]

        return


    def _predict_tree(self, model, tree_idx=0, sample_indices=None):
        """ Get the feature contributions from a tree applied on self.X

        Feature Contributions associated to the model should have already been
        generated using the self.build_FC() function ; this created a list with
        a dict of feature contribution matrix per tree (so single element list
        for a tree model, and multiple elemnets for a forest model).

        The model is used to assign samples from self.X to the tree <model> leaves.
        These leaves then can be used to map each sample to its feature
        contribution matrix. A new multi-dimensional matrix is created out of
        these (of shape N_sample x N_feat x N_class)

        The sample_indices are then used to add the newly created FC values to
        the `self.finale_FC_mat` in place.


        In:
            self
            model: DecisionTree trained model
            tree_idx (int): used to retrieve the `self.model_FC` dict.
            sample_indices (list, default=None): sample indices for the self.X
                matrix, to specifically retrieve their FC.

        """
        if tree_idx>0 and tree_idx%50==0:
            print("\tTree {} - get predictions on X".format(tree_idx))

        # Get leaves assignement for this tree.
        tmp_leaves = model.apply(self.X[sample_indices])

        if self.joint_contribution:
            raise NotImplementedError("Not yet implemented...")

        else:
            # Retrieve the FC matrix for each sample's assigned leaf.
            tree_FC = np.array([self.model_FC[tree_idx][l] for l in tmp_leaves])
            # And now update the full table.
            finale_FC_mat_idx = [self._sample_idx_map[i] for i in sample_indices]
            #self.lock.acquire()
            self.finale_FC_mat[finale_FC_mat_idx] += tree_FC 
            # Need to track the nb of trees evaluating each sample
            self.count_trees_FC[finale_FC_mat_idx] += 1
            #self.lock.release()

            # Here you have a full matrix of feature contributions,
            # with all the train-samples, even those which are kept
            # with nan values as they were used for growing the tree.
            # NOTE : This may be exported for exploration.
            #tree_FC_full = self.prototype_FC_mat.copy()
            #tree_FC_full[finale_FC_mat_idx] = tree_FC
        return


    def _initialize_finale_FC_matrix(self, model_type):
        """ Initialize the different instance variables for FC predictions.

        Initialized variables are (for joint_contribution=False):
        - self.prototype_FC_mat : not used yet ; a N_samp x N_feat x N_class
          matrix that may be used to export the retrieved FC values *per tree*

        - self.finale_FC_mat : the finale matrix of N_samp x N_feat x N_class
          feature contribution values, either for a single DecisionTree model,
          or averaged over all trees of a Forest model

        - self.count_trees_FC : used to average the Feature Contribution values
          in a forest model.


        """
        if not (model_type in ('tree','forest')):
            raise ValueError('<model_type> "{}" not recognized'.format(model_type))
        
        if model_type=='tree':
            model_tmp = self.model

        else:
            model_tmp = self.model.estimators_[0]


        N_class = model_tmp.n_classes_
        # Here : add 2 because the prediction and the bias are stored
        # in the same matrix.
        N_feat = len(model_tmp.feature_importances_)+2

        if self.joint_contribution:
            raise NotImplementedError("Not implemented yet...")

        else:
            if model_type=='tree':
                N_samp = self.X.shape[0]
            
            else:
                if self.train_dataset:
                    if self._unsampled_idx_per_tree is None:
                        self._generate_unsampled_idx()

                    N_samp = len(self._full_training_indices_set)
                    # Create a mapping of the X indices to the set of training
                    # sample indices.
                    self._sample_idx_map = dict(zip(
                                            sorted(list(self._full_training_indices_set)),
                                            np.arange(N_samp)))
                else:
                    N_samp = self.X.shape[0]
                    self._sample_idx_map = dict(zip(np.arange(N_samp),
                                                    np.arange(N_samp)))

                # Here the counts are monitored to divide the sum of FC by the
                # correct nb of trees.
                self.count_trees_FC = np.zeros(N_samp)

            # Finale matrix with sum of FC per sample across all trees.
            self.finale_FC_mat = np.zeros((N_samp, N_feat, N_class))
            # prototype FC matrix, which can be used to export a table in the
            # `_predict_tree` function.
            self.prototype_FC_mat = np.full((N_samp, N_feat, N_class), np.nan)

        return 
        

    def predict(self):
        """ Main function to get feature contributions from X samples.

        NOTE: joint_contribution is not implemented yet. The structure of the
        feature contributions generated by self.build_FC() are not so easy to
        manipulate, and require more time.

        For not-joint contributions, the feature contribution matrix are
        generated for the leaves found in the model-tree(s).

        Different instance variables are then initialized, the most important
        being self.finale_FC_mat, which will contain the feature contribution
        values for all samples, calculated from the model.

        To fill this matrix, the model-tree(s) are "applied" on self.X to
        assign samples to the leaves, and then retrieve the precalculated
        feature contribution matrices of these leaves.
        """

        if self.X is None:
            raise ValueError("X is not set.")

        if self.joint_contribution:
            raise NotImplementedError("Not implemented yet...")
        
        if self.model_FC is None: 
            self.build_FC()

        if (isinstance(self.model, DecisionTreeClassifier) or
            isinstance(self.model, DecisionTreeRegressor)):
            # DecisionTrees do not perform bootstrap ; so train_dataset set to
            # True won't change any thing to the way the FC are calculated
            # compared to a regular X dataset. 

            self._initialize_finale_FC_matrix('tree')
            sample_idx_tree = np.arange(self.X.shape[0])
            # This function will modify the "self.finale_FC_mat" in place.
            print("Predict for tree model on samples in X.")
            return self._predict_tree(self.model, 0, sample_idx_tree)


        elif (isinstance(self.model, ForestClassifier) or
              isinstance(self.model, ForestRegressor)):

            self._initialize_finale_FC_matrix('forest')
            print("Predict for forest model on samples in X.")
            return self._predict_forest()

        else:
            raise ValueError("Wrong model type. Base learner needs to be a "
                             "DecisionTreeClassifier or DecisionTreeRegressor.")
