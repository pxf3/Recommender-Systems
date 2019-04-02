# Homework 4
# Yushuo Ruan
# INFO 5871, Spring 2019
# Robin Burke
# University of Colorado, Boulder

from lenskit.algorithms import Recommender
from lenskit.algorithms.basic import UnratedItemCandidateSelector
from collections import defaultdict

import numpy as np
import pandas as pd
import math


class NaiveBayesRecommender(Recommender):
    _count_tables = {}
    _item_features = None
    _nb_table = None
    _min_float = np.power(2.0, -149)

    def __init__(self, item_features=None, thresh=2.9, alpha=0.01, beta=0.01):
        self._item_features = item_features
        self.selector = UnratedItemCandidateSelector()
        self._nb_table = NaiveBayesTable(thresh, alpha, beta)

    # TODO: HOMEWORK 4
    def fit(self, ratings, *args, **kwargs):
        # Must fit the selector
        self.selector.fit(ratings)
        self._nb_table.reset()

        self._item_features.columns = ['item', 'feature']
        # For each rating
        for indexR, rowR in ratings.iterrows():
            user = rowR['user']
            item = rowR['item']
            rating = rowR['rating']
            # print("processing: ", user)
            # Get associated item features
            feature = self.get_features_list(item)
            self._nb_table.process_rating(user, rating, feature)


    # TODO: HOMEWORK 4
    # Should return ordered data frame with items and score
    def recommend(self, user, n=None, candidates=None, ratings=None):
        # n is None or zero, return DataFrame with an empty item column
        if n is None or n == 0:
            return pd.DataFrame({'item': []})

        if candidates is None:
            candidates = self.selector.candidates(user, ratings)

        # Initialize scores
        scores = []

        # for each candidate
        for candidate in candidates:
            scores.append(self.score_item(user, candidate))
            # Score the candidate for the user

            # Build list of candidate, score pairs


        # Turn result into data frame
        data = {'item': candidates, 'score': scores}
        df = pd.DataFrame(data, columns=['item', 'score'])

        # Retain n largest scoring rows (nlargest)
        df = df.nlargest(n, 'score')
        # Sort by score (sort_values)
        df = df.sort_values(by=['score'], ascending=False)

        # return data frame
        return df

    # TODO: HOMEWORK 4
    # Helper function to return a list of features for an item from features data frame
    def get_features_list(self, item):
        features_list = []
        for indexF, rowF in self._item_features.loc[self._item_features['item'] == item].iterrows():
            features_list.append(rowF['feature'])
        return features_list

    # TODO: HOMEWORK 4
    def score_item(self, user, item):
        # get the features
        features = self.get_features_list(item)
        # initialize the liked and nliked scores with the base probability
        baseP = self._nb_table.user_prob(user, True)
        baseNP = self._nb_table.user_prob(user, False)

        likeP = 1
        nlikeP = 1
        # for each feature
        for feature in features:
            likeP = likeP * self._nb_table.user_feature_prob(user, feature, True)
            nlikeP = nlikeP * self._nb_table.user_feature_prob(user, feature, False)
        # update scores by multiplying with conditional probability
        likeP = likeP * baseP
        nlikeP = nlikeP * baseNP

        try:
            ratio = likeP/nlikeP
        except ZeroDivisionError:
            # Handle the case when scores go to zero.
            return 0

        # Compute log-likelihood
        try:
            LL=math.log(ratio, math.e)
        except ValueError:
            # Handle zero again
            return 0
        # Return result
        return LL

    # DO NOT ALTER
    def get_params(self, deep=True):

        return {'item_features': self._item_features,
                'thresh': self._nb_table.thresh,
                'alpha': self._nb_table.alpha,
                'beta': self._nb_table.beta}

    # DO NOT ALTER
    def ensure_minimum_score(self, val):
        if val == 0.0:
            return self._min_float
        else:
            return val


# TODO: HOMEWORK 4
# Helper class
class NaiveBayesTable:
    liked_cond_table = {}
    nliked_cond_table = {}
    liked_table = {}
    nliked_table = {}
    thresh = 0
    alpha = 0.01
    beta = 0.01

    # TODO: HOMEWORK 4
    def __init__(self, thresh=2.9, alpha=0.01, beta=0.01):
        self.thresh = thresh
        self.alpha = alpha
        self.beta = beta
        return

    # TODO: HOMEWORK 4
    # Reset all the tables
    def reset(self):
        self.liked_cond_table = {}
        self.nliked_cond_table = {}
        self.liked_table = {}
        self.nliked_table = {}
        return

    # TODO: HOMEWORK 4
    # Return the count for a feature for a user (either liked or ~liked)
    # Should be robust if the user or the feature are not currently in table: return 0 in these cases
    def user_feature_count(self, user, feature, liked=True):
        # print(type(user))
        # print(self.liked_cond_table[user][feature])
        try:
            if liked:
                return self.liked_cond_table[user][feature]
            elif ~liked:
                return self.nliked_cond_table[user][feature]
        except KeyError:
            return 0

    # TODO: HOMEWORK 4
    # Sets the count for a feature for a user (either liked or ~liked)
    # Should be robust if the user or the feature are not currently in table. Create appropriate entry or entries
    def set_user_feature_count(self, user, feature, count, liked=True):
        if liked:
            try:
                self.liked_cond_table[user][feature] = count
            except KeyError:
                self.liked_cond_table[user] = {}
                self.liked_cond_table[user][feature] = count
        elif ~liked:
            try:
                self.nliked_cond_table[user][feature] = count
            except KeyError:
                self.nliked_cond_table[user] = {}
                self.nliked_cond_table[user][feature] = count
        return

    def incr_user_feature_count(self, user, feature, liked=True):
        val = self.user_feature_count(user, feature, liked)
        self.set_user_feature_count(user, feature, val + 1, liked)

    # TODO: HOMEWORK 4
    # Computes P(f|L) or P(f|~L) as the observed ratio of features and total likes/dislikes
    # Should include smooting with beta value
    def user_feature_prob(self, user, feature, liked=True):
        return (self.user_feature_count(user, feature, liked) + self.beta) / \
               (self.user_count(user, liked) + 2 * self.beta)


    # TODO: HOMEWORK 4
    # Return the liked (disiked) count for a user (
    # Should be robust if the user is not currently in table: return 0 in this cases
    def user_count(self, user, liked=True):
        try:
            if liked:
                return self.liked_table[user]
            elif ~liked:
                return self.nliked_table[user]
        except KeyError:
            return 0

    # TODO: HOMEWORK 4
    # Sets the liked/disliked count for a user
    # Should be robust if the user is not currently in table. Create appropriate entry
    def set_user_count(self, user, value, liked=True):
        if liked:
            self.liked_table[user] = value
        elif ~liked:
            self.nliked_table[user] = value
        return

    def incr_user_count(self, user, liked=True):
        val = self.user_count(user, liked)
        self.set_user_count(user, val + 1, liked)

    # TODO: HOMEWORK 4
    # Computes P(L) or P(~L) as the observed ratio of liked/dislike and total rated item count
    # Should include smooting with alpha value
    def user_prob(self, user, liked=True):
        #print(user, ", ", self.user_count(user, False))
        return (self.user_count(user, liked)+self.alpha) / \
               (self.user_count(user, True)+self.user_count(user, False)+2*self.alpha)

    # TODO:HOMEWORK 4
    # Update the table to take into account one new rating
    def process_rating(self, user, rating, features):
        # print(user, ", ", rating, ", ", features)
        # Determine if liked or disliked
        liked = False
        if rating > self.thresh:
            liked = True
        # Increment appropriate count for the user
        self.incr_user_count(user, liked)
        #print(user, ",", rating, ",", liked)
        # For each feature
        for feature in features:
            # Increment appropriate feature count for the user
            self.incr_user_feature_count(user, feature, liked)
