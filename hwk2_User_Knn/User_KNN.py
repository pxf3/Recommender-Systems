# Homnework 2
# INFO 5871, Spring 2019
# Yushuo Ruan
# University of Colorado, Boulder

from typing import Dict

import pandas as pd
import numpy as np
import logging
from heapq import nlargest
import math

_logger = logging.getLogger(__name__)

class User_KNN:
    """
    User-user nearest-neighbor collaborative filtering with ratings. Not a very efficient implementation
    using data frames and tables instead of numpy arrays, which would be _much_ faster.

    Attributes:
        _ratings (pandas.DataFrame): Ratings with user, item, ratings
        _sim_cache (Dict of Dicts): a multi-level dictionary with user/user similarities pre-calculated
        _profile_means (Dict of float): a dictionary of user mean ratings
        _profile_lenghts (Dict of float): a dictionary of user profile vector lengths
        _item_means (Dict of float): a dictionary of item mean ratings
        _nhood_size (int): number of peers in each prediction
        _sim_threshold (float): minimum similarity for a neighbor
    """
    _ratings = None
    _sim_cache: Dict[int, Dict] = {}
    _profile_means: Dict[int, float] = {}
    _profile_lengths: Dict[int, float] = {}
    _item_means: Dict[int, float] = {}
    _nhood_size = 1
    _sim_threshold = 0

    def __init__(self, nhood_size, sim_threshold=0):
        """
        Args:
        :param nhood_size: number of peers in each prediction
        :param sim_threshold: minimum similarity for a neighbor
        """
        self._nhood_size = nhood_size
        self._sim_threshold = sim_threshold

    def get_users(self): return list(self._ratings.index.levels[0])

    def get_items(self): return list(self._ratings.index.levels[1])

    def get_profile(self, u): return self._ratings.loc[u]

    def get_profile_length(self, u): return self._profile_lengths[u]

    def get_profile_mean(self, u): return self._profile_means[u]

    def get_similarities(self, u): return self._sim_cache[u]

    def get_rating(self, u, i):
        """
        Args:
        :param u: user
        :param i: item
        :return: user's rating for item or None
        Issues a warning if the user has more than one rating for the same item. This indicates a problem
        with the data.
        """
        if (u,i) in self._ratings.index:
            maybe_rating = self._ratings.loc[u, i]
            if len(maybe_rating) == 1:
                return float(maybe_rating.iloc[0])
            else:  # More than one rating for the same item, shouldn't happen
                _logger.warning('Multiple ratings for an item - User %d Item %d', u, i)
                return None
        else: # User, item pair doesn't exist in index
            return None

    # TODO HOMEWORK 2: IMPLEMENT
    def compute_profile_length(self, u):
        """
        Computes the geometric length of a user's profile vector.
        :param u: user
        :return: length
        """
        length = math.sqrt((self.get_profile(u)**2).sum())
        return length

    # TODO HOMEWORK 2: IMPLEMENT
    def compute_profile_lengths(self):
        """
        Computes the profile length table `_profile_lengths`
        :return: None
        """
        users = self.get_users()
        for i in users:
            self._profile_lengths[i] = math.sqrt((self._ratings.loc[i]**2).sum())
            
    # TODO HOMEWORK 2: IMPLEMENT
    def compute_profile_means(self):
        """
        Computes the user mean rating table `_user_means`
        :return: None
        """
        for u in self.get_users():
            self._profile_means[u] = np.array(self.get_profile(u)).mean()
        
    # TODO HOMEWORK 2: IMPLEMENT
    def compute_item_means(self):
        """
        Computes the item means table `_item_means`
        :return: None
        """
        ratings2 = self._ratings.reset_index().sort_values('movieId').set_index(['movieId', 'userId'])
        items = self.get_items()
        for i in items:
            self._item_means[i] = (ratings2.loc[i]).mean()
        

    # TODO HOMEWORK 2: FINISH IMPLEMENTATION
    def compute_similarity_cache(self):
        """
        Computes the similarity cache table `_sim_cache`
        :return: Non
        """
        count = 0
        for u in self.get_users():
            # TODO Rest of code here
            self._sim_cache[u] = {}
            for v in self.get_users():
                if v == u:
                    continue
                self._sim_cache[u][v] = self.cosine(u, v)

            if count % 10 == 0:
                print("Processed user {} ({})".format(u, count))
            count += 1

    # TODO HOMEWORK 2: IMPLEMENT
    def get_overlap(self, u, v):
        """
        Computes the items in common between profiles. Hint: use set operations
        :param u: user1
        :param v: user2
        :return: set intersection
        """
        list_overlap = self.get_profile(u).index.intersection(self.get_profile(v).index)
        return np.array(list_overlap)

    # TODO HOMEWORK 2: FINISH IMPLEMENTATION
    def cosine(self, u, v):
        """
        Computes the cosine between u and v vectors
        :param u:
        :param v:
        :return: cosine value
        """
        #print("cosine of ", u, " and ", v)
        dot_prod = 0
        overlap = self.get_overlap(u, v)
        for movieId in overlap:
            # TODO Rest of implementation
            dot_prod = dot_prod + self.get_rating(u, movieId) * self.get_rating(v, movieId)

        #print(dot_prod/(self.get_profile_length(u)*self.get_profile_length(v)))
        return dot_prod/(self.get_profile_length(u)*self.get_profile_length(v))

    def fit(self, ratings):
        """
        Trains the model by computing the various cached elements. Technically, there isn't any training
        for a memory-based model.
        :param ratings:
        :return: None
        """
        self._ratings = ratings.set_index(['userId', 'movieId'])
        self.compute_profile_lengths()
        self.compute_profile_means()
        self.compute_similarity_cache()

    # TODO HOMEWORK 2: IMPLEMENT
    def neighbors(self, u, i):
        """
        Computes the user neighborhood
        :param u: user
        :param i: item
        :return:
        """
        candidates=[]
        for ui in self.get_users():
            if ui == u:
                continue
            if self.get_rating(ui, i) is not None:
                if self._sim_cache[u][ui] >= self._sim_threshold:
                    candidates.append(ui)
        new_candidates = nlargest(self._nhood_size, candidates, key=lambda v: self._sim_cache[u][v])

        #for j in new_candidates:
            #print(self._sim_cache[u][j])

        return new_candidates

     # TODO HOMEWORK 2: FINISH IMPLEMENTATION
    def predict(self, u, i):
        """
        Predicts the rating of user for item
        :param u: user
        :param i: item
        :return: predicted rating
        """
        #print("=================================")
        #print("predicting: ", u, " to ", i)
        peers = self.neighbors(u, i)
        #print("peers are: ")
        #for ij in peers:
            #print(ij, ": ", self._sim_cache[u][ij])
        sum1=0
        sum2=0
        if len(peers)==0:
            return self._profile_means[u]
        for v in peers:
            sum1 = sum1 + (self._sim_cache[u][v] * (self.get_rating(v, i) - self._profile_means[v]))
            #print("profile mean is: ", self._profile_means[v])
            sum2 = sum2 + abs(self._sim_cache[u][v])
        # TODO Rest of code
        #print("predicted rating is: ", self._profile_means[u] + (sum1 / sum2))

        return self._profile_means[u] + (sum1 / sum2)

    def predict_for_user(self, user, items, ratings=None):
        """
        Predicts the ratings for a list of items. This is used to calculate ranked lists.
        Note that the `ratings` parameters is required by the LKPy interface, but is not
        used by this algorithm.
        :param user:
        :param items:
        :param ratings:
        :return (pandas.Series): A score Series indexed by item.
        """
        scores = [self.predict(user, i) for i in items]

        return pd.Series(scores, index=pd.Index(items))
