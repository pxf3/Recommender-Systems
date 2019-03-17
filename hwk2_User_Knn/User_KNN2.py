# Homnework 2
# INFO 5871, Spring 2019
# Yushuo Ruan
# University of Colorado, Boulder

import numpy as np
from User_KNN import User_KNN

# Subclass of User_KNN, so only unique functionality needs to be implemented.
class User_KNN2(User_KNN):
    _shrinkage = 0
    _user_label = 'userId'
    _item_label = 'itemId'
    _rating_label = 'rating'

    def __init__(self, nhood_size, sim_threshold=0, shrinkage=2,
                 user_label='user', item_label='item', rating_label='score'):
        User_KNN.__init__(self, nhood_size, sim_threshold=sim_threshold)

        self._shrinkage = shrinkage
        self._user_label = user_label
        self._item_label = item_label
        self._rating_label = rating_label

    # TODO HOMEWORK 2: FINISH IMPLEMENTATION
    #Need override because of rating label
    def compute_profile_length(self, u):
        """
        Computes the length of a user's profile vector.
        :param u: user
        :return: length
        """

        return 0

    # TODO HOMEWORK 2: IMPLEMENT
    # Need override because of rating label.
    def compute_item_means(self):
        """
        Computes the item means table `_item_means`
        :return: None
        """
        ratings2 = self._ratings.reset_index().sort_values(self._item_label).set_index([self._item_label, self._user_label])
        items = self.get_items()
        for i in items:
            self._item_means[i] = np.array(ratings2.loc[i]).mean()

    # TODO HOMEWORK 2: IMPLEMENT
    # Need override because of shrinkage calculation
    def compute_similarity_cache(self):
        count = 0
        for u in self.get_users():
            # TODO Rest of code here
            self._sim_cache[u] = {}
            for v in self.get_users():
                if v == u:
                    continue
                overlap = len(self.get_overlap(u, v))
                shrink = overlap/(overlap+self._shrinkage)
                self._sim_cache[u][v] = self.cosine(u, v)*shrink

            if count % 10 == 0:
                print("Processed user {} ({})".format(u, count))
            count += 1


    # TODO HOMEWORK 2: IMPLEMENT
    # Need override because of user and item labels for indexing on columns
    def fit(self, ratings):
        """
        Trains the model by computing the various cached elements. Technically, there isn't any training
        for a memory-based model.
        :param ratings:
        :return: None
        """
        self._ratings = ratings.set_index([self._user_label, self._item_label])
        self.compute_profile_lengths()
        self.compute_profile_means()
        self.compute_similarity_cache()

