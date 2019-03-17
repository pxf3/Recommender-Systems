# Homnework 2
# INFO 5871, Spring 2019
# Yushuo Ruan
# University of Colorado, Boulder

import User_KNN2
import unittest
import pandas as pd

class test_User_KNN2(unittest.TestCase):

    _user_knn2 = None
    _ratings = None

    def setUp(self):
        self._user_knn2 = User_KNN2.User_KNN2(2, sim_threshold=0, shrinkage=2, user_label='user', item_label='item',
                                             rating_label='score')
        self._ratings = pd.read_csv('test2.csv')

        self._user_knn2.fit(self._ratings)

    def test_get_users(self):
        users = self._user_knn2.get_users()
        self.assertEqual(len(users), 5, "Incorrect number of test users")

    def test_get_items(self):
        items = self._user_knn2.get_items()
        self.assertEqual(len(items), 5, "Incorrect number of test items")

    def test_get_profile(self):
        prof_3 = self._user_knn2.get_profile(3)
        self.assertEqual(len(prof_3), 3, "Error in get_profile. Wrong profile length")

    def test_get_rating(self):
        # Check one from each profile
        prof_1_1 = self._user_knn2.get_rating(1,1)
        self.assertAlmostEqual(prof_1_1, 1, 5, "Error in get_rating. Wrong rating value")
        prof_2_2 = self._user_knn2.get_rating(2,2)
        self.assertAlmostEqual(prof_2_2, 4, 5, "Error in get_rating. Wrong rating value")
        prof_3_3 = self._user_knn2.get_rating(3,3)
        self.assertAlmostEqual(prof_3_3, 5, 5, "Error in get_rating. Wrong rating value")
        prof_4_4 = self._user_knn2.get_rating(4,4)
        self.assertAlmostEqual(prof_4_4, 2, 5, "Error in get_rating. Wrong rating value")
        prof_5_1 = self._user_knn2.get_rating(5,1)
        self.assertAlmostEqual(prof_5_1, 3, 5, "Error in get_rating. Wrong rating value")

    def test_get_profile_length(self):
        len1 = self._user_knn2.get_profile_length(1)
        self.assertAlmostEqual(len1, 5.4772256, 5, "Error in profile length computation")

    def test_get_profile_mean(self):
        mean4 = self._user_knn2.get_profile_mean(4)
        self.assertAlmostEqual(mean4, 3.0, 5, "Error in profile mean computation")

    def test_similarities_shrinkage(self):
        sim4 = self._user_knn2.get_similarities(4)
        self.assertAlmostEqual(sim4[1], 0.22247, 5, "Error in similarity computation")
        self.assertAlmostEqual(sim4[2], 0.28604, 5, "Error in similarity computation")
        self.assertAlmostEqual(sim4[3], 0.10101, 5, "Error in similarity computation")
        self.assertAlmostEqual(sim4[5], 0.33183, 5, "Error in similarity computation")

    ### TODO HOMEWORK 2: CALCULATE THE CORRECT SIMILARITY BETWEEN USERS 1 AND 2 AND SUBSTITUTE THE VALUE IN THE
    ### ASSERT STATEMENT BELOW
    def test_similarities_hwk2_shrinkage(self):
        sim2 = self._user_knn2.get_similarities(2)
        self.assertAlmostEqual(sim2[1], 0.5, 5, "Error in similarity computation FIX FOR HOMEWORK 2")
        ### INSERT CORRECT VALUE HERE   ^

    def test_get_overlap(self):
        ol = self._user_knn2.get_overlap(1,4)
        self.assertEqual(len(ol), 2, "Overlap computation incorrect")

    def test_neighbors(self):
        peers = self._user_knn2.neighbors(1, 5)
        self.assertEqual(len(peers), 2, "Neighbor computation incorrect")

    def test_predict_shrinkage(self):
        pred = self._user_knn2.predict(1, 5)
        self.assertAlmostEqual(pred, 1.27821, 4, 'Prediction incorrect')
        pred = self._user_knn2.predict(4, 2) # This tests division by zero
        self.assertAlmostEqual(pred, 3.62501, 4, 'Prediction incorrect')

    ### TODO HOMEWORK 2: CALCULATE THE CORRECT PREDICTION FOR USER 3 AND ITEM 5 AND SUBSTITUTE THE VALUE IN THE
    ### ASSERT STATEMENT BELOW
    def test_predict_hwk2_shrinkage(self):
        pred = self._user_knn2.predict(3, 5)
        self.assertAlmostEqual(pred, 1.94132, 4, 'Prediction incorrect. FIX FOR HOMEWORK 2')
        ### INSERT CORRECT VALUE HERE^


if __name__ == '__main__':
    unittest.main()
