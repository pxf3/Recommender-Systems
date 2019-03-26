from naive_bayes import NaiveBayesRecommender
import unittest
import pandas as pd

ratings = pd.read_csv('test.csv')
ratings.columns = ['user', 'item', 'rating']

print(ratings)