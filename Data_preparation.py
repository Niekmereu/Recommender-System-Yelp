# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 10:59:13 2019

@author: Niek
"""

import pandas as pd
ratings = pd.read_csv("Enter location", sep=',',header=0)
ratings = ratings[['user_id', 'business_id', 'stars']]

#convert id's to numbers
ratings['user_id'] = pd.factorize(ratings.user_id)[0] + 1
ratings['business_id'] = pd.factorize(ratings.business_id)[0] + 1

ratings.to_csv("Enter location 2", sep =';')
