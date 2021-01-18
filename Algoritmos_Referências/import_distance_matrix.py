# -*- coding: utf-8 -*-
"""
Created on Tue Dec 29 13:52:12 2020

@author: Pedro Ayres
"""

import numpy as np
import pandas as pd
from scipy.spatial import distance_matrix

# Original code from OP, slightly reformatted
DF_var = pd.DataFrame.from_dict({
    "s1":[1.2,3.4,10.2],
    "s2":[1.4,3.1,10.7],
    "s3":[2.1,3.7,11.3],
    "s4":[1.5,3.2,10.9]
}).T
DF_var.columns = ["g1","g2","g3"]

# Whole similarity algorithm in one line
df_euclid = pd.DataFrame(
    1 / (1 + distance_matrix(DF_var.T, DF_var.T)),
    columns=DF_var.columns, index=DF_var.columns
)

print(df_euclid)

#           g1        g2        g3
# g1  1.000000  0.215963  0.051408
# g2  0.215963  1.000000  0.063021
# g3  0.051408  0.063021  1.000000