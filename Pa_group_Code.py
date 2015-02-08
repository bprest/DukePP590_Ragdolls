from __future__ import division
from pandas import Series, DataFrame
import pandas as pd
import numpy as np
import os

main_dir = "/Users/Pa/Desktop/2015Spring/PUBPOL590/"
root = main_dir + "Data/Group/"
paths = [root + "File" + str(v) + ".txt" for v in range(1,7)]
SME_file = "SME\ and\ Residential\ allocations.xlsx"

list_of_dfs = [pd.read_table(v, names = ['ID', 'DayTime', 'kwh'], sep = "\s") for v in paths]

df_assign = pd.read_csv(root + SME_file, usecols = [0:5])

df = pd.concat(list_of_dfs, ignore_index = True)
df = pd.merge(df, df_assign)

df1 = df.copy()
df.date[66949] = 67001
df.date[66950] = 67002
df.date[29849] = 29901
df.date[29850] = 29902