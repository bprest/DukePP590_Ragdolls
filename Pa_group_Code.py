from __future__ import division
from pandas import Series, DataFrame
import pandas as pd
import numpy as np
import os

main_dir = "/Users/Pa/Desktop/2015Spring/PUBPOL590/"
root = main_dir + "Data/Group/"
paths = [root + "File" + str(v) + ".txt" for v in range(1,7)]
SME_file = "SME and Residential allocations.xlsx"

list_of_dfs = [pd.read_table(v, names = ['ID', 'DayTime', 'kwh'], sep = "\s", ) for v in paths]

df = pd.concat(list_of_dfs, ignore_index = True)

t_b = df.duplicated(['ID', 'DayTime'])
b_t = df.duplicated(['ID', 'DayTime'], take_last = True)
unique = ~(t_b | b_t)
df_duplicates_dropped = df[unique]

df_cleaned = df_duplicates_dropped.dropna()

df_assign = pd.read_csv(root + SME_file, usecols = range(0,5))

df_merge = pd.merge(df_cleaned, df_assign)

df1 = df_merge.copy()
df_merge.date[66949] = 67001
df_merge.date[66950] = 67002
df_merge.date[29849] = 29901
df_merge.date[29850] = 29902