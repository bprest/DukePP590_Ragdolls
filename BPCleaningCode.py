from __future__ import division  # imports the division capacity from the future version of Python
from pandas import Series, DataFrame
import pandas as pd
import numpy as np
import os

main_dir = u'C:/Users/bcp17/Google Drive/THE RAGDOLLS_ 590 Big Data/CER_Data/CER Electricity Revised March 2012/UnzippedData'
git_dir = "C:/Users/bcp17/OneDrive/Grad School/GitHub/PubPol590"

# Read first 10 lins of the file to determine type.
N = 10
with open(main_dir + "/" + "File1.txt") as myfile:
    head = [next(myfile) for x in xrange(N)]
print head
# It's space delimited

df_dict = dict()
for i in range(1,7):
    df_dict[i] = pd.read_csv(main_dir + "/" + "File" + str(i) +".txt", sep = " ", names = ['panid', 'time','consump'], header=None, na_values=['-','NA'])
print("done!")
df = pd.concat([df_dict[1], df_dict[2], df_dict[3], df_dict[4], df_dict[5], df_dict[6]], ignore_index=True)
#df1 = pd.read_csv(main_dir+"/"+"File1.txt", sep = " ", names = ['panid', 'time','consump'], header=None)
#df2 = pd.read_csv(main_dir+"/"+"File2.txt", sep = " ", names = ['panid', 'time','consump'], header=None)
#df3 = pd.read_csv(main_dir+"/"+"File3.txt", sep = " ", names = ['panid', 'time','consump'], header=None)
#df4 = pd.read_csv(main_dir+"/"+"File4.txt", sep = " ", names = ['panid', 'time','consump'], header=None)
#df5 = pd.read_csv(main_dir+"/"+"File5.txt", sep = " ", names = ['panid', 'time','consump'], header=None)
#df6 = pd.read_csv(main_dir+"/"+"File6.txt", sep = " ", names = ['panid', 'time','consump'], header=None)
#df = pd.concat([df1, df2, df3, df4, df5, df6], ignore_index = True)
#del [df1, df2, df3, df4, df5, df6]

df=df_dict[1] # temporary for code testing.
del df_dict
df.dtypes
sum(df.duplicated()) # No duplicates
dupe_on_panidtime_tb = df.duplicated(['panid','time'])
dupe_on_panidtime_bt = df.duplicated(['panid','time'], take_last=True)
sum(dupe_on_panidtime_tb)

# drop duplicated values, taking only the last.
df = df.drop_duplicates(['panid','date'], take_last=True)

# check for NaNs. Drop full row if any is missing.
isnan = np.isnan(df)
anyisnan = sum(isnan, axis = 1)>0 # True if any column is missing.
df = df[~anyisnan] # keep only the ones where no column is missing.

# Load in identifiers.



pd.merge(df1,df2, on = ['panid'])


print("done!")
    