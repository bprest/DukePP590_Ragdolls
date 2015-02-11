from __future__ import division  # imports the division capacity from the future version of Python
from pandas import Series, DataFrame
import pandas as pd
import numpy as np
import os
import xlrd
import time
import csv

print(time.ctime())
main_dir = u'C:/Users/bcp17/Google Drive/THE RAGDOLLS_ 590 Big Data/CER_Data/CER Electricity Revised March 2012'
data_dir = main_dir+"/UnzippedData/"
git_dir = "C:/Users/bcp17/OneDrive/Grad School/GitHub/PubPol590"
assignmentfile = "SME and Residential allocations.xlsx"

# Read first 10 lines of the file to determine type.
# This reveals that it's space delimited
N = 10
with open(data_dir + "/" + "File1.txt") as myfile:
    head = [next(myfile) for x in xrange(N)]
print head

## Read in the data
# Make a list of file paths for each of the file and then read it in as a dataframe
pathlist = [data_dir + "File" + str(v) + ".txt" for v in range(1,7)]
list_of_dfs = [ pd.read_table(v, skiprows = 6*10**6, nrows = 1.5*10**6, names = ['panid', 'time', 'kwh'], sep = " ", header=None, na_values=['-','NA']) for v in pathlist]
## For the full data, use this line instead:
#list_of_dfs = [ pd.read_table(v, names = ['panid', 'time', 'kwh'], sep = " ", header=None, na_values=['-','NA']) 

# Clean each df separately (this is for efficiency, when running the full data)
for i in list_of_dfs:
    i.drop_duplicates(['panid','time'], take_last=True)
    i.dropna(axis = 0 , how='any')
    hour = i.time % 100
    day = (i.time - hour)/100
    
    # Drop and replace hours on the October DST days.
    droprows = ((hour==5) | (hour==6)) & ((day==669) | (day==298)) 
    i.drop(droprows, inplace = True)
    replacerows = ((day==669) | (day==298)) & ((hour>=7) & (hour<=50))
    i.time[replacerows] = i.time[replacerows] - 2
    
    # Now replace hours on the March DST day.
    replacerows = ((day==452)) & ((hour>=5))    
    i.time[replacerows] = i.time[replacerows] + 2

# Stack
# For the longer data set, it was more efficient to concat 2 dfs and then delete in memory
# in order to reduce RAM usage
df = list_of_dfs[0]
for i in range(5,0,-1):
    df = pd.concat([df, list_of_dfs[i]], ignore_index = True)
    del list_of_dfs[i]
    
# Additional Cleaning (originally using complete dataset)
# df[hour>48]
# Returned weird hour reading besides day = 669 and 298
# panid 1208 and 5221 have hour readings up to 95. Lets drop those.
# Note: in the shortened data, these anomalous results do not appear. But we drop them anyway.
droprows = (df.panid==1208) | (df.panid==5221)
df = df[~droprows]

# Load in treatment assignment info.
assignment = pd.read_excel(main_dir+"/"+assignmentfile, sep = ",", na_values=[' ','-','NA'], usecols = range(0,4))
assignment = assignment[assignment.Code==1] # keep only the residential guys
assignment = assignment[[0,2,3]] # drop "Code" now, since it's always 1.
assignment.columns = ['panid','tariff','stimulus']

# Merge with panel data.
mergeddf = pd.merge(df,assignment, on = ['panid'])

# Write to csv to save it.
mergeddf.to_csv(main_dir + "/merged_data.csv", sep = ',')

print("done!")
print(time.ctime())