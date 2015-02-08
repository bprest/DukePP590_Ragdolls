#%reset -f
from __future__ import division  # imports the division capacity from the future version of Python
from pandas import Series, DataFrame
import pandas as pd
import numpy as np
import os

main_dir = u'C:/Users/bcp17/Google Drive/THE RAGDOLLS_ 590 Big Data/CER_Data/CER Electricity Revised March 2012'
data_dir = main_dir+"/UnzippedData/"
git_dir = "C:/Users/bcp17/OneDrive/Grad School/GitHub/PubPol590"
assignmentfile = "SME and Residential allocations.csv"

# Read first 10 lines of the file to determine type.
# This reveals that it's space delimited
N = 10
with open(data_dir + "/" + "File1.txt") as myfile:
    head = [next(myfile) for x in xrange(N)]
print head

## Read in the data.
pathlist = [data_dir + v for v in os.listdir(data_dir) if v.startswith("File")]
list_of_dfs = [ pd.read_csv(v, names = ['panid', 'time', 'kwh'], sep = " ", header=None, na_values=['-','NA']) for v in pathlist]


# Remove Duplicates from each df
for i in list_of_dfs:
    i.drop_duplicates(['panid','time'], take_last=True)
    i.dropna(axis = 0 , how='any')
    hour = i.time % 100
    day = (i.time - hour)/100
    droprows = ((hour==4) | (hour==5)) & ((day==669) | (day==298)) 
    i.drop(droprows, inplace = True)
    replacerows = ((day==669) + (day==298)) & ((hour>=6) & (hour<=50))
    i.time[replacerows] = i.time[replacerows] - 2
    
#df[hour>50] # weird. panid 1208
##[min(df.panid[hour>50]), max(df.panid[hour>50])] # weird. panid 1208 has hour readings up to 95. Lets drop that one.
#droprows = (df.panid==1208)
##df.drop(droprows, inplace = True) # this failed on me. not sure why
#df = df[~droprows]
 
for i in range(0,6):
    print(len(list_of_dfs[i]))
# Stack dfs
df = pd.concat(list_of_dfs, ignore_index = True)


###### Fix daylight savings issues:
#* Day 452 has 2 missing entries numbered 2 and 3
#* Day 669 has 2 extra entries numbered 49 and 50
#* Day 298 has 2 extra entries numbered 49 and 50
# Fix these by dropping observations 4-5, and pulling 6-50 all back by 2.
hour = df.time % 100
day = (df.time - hour)/100
droprows = ((hour==4) | (hour==5)) & ((day==669) | (day==298)) 
df[droprows]
df.drop(droprows, inplace = True)
replacerows = ((day==669) + (day==298)) & ((hour>=6) & (hour<=50))
df.time[replacerows] = df.time[replacerows] - 2

df[hour>50] # weird. panid 1208
[min(df.panid[hour>50]), max(df.panid[hour>50])] # weird. panid 1208 has hour readings up to 95. Lets drop that one.
droprows = (df.panid==1208)
#df.drop(droprows, inplace = True) # this failed on me. not sure why
df = df[~droprows]

# Load in treatment assignment info.
assignment = pd.read_csv(main_dir+"/"+assignmentfile, sep = ",", na_values=[' ','-','NA'], usecols = range(0,4))
assignment = assignment[assignment.Code==1] # keep only the residential guys
assignment = assignment[[0,2,3]] # drop "Code" now, since it's always 1.
assignment.columns = ['panid','tariff','stimulus']

# Merge with panel data.
mergeddf = pd.merge(df,assignment, on = ['panid'])

print("done!")
    