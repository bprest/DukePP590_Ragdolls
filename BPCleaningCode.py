from __future__ import division  # imports the division capacity from the future version of Python
from pandas import Series, DataFrame
import pandas as pd
import numpy as np
import os
import xlrd

main_dir = u'C:/Users/bcp17/Google Drive/THE RAGDOLLS_ 590 Big Data/CER_Data/CER Electricity Revised March 2012'
data_dir = main_dir+"/UnzippedData/"
git_dir = "C:/Users/bcp17/OneDrive/Grad School/GitHub/PubPol590"
assignmentfile = "SME and Residential allocations.csv"

# Read first 10 lins of the file to determine type.
# This reveals that it's space delimited
N = 10
with open(main_dir + "/" + "File1.txt") as myfile:
    head = [next(myfile) for x in xrange(N)]
print head


# Read in the data.
pathlist = [data_dir + v for v in os.listdir(data_dir) if v.startswith("File")]
list_of_dfs = [ pd.read_csv(v, names = ['panid', 'date', 'kwh'], sep = " ", header=None, na_values=['-','NA']) for v in pathlist]

# Now clean the data
df=pd.read_csv(data_dir+"File1.txt", names = ['panid', 'date', 'kwh'], sep = " ", header=None, na_values=['-','NA']) # temporary, for debugging code.

# Determine number of fully duplicated rows.
print("Number of Fully Duplicated Rows: ") 
print(sum(df.duplicated())) 
# No full duplicates

# Determine duplicates on Panid & Time.
dupe_on_panidtime_tb = df.duplicated(['panid','time'])
dupe_on_panidtime_bt = df.duplicated(['panid','time'], take_last=True)

# Print Number of Duplicated Rows (on Panid & Time)
print("Number of Duplicated Rows on Panid & Time: ")
print(sum(dupe_on_panidtime_tb))

# Print duplicated values
print("Duplicates on Panid & Time: ")
print(df[dupe_on_panidtime_bt | dupe_on_panidtime_tb])
# Ack! There are at least 12 duplicate values (same panid, same time). Only take the last set of these.

# drop duplicated values, taking only the last.
df = df.drop_duplicates(['panid','date'], take_last=True)

# check for NaNs. Drop full row if any column is missing.
isnan = np.isnan(df)
print("Number of NaNs:",sum(isnan))
anyisnan = sum(isnan, axis = 1)>0 # True if any column is missing.
df = df[~anyisnan] # keep only the ones where no column is missing.

# Fix daylight savings issues:
#* Day 452 has 2 missing entries numbered 2 and 3
#* Day 669 has 2 extra entries numbered 49 and 50
#* Day 298 has 2 extra entries numbered 49 and 50
# Fix these by dropping observations 4-5, and pulling 6-50 all back by 2.


# Load in treatment assignment info.
assignment = pd.read_csv(main_dir+"/"+assignmentfile, sep = ",", na_values=[' ','-','NA'], usecols = range(0,4))
assignment = assignment[assignment.Code==1] # keep only the residential guys
assignment = assignment[[0,2,3]] # drop "Code" now, since it's always 1.
assignment.columns = ['panid','tariff','stimulus']

# Merge with panel data.
mergeddf = pd.merge(df,assignment, on = ['panid'])

print("done!")
    