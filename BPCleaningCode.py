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
#N = 10
#with open(data_dir + "/" + "File1.txt") as myfile:
#    head = [next(myfile) for x in xrange(N)]
#print head

# Read in the data.
pathlist = [data_dir + v for v in os.listdir(data_dir) if v.startswith("File")]
list_of_dfs = [ pd.read_csv(v, names = ['panid', 'date', 'kwh'], sep = " ", header=None, na_values=['-','NA']) for v in pathlist]
df = pd.concat(list_of_dfs, ignore_index = True)

#################### temporary, for debugging code.
#df=pd.read_csv(data_dir+"File1.txt", names = ['panid', 'time', 'kwh'], sep = " ", header=None, na_values=['-','NA']) 
#################### temporary, for debugging code.

### Now clean the data
# Determine number of fully duplicated rows.
#print("Number of Fully Duplicated Rows: ") 
#print(sum(df.duplicated())) 
# No full duplicates

# Determine duplicates on Panid & Time.
dupe_on_panidtime_tb = df.duplicated(['panid','time'])
dupe_on_panidtime_bt = df.duplicated(['panid','time'], take_last=True)

# Print Number of Duplicated Rows (on Panid & Time)
#print("Number of Duplicated Rows on Panid & Time: ")
#print(sum(dupe_on_panidtime_tb))

# Print duplicated values
#print("Duplicates on Panid & Time: ")
#print(df[dupe_on_panidtime_bt | dupe_on_panidtime_tb])
# Ack! There are at least 12 duplicate values (same panid, same time). Only take the last set of these.

# drop duplicated values, taking only the last.
df = df.drop_duplicates(['panid','time'], take_last=True)
del [dupe_on_panidtime_bt, dupe_on_panidtime_tb]
# check for NaNs. Drop full row if any column is missing.
missing = np.isnan(df)
#anymissing = sum(missing, axis = 1)
#print("Number of NaNs: ")
#print(sum(missing))
df.dropna(axis = 0 , how='any') # keep only the ones where no column is missing.
#print("Rows dropped: ")
#print(str(sum(anymissing)))
#del [anymissing, missing]

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
    