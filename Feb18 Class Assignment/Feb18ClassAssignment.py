from __future__ import division  # imports the division capacity from the future version of Python
from pandas import Series, DataFrame
import pandas as pd
import numpy as np
import os
import xlrd
import time
import csv
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind

print(time.ctime())
main_dir = u'C:/Users/bcp17/Google Drive/THE RAGDOLLS_ 590 Big Data/CER_Data/CER Electricity Revised March 2012'
data_dir = main_dir+"/UnzippedData/"
git_dir = "C:/Users/bcp17/OneDrive/Grad School/GitHub/PubPol590"
assignmentfile = "SME and Residential allocations.xlsx"

## Read in the data
# Make a list of file paths for each of the file and then read it in as a dataframe
pathlist = [data_dir + "File" + str(v) + ".txt" for v in range(1,7)]
df = pd.concat([pd.read_table(v, sep = " ", names = ['panid','time','kwh']) for v in pathlist], ignore_index = True)
#df = pd.concat([pd.read_table(v, sep = " ", names = ['panid','time','kwh'], nrows = 1.5*10**6) for v in pathlist], ignore_index = True)

print(time.ctime())
# Clean each df separately (this is for efficiency, when running the full data)

# Load in treatment assignment info.
assignment = pd.read_excel(main_dir+"/"+assignmentfile, sep = ",", na_values=[' ','-','NA'], usecols = range(0,4))
assignment = assignment[assignment.Code==1] # keep only the residential guys
assignment = assignment[[0,2,3]] # drop "Code" now, since it's always 1.
assignment.columns = ['panid','tariff','stimulus']

# keep only the control guys (E,E) or guys with tariff A & bi-monthly only stimulus (A,1).
keeprows = ((assignment.tariff =="E") & (assignment.stimulus == "E")) | ((assignment.tariff == "A") & (assignment.stimulus == "1"))

# trim out all others
assignment = assignment[keeprows]

# Merge with panel data.
df = pd.merge(df,assignment, on = ['panid'])
del assignment

# Group variables on tariff and time. Note that we can ignore the stimulus, since within tariff A, stimulus is always 1.
grp = df.groupby(['tariff','time'])

trt = {k[1]: df.kwh[v].values for k,v in grp.groups.iteritems() if k[0]=="A"} 
ctrl = {k[1]: df.kwh[v].values for k,v in grp.groups.iteritems() if k[0]=="E"}
del [df, grp]

keys = trt.keys()

# create dataframes of this info
tstats = DataFrame([(k, np.abs(ttest_ind(trt[k],ctrl[k], equal_var=False)[0])) for k in keys], columns=['time','tstat'])
del [trt, ctrl, keys]
tstats.sort(['time'], inplace=True) # inplace replaces it (assigns t_p to the new thing. equivalent to t_p = t_p.sort(['date'])
tstats.reset_index(inplace=True, drop=True) # By default, it saves the old index. But drop=True drops this.

# Plotting -----------------------------------------
fig1 = plt.figure()
ax1 = fig1.add_subplot(1,1,1) 
ax1.plot(tstats['tstat'])
ax1.axhline(2, color='red', linestyle="--")
ax1.set_title('t-stats over time')
plt.show()
print("done!")
print(time.ctime())