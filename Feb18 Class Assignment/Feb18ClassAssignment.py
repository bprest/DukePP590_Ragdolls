from __future__ import division  # imports the division capacity from the future version of Python
from pandas import Series, DataFrame
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind

print(time.ctime())
main_dir = u'C:/Users/bcp17/Google Drive/THE RAGDOLLS_ 590 Big Data/CER_Data/CER Electricity Revised March 2012'
data_dir = main_dir+"/UnzippedData/"
assignmentfile = "SME and Residential allocations.xlsx"

## Read in the data
# Make a list of file paths for each of the file and then read it in as a dataframe
pathlist = [data_dir + "File" + str(v) + ".txt" for v in range(1,7)]
df = pd.concat([pd.read_table(v, sep = " ", names = ['panid','time','kwh']) for v in pathlist], ignore_index = True)
#df = pd.concat([pd.read_table(v, sep = " ", names = ['panid','time','kwh'], nrows = 1.5*10**6) for v in pathlist], ignore_index = True)

# Group variables on panid and day, then sum consumption across each day.
df['day'] = (df.time - (df.time % 100))/100
hourgrp = df.groupby(['panid','day'])
del df
df_daily = DataFrame(zip(hourgrp.panid.mean(),hourgrp.day.mean(),hourgrp['kwh'].sum()), columns = ['panid','day','kwh'])
del hourgrp

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
df_daily = pd.merge(df_daily,assignment, on = ['panid'])
del [assignment, keeprows]

# Group on treatment status and day
grp = df_daily.groupby(['tariff','day'])

trt = {k[1]: df_daily.kwh[v].values for k,v in grp.groups.iteritems() if k[0]=="A"} 
ctrl = {k[1]: df_daily.kwh[v].values for k,v in grp.groups.iteritems() if k[0]=="E"}
del [df_daily, grp]

keys = trt.keys()

# create dataframes of tstats over time
tstats = DataFrame([(k, np.abs(ttest_ind(trt[k],ctrl[k], equal_var=False)[0])) for k in keys], columns=['time','tstat'])
del [trt, ctrl, keys]
tstats.sort(['time'], inplace=True)
tstats.reset_index(inplace=True, drop=True)

# Plotting -----------------------------------------
fig1 = plt.figure()
ax1 = fig1.add_subplot(1,1,1) 
ax1.plot(tstats['tstat'])
ax1.axhline(2, color='red', linestyle="--")
ax1.set_title('t-stats over time')
plt.show()

tstats.to_csv(main_dir + "/tstats_daily.csv", sep = ',')

share_over_two = sum(tstats.tstat>2)/tstats.tstat.count()
print("Share of tstats>2 is " + str(share_over_two))
print("done!")
print(time.ctime())