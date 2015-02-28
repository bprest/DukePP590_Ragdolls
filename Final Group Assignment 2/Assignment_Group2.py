from __future__ import division  # imports the division capacity from the future version of Python
from pandas import Series, DataFrame
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind

print(time.ctime())
main_dir = u'C:/Users/Brianprest/Google Drive/THE RAGDOLLS_ 590 Big Data/CER_Data/CER Electricity Revised March 2012'
data_dir = main_dir+"/UnzippedData/"
assignmentfile = "SME and Residential allocations.xlsx"
timeseriescorrection = "timeseries_correction.csv"

## Read in the data
# Make a list of file paths for each of the file and then read it in as a dataframe
pathlist = [data_dir + "File" + str(v) + ".txt" for v in range(1,7)]
df = pd.concat([pd.read_table(v, sep = " ", names = ['panid','time','kwh']) for v in pathlist], ignore_index = True)
#df = pd.concat([pd.read_table(v, sep = " ", names = ['panid','time','kwh'], nrows = 1.5*10**6) for v in pathlist], ignore_index = True)

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
#df_monthly = pd.merge(df_monthly,assignment, on = ['panid'])
del [assignment, keeprows]

# Group variables on panid and day, then sum consumption across each day.
df['hour_cer'] = (df.time % 100)
df['day_cer'] = (df.time - df['hour_cer'])/100
del df['time']
# Pull in timestamps
tscorr = pd.read_csv(main_dir+"/"+timeseriescorrection, header=0, parse_dates=[1])
tscorr = tscorr[['year','month','day','hour_cer','day_cer']]

df = pd.merge(df,tscorr, on=['day_cer','hour_cer'])
del tscorr
del [[df['day_cer'],df['hour_cer']]]

# Aggregate on day
daygrp = df.groupby(['panid','tariff','year','month','day'])
df_daily= daygrp['kwh'].sum().reset_index()

# Aggregate on month
monthgrp = df.groupby(['panid','tariff','year','month'])
df_monthly = monthgrp['kwh'].sum().reset_index()
del df
del [daygrp, monthgrp]

# Group on treatment status and day
grp_daily = df_daily.groupby(['tariff','year','month','day'])
trt_daily = {(k[1],k[2],k[3]): df_daily.kwh[v].values for k,v in grp_daily.groups.iteritems() if k[0]=="A"} 
ctrl_daily = {(k[1],k[2],k[3]): df_daily.kwh[v].values for k,v in grp_daily.groups.iteritems() if k[0]=="E"}
del [df_daily, grp_daily]

# Group on treatment status and month
grp_monthly = df_monthly.groupby(['tariff','year','month'])
trt_monthly = {(k[1],k[2]): df_monthly.kwh[v].values for k,v in grp_monthly.groups.iteritems() if k[0]=="A"} 
ctrl_monthly = {(k[1],k[2]): df_monthly.kwh[v].values for k,v in grp_monthly.groups.iteritems() if k[0]=="E"}
del [df_monthly, grp_monthly]

keys_daily = trt_daily.keys()
keys_monthly = trt_monthly.keys()

# create dataframes of tstats over time
tstats_daily = DataFrame([(k[0], k[1], k[2], np.abs(ttest_ind(trt_daily[k],ctrl_daily[k], equal_var=False)[0])) for k in keys_daily], columns=['year','month','day','tstat'])
pvals_daily  = DataFrame([(k[0], k[1], k[2], np.abs(ttest_ind(trt_daily[k],ctrl_daily[k], equal_var=False)[1])) for k in keys_daily], columns=['year','month','day','pval'])
t_p_daily = pd.merge(tstats_daily,pvals_daily)
t_p_daily.sort(['year','month','day'], inplace=True)
t_p_daily.reset_index(inplace=True, drop=True)

tstats_monthly = DataFrame([(k[0], k[1], np.abs(ttest_ind(trt_monthly[k],ctrl_monthly[k], equal_var=False)[0])) for k in keys_monthly], columns=['year','month','tstat'])
pvals_monthly  = DataFrame([(k[0], k[1], np.abs(ttest_ind(trt_monthly[k],ctrl_monthly[k], equal_var=False)[1])) for k in keys_monthly], columns=['year','month','pval'])
t_p_monthly = pd.merge(tstats_monthly,pvals_monthly)
t_p_monthly.sort(['year','month'], inplace=True)
t_p_monthly.reset_index(inplace=True, drop=True)

del [trt_daily, ctrl_daily,trt_monthly, ctrl_monthly, keys_daily, keys_monthly]
del [tstats_daily, tstats_monthly, pvals_daily, pvals_monthly]

# Plotting -----------------------------------------
fig1 = plt.figure()
ax1 = fig1.add_subplot(2,1,1) 
ax1.plot(t_p_monthly.tstat)
ax1.axhline(2, color='red', linestyle="--")
ax1.axvline(x=6,ymin=0, ymax=3, color='green', linestyle="--")
ax1.set_title('Monthly t-stats over time')
ax2 = fig1.add_subplot(2,1,2) 
ax2.plot(t_p_monthly.pval)
ax2.axhline(0.05, color='red', linestyle="--")
ax2.axvline(x=6,ymin=0, ymax=1, color='green', linestyle="--")
ax2.set_title('Monthly p-values over time')
plt.show()

fig2 = plt.figure()
ax3 = fig2.add_subplot(2,1,1) 
ax3.plot(t_p_daily.tstat)
ax3.axhline(2, color='red', linestyle="--")
ax3.axvline(x=172, color='green', linestyle="--")
ax3.set_title('Daily t-stats over time')
ax4 = fig2.add_subplot(2,1,2) 
ax4.plot(t_p_daily.pval)
ax4.axhline(0.05, color='red', linestyle="--")
ax4.axvline(x=172, color='green', linestyle="--")
ax4.set_title('Daily p-values over time')
plt.show()

#t_p_daily.to_csv(main_dir + "/tstats_daily.csv", sep = ',')
#t_p_monthly.to_csv(main_dir + "/tstats_monthly.csv", sep = ',')

share_under_five_pct_daily   = sum(t_p_daily.pval<0.05)/t_p_daily.pval.count()
share_under_five_pct_monthly = sum(t_p_monthly.pval<0.05)/t_p_monthly.pval.count()
print("Share of daily p-values<0.05 is " + str(share_under_five_pct_daily))
print("Share of monthly p-values<0.05 is " + str(share_under_five_pct_monthly))
print("done!")
print(time.ctime())