from __future__ import division  # imports the division capacity from the future version of Python
from pandas import Series, DataFrame
import pandas as pd
import numpy as np
import os
import xlrd
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt

main_dir = "/Users/Pa/Desktop/2015Spring/PUBPOL590/"
root = main_dir + "Data/Group/"
assignmentfile = "SME and Residential allocations.xlsx"

paths = [os.path.join(root, v) for v in os.listdir(root) if v.startswith("File")]

# READ IN THE DATA--------------------------------------------------------------
df = pd.concat([pd.read_table(v, names = ['panid', 'time', 'kwh'], sep = ' ') 
    for v in paths], ignore_index = True)

## Create new variables
df['hour'] = (df['time'] % 100)
df['day'] = (df['time'] - df['hour'])/100

# MERGE WITH TIME CORRECTION FILE-----------------------------------------------
df_time_correction = pd.read_csv(main_dir + "Data/Demo9/timeseries_correction.csv",
    sep = ',', parse_dates=[1])

df_time_correction = df_time_correction[['ts', 'year', 'month', 'hour_cer', 'day_cer']]
df_time_correction.rename(columns = {'hour_cer': 'hour', 'day_cer': 'day'}, 
    inplace = True)

df = pd.merge(df, df_time_correction, on = ['day', 'hour'])

# ASSIGNMENT--------------------------------------------------------------------
assignment = pd.read_excel(root + assignmentfile, sep = ',', usecols = range(0,4))
assignment.columns = ['panid', 'code', 'tariff', 'stimulus']
    
assignment = assignment[(assignment.code == 1) & ((assignment.stimulus == "E") 
    | (assignment.stimulus == "1")) & ((assignment.tariff == "A") 
    | (assignment.tariff == "E"))]

df = pd.merge(df, assignment, on = ['panid'])

# AGGREGATION (daily)-----------------------------------------------------------
grp_daily = df.groupby(['year', 'month', 'day', 'panid', 'tariff', 'stimulus'])
agg_daily = grp_daily['kwh'].sum()

agg_daily = agg_daily.reset_index()
grp_daily = agg_daily.groupby(['year', 'month', 'day', 'tariff', 'stimulus'])

# Split up T/C (daily)
trt_daily = {(k[0], k[1], k[2]): agg_daily.kwh[v].values 
        for k,v in grp_daily.groups.iteritems() if k[3] == 'A' or k[4] == '1'}
ctrl_daily = {(k[0], k[1], k[2]): agg_daily.kwh[v].values 
        for k,v in grp_daily.groups.iteritems() if k[3] == 'E' and k[4] == 'E'}
keys_daily = trt_daily.keys()

# tstats and pvals (daily)
tstats_daily = DataFrame([(k[0], k[1], k[2], np.abs(ttest_ind(trt_daily[k], 
    ctrl_daily[k], equal_var = False)[0])) for k in keys_daily], 
    columns = ['year', 'month', 'day', 'tstat_daily'])
pvals_daily = DataFrame([(k[0], k[1], k[2], (ttest_ind(trt_daily[k], 
    ctrl_daily[k], equal_var = False)[1])) for k in keys_daily], 
    columns = ['year', 'month', 'day', 'pval_daily'])
t_p_daily = pd.merge(tstats_daily, pvals_daily)

# sort and reset (daily)
t_p_daily.sort(['year', 'month', 'day'], inplace = True)
t_p_daily.reset_index(inplace = True, drop = True)

# PLOTTING (daily)--------------------------------------------------------------
fig_daily= plt.figure()
ax1 = fig_daily.add_subplot(2,1,1)
ax1.plot(t_p_daily['tstat_daily'])
ax1.axhline(2, color = 'r', linestyle = '--')
ax1.axvline(172, color = 'g', linestyle = '--')
ax1.set_title('Daily t-stats over time')

ax2 = fig_daily.add_subplot(2,1,2)
ax2.plot(t_p_daily['pval_daily'])
ax2.axhline(0.05, color = 'r', linestyle = '--')
ax2.axvline(172, color = 'g', linestyle = '--')
ax2.set_title('Daily p-values over time')

# AGGREGATION (monthly)---------------------------------------------------------
grp_monthly = df.groupby(['year', 'month', 'panid', 'tariff', 'stimulus'])
agg_monthly = grp_monthly['kwh'].sum()

agg_monthly = agg_monthly.reset_index()
grp_monthly = agg_monthly.groupby(['year', 'month', 'tariff', 'stimulus'])

# split up T/C (monthly)
trt_monthly = {(k[0], k[1]): agg_monthly.kwh[v].values 
        for k,v in grp_monthly.groups.iteritems() if k[2] == 'A' or k[3] == '1'}
ctrl_monthly = {(k[0], k[1]): agg_monthly.kwh[v].values 
        for k,v in grp_monthly.groups.iteritems() if k[2] == 'E' and k[3] == 'E'}
keys_monthly = trt_monthly.keys()

# tstats and pvals (monthly)
tstats_monthly = DataFrame([(k[0], k[1], np.abs(ttest_ind(trt_monthly[k], 
    ctrl_monthly[k], equal_var = False)[0])) for k in keys_monthly], 
    columns = ['year', 'month', 'tstat_monthly'])    
pvals_monthly = DataFrame([(k[0], k[1], (ttest_ind(trt_monthly[k], 
    ctrl_monthly[k], equal_var = False)[1])) for k in keys_monthly], 
    columns = ['year', 'month', 'pval_monthly'])
t_p_monthly = pd.merge(tstats_monthly, pvals_monthly)

# sort and reset (monthly)
t_p_monthly.sort(['year', 'month'], inplace = True)
t_p_monthly.reset_index(inplace = True, drop = True)

# PLOTTING (monthly)------------------------------------------------------------
fig_monthly= plt.figure()
ax3 = fig_monthly.add_subplot(2,1,1)
ax3.plot(t_p_monthly['tstat_monthly'])
ax3.axhline(2, color = 'r', linestyle = '--')
ax3.axvline(6, color = 'g', linestyle = '--')
ax3.set_title('Monthly t-stats over time')

ax4 = fig_monthly.add_subplot(2,1,2)
ax4.plot(t_p_monthly['pval_monthly'])
ax4.axhline(0.05, color = 'r', linestyle = '--')
ax4.axvline(6, color = 'g', linestyle = '--')
ax4.set_title('Monthly p-values over time')
