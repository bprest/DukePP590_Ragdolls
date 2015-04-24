"""9/9 pt"""

from __future__ import division
import pandas as pd
import numpy as np
import os
from pandas import Series, DataFrame
import statsmodels.api as sm

########################## SECTION 1: Finding Imbalance #########################
main_dir = '/Users/dnoriega/Dropbox/pubpol590_sp15/data_sets/CER/tasks/4_task_data/'

os.chdir(main_dir)
from logit_functions import *

df = pd.read_csv(main_dir + "task_4_kwh_w_dummies_wide.csv")
df = df.dropna()

## 1.2.1 Test for imbalance running Logit
tariffs = [v for v in pd.unique(df['tariff']) if v != 'E']
stimuli = [v for v in pd.unique(df['stimulus']) if v != 'E']
tariffs.sort()
stimuli.sort()

drop = [v for v in df.columns if v.startswith("kwh_2010")]
df_pretrial = df.drop(drop, axis = 1)

for i in tariffs:
    for j in stimuli:
        logit_results, df_logit = do_logit(df_pretrial, i, j, add_D = None, mc = False)

## 1.2.2 Test for imbalance with a "Quick Means Comparison"
# create mean
grp = df_logit.groupby('tariff')
df_mean = grp.mean().transpose()
df_mean.C - df_mean.E

# do a t-test "by hand"
df_s = grp.std().transpose()
df_n = grp.count().transpose().mean()
top = df_mean['C'] - df_mean['E']
bottom = np.sqrt(df_s['C']**2/df_n['C'] + df_s['E']**2/df_n['E'])
tstats = top/bottom
sig = tstats[np.abs(tstats) > 2]
sig.name = 't-stats'

##################### SECTION 2: Propensity Score Weighting ####################
## 2.1 Get the predicted values of the logit model
df_logit['p_val'] = logit_results.predict()

## 2.2 Generate a column of weights called w
df_logit['trt'] = 0 + (df_logit['tariff'] == 'C')
df_logit['w'] = np.sqrt(df_logit['trt']/df_logit['p_val'] + (1 - df_logit['trt'])/(1 - df_logit['p_val']))

## 2.3 Create a smaller dataframe with just the IDs, treatments, and weights
df_w = df_logit[['ID', 'trt', 'w']]

##################### SECTION 3: Fixed Effects with Weights ####################
df = pd.read_csv(main_dir + "task_4_kwh_long.csv")
df = pd.merge(df, df_logit)

## 3.3 Create the necessary variables
# A treatment and trial interaction variable
df['trt_trial'] = df['trt']*df['trial']

# Log of kwh consumption plus 1
df['log_kwh'] = (df['kwh'] + 1).apply(np.log)

# A year-month column
df['mo_str'] = np.array(["0" + str(v) if v < 10 else str(v) for v in df['month']])
df['ym'] = df['year'].apply(str) + "_" + df['mo_str']

## 3.4 Set up regression variables from the merge dataframe
y = df['log_kwh']
TP = df['trt_trial']
P = df['trial']
w = df['w']
mu = pd.get_dummies(df['ym'], prefix = 'ym').iloc[:, 1:-1]
X = pd.concat([P, TP, mu], axis=1)

os.chdir(main_dir)
from fe_functions import *

## 3.5 De-mean y and X
ids = df['ID']
y = demean(y, ids)
X = demean(X, ids)

## 3.6 Run the Fixed Effects without AND with weights
### WITHOUT WEIGHTS
fe_model = sm.OLS(y, X) # linearly prob model
fe_results = fe_model.fit() # get the fitted values
print(fe_results.summary()) # print pretty results (no results given lack of obs)

### WITH WEIGHTS
# apply weights to data
y = y*w # weight each y
nms = X.columns.values # save column names
X = np.array([x*w for k, x in X.iteritems()]) # weight each X value
X = X.T # transpose (necessary as arrays create "row" vectors, not column)
X = DataFrame(X, columns = nms) # update to dataframe; use original names

fe_w_model = sm.OLS(y, X) # linearly prob model
fe_w_results = fe_w_model.fit() # get the fitted values
print(fe_w_results.summary()) # print pretty results (no results given lack of obs)