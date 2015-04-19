from __future__ import division
from pandas import Series, DataFrame
import pandas as pd
import numpy as np
import statsmodels.api as sm
import os
from scipy import linalg

main_dir = u'C:/Users/Brianprest/OneDrive/Grad School/GitHub/DukePP590_Ragdolls/Group Assignment 4/'
#main_dir = u'C:/Users/bcp17/OneDrive/Grad School/GitHub/DukePP590_Ragdolls/Group Assignment 4/'

# Change wd
os.chdir(main_dir)
from logit_functions import *
from fe_functions import *

#########################################################
#############        Section 1            ###############
#########################################################
# Import data
#df = pd.read_csv(main_dir + '14_B3_EE_w_dummies.csv')
df = pd.read_csv(main_dir + 'task_4_kwh_w_dummies_wide.csv')
df = df.dropna(axis=0, how='any')


tariffs = [v for v in pd.unique(df.tariff) if v!='E']
stimuli = [v for v in pd.unique(df.stimulus) if v!='E']
tariffs.sort()
stimuli.sort()

# Logit
drop = [v for v in df.columns if v.startswith("kwh_2010")]
df_pretrial = df.drop(drop, axis=1)
df_pretrial.columns.values

for i in tariffs:
    for j in stimuli:
        # Dummies start with "D_" and consumption starts with "kwh_"
        logit_results, df_logit = do_logit(df_pretrial, i, j, add_D=None, mc=False)
# Logit yields 4 significant variables out of 41 (ignoring the constant). 
# The significant ones are August kwh, December kwh, the dummy for having 3 people 
# over the age of 15 in the house (420.3), and the dummy for having an immersion electric water heater (4701.2).
# The dummies are not very concerning, but it is concerning that people with 
# higher past consumption are more likely to be treated.

df_logit.reset_index(inplace=True)
# Quick means
df_mean = df_logit.groupby('tariff').mean().transpose()
df_mean.C - df_mean.E

# t-test by hand
df_s = df_logit.groupby('tariff').std().transpose()
df_n = df_logit.groupby('tariff').count().transpose().mean()

diff = df_mean['C'] - df_mean['E']
btm = np.sqrt(df_s['C']**2/df_n['C'] + df_s['E']**2/df_n['E'])

tstats = diff/btm
sig = tstats[np.abs(tstats)>2]
sig.name = 'tstats'

# The means confirm that the treatment group (C) has higher average consumption
# than the control group in each month.
# The tstats for these differences are about 15, making this difference wildly significant.

#########################################################
#############        Section 2            ###############
#########################################################
df_logit['trt'] = 0 + (df_logit['tariff']=='C')

# Save predicted values
p_hat = pd.DataFrame(logit_results.predict())
p_hat.columns = ['p']

# Generate weights
df_logit['w'] = np.sqrt(df_logit['trt']/p_hat['p']+(1-df_logit['trt'])/(1-p_hat['p']))

# Create smaller dataset
df_w = df_logit[['ID','trt','w']]

#########################################################
#############        Section 3            ###############
#########################################################

df_trial = pd.read_csv(main_dir + "task_4_kwh_long.csv")

df_fe = pd.merge(df_w, df_trial)
df_fe.drop('tariff', axis=1, inplace=True)
df_fe.drop('stimulus', axis=1, inplace=True)

df_fe.reset_index(inplace=True)

#NT = len(df_fe)
#N = len(df_w)
#T = NT/N

df_fe['trt&trial'] = df_fe['trt']*df_fe['trial']
df_fe['log_kwh'] = (df_fe['kwh'] + 1).apply(np.log)

df_fe['mo_str'] = np.array(["0" + str(v) if v < 10 else str(v) for v in df_fe['month']])
# concatenate to make ym string values
df_fe['ym'] = df_fe['year'].apply(str) + "_" + df_fe['mo_str']

y = df_fe['log_kwh']
T = df_fe['trt']
TP = df_fe['trt&trial']
P = df_fe['trial']
w = df_fe['w']
mu = pd.get_dummies(df_fe['ym'], prefix = 'ym').iloc[:, 1:-1] # iloc[r:r+2, c:c+2] returns the r and r+1 rows, c and c+1 cols, of its object
#X = pd.concat([T, TP, mu], axis=1)
X = pd.concat([P, TP, mu], axis=1)
ids = df_fe['ID']

y = demean(y,ids)
X = demean(X,ids)

fe_model = sm.OLS(y, X) 
fe_results = fe_model.fit() 
print(fe_results.summary()) 

# WITH WEIGHTS
## apply weights to data
y_w = y*w # weight each y
nms = X.columns.values 
X_w = np.array([x*w for k, x in X.iteritems()]) # weight each X value
X_w = X_w.T # transpose (necessary as arrays create "row" vectors, not column)
X_w = DataFrame(X_w, columns = nms) # update to dataframe; use original names

fe_w_model = sm.OLS(y_w, X_w) 
fe_w_results = fe_w_model.fit()
print(fe_w_results.summary()) 

