from __future__ import division
from pandas import Series, DataFrame
import pandas as pd
import numpy as np
import statsmodels.api as sm
import os

main_dir = "/Users/Pa/Desktop/2015Spring/PUBPOL590/"
root = main_dir +"Data/3_task_data/"
df_assign = pd.read_csv(root + "allocation_subsamp.csv", usecols = [0,2,3], header = 0)

## 1. create 5 unique vectors using the data from allocation_subsamp.csv
    ### df_assign['treatment'] = df_assign['tariff'] + df_assign['stimulus']
    ### dfEE = df_assign[df_assign['treatment']=='EE']
    ### dfA1 = df_assign[df_assign['treatment']=='A1']
    ### dfA3 = df_assign[df_assign['treatment']=='A3']
    ### dfB1 = df_assign[df_assign['treatment']=='B1']
    ### dfB3 = df_assign[df_assign['treatment']=='B3']

    ### control = dfEE.ID
    ### treatA1 = dfA1.ID
    ### treatA3 = dfA3.ID
    ### treatB1 = dfB1.ID
    ### treatB3 = dfB3.ID

control = df_assign.ID[(df_assign.tariff == 'E') & (df_assign.stimulus == 'E')]
treatA1 = df_assign.ID[(df_assign.tariff == 'A') & (df_assign.stimulus == '1')]
treatA3 = df_assign.ID[(df_assign.tariff == 'A') & (df_assign.stimulus == '3')]
treatB1 = df_assign.ID[(df_assign.tariff == 'B') & (df_assign.stimulus == '1')]
treatB3 = df_assign.ID[(df_assign.tariff == 'B') & (df_assign.stimulus == '3')]

## 2. set the random seed to 1789
np.random.seed(seed=1789)

## 3. use the function np.random.choice to extract samples without replacement
sample_control = np.random.choice(control, 300, replace=False)
sample_A1 = np.random.choice(treatA1, 150, replace=False)
sample_A3 = np.random.choice(treatA3, 150, replace=False)
sample_B1 = np.random.choice(treatB1, 50, replace=False)
sample_B3 = np.random.choice(treatB3, 50, replace=False)

## 4. create a DataFrame with all the the sampled IDs.
sample = sample_control.tolist() + sample_A1.tolist() + sample_A3.tolist() + sample_B1.tolist() + sample_B3.tolist()
sample = DataFrame(sample, columns = ['ID'])

## 5. import the consumption data from kwh_redux_pretrail.csv
df = pd.read_csv(root + "kwh_redux_pretrial.csv", header = 0)

## 6. merge the consumption data with the sampled IDs
df = pd.merge(df, sample, on = ['ID'])

## 7. aggregate all the consumption data by month for each separate group
grp = df.groupby(['ID', 'year', 'month'])
agg = grp['kwh'].sum().reset_index()

## 8. pivot the data from long to wide, so that kwh for each month is a variable.
agg['kwh_month'] = 'kwh_' + agg.month.apply(str) 
df_piv = agg.pivot('ID', 'kwh_month', 'kwh')
df_piv.reset_index(inplace = True)
df_piv.columns.name = None

## 9. merge the wide dataset with the treatment data
df = pd.merge(df_piv, df_assign, on = ['ID'])

## 10. compute a logit model comparing each treatment group to the control using only the consumption data.
kwh_cols = [v for v in df.columns.values if v.startswith('kwh')]

# SET UP Y, X
df['treatment'] = df['tariff'] + df['stimulus']
dfEEA1 = df[(df.treatment == 'EE') | (df.treatment == 'A1')]
dfEEA3 = df[(df.treatment == 'EE') | (df.treatment == 'A3')]
dfEEB1 = df[(df.treatment == 'EE') | (df.treatment == 'B1')]
dfEEB3 = df[(df.treatment == 'EE') | (df.treatment == 'B3')]


dfEEA1['T'] = 0 + (df.treatment == 'A1')
dfEEA3['T'] = 0 + (df.treatment == 'A3')
dfEEB1['T'] = 0 + (df.treatment == 'B1')
dfEEB3['T'] = 0 + (df.treatment == 'B3')

 # Logit EE, A1
yEEA1 = dfEEA1['T']
XEEA1 = dfEEA1[kwh_cols]
XEEA1 = sm.add_constant(XEEA1)

logit_model_EEA1 = sm.Logit(yEEA1, XEEA1)
logit_results_EEA1 = logit_model_EEA1.fit()
print(logit_results_EEA1.summary())

 # Logit EE, A3
yEEA3 = dfEEA3['T']
XEEA3 = dfEEA3[kwh_cols]
XEEA3 = sm.add_constant(XEEA3)

logit_model_EEA3 = sm.Logit(yEEA3, XEEA3)
logit_results_EEA3 = logit_model_EEA3.fit()
print(logit_results_EEA3.summary())

 # Logit EE, B1
yEEB1 = dfEEB1['T']
XEEB1 = dfEEB1[kwh_cols]
XEEB1 = sm.add_constant(XEEB1)

logit_model_EEB1 = sm.Logit(yEEB1, XEEB1)
logit_results_EEB1 = logit_model_EEB1.fit()
print(logit_results_EEB1.summary())

 # Logit EE, B3
yEEB3 = dfEEB3['T']
XEEB3 = dfEEB3[kwh_cols]
XEEB3 = sm.add_constant(XEEB3)

logit_model_EEB3 = sm.Logit(yEEB3, XEEB3)
logit_results_EEB3 = logit_model_EEB3.fit()
print(logit_results_EEB3.summary())

## -----------------------------------------------------------------------------

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

# DEFINE FUNCTIONS -----------------
def ques_recode(srvy):

    DF = srvy.copy()
    import re
    q = re.compile('Question ([0-9]+):.*')
    cols = [unicode(v, errors ='ignore') for v in DF.columns.values]
    mtch = []
    for v in cols:
        mtch.extend(q.findall(v))

    df_qs = Series(mtch, name = 'q').reset_index() # get the index as a variable. basically a column index
    n = df_qs.groupby(['q'])['q'].count() # find counts of variable types
    n = n.reset_index(name = 'n') # reset the index, name counts 'n'
    df_qs = pd.merge(df_qs, n) # merge the counts to df_qs
    df_qs['index'] = df_qs['index'] + 1 # shift index forward 1 to line up with DF columns (we ommited 'ID')
    df_qs['subq'] = df_qs.groupby(['q'])['q'].cumcount() + 1
    df_qs['subq'] = df_qs['subq'].apply(str)
    df_qs.ix[df_qs.n == 1, ['subq']] = '' # make empty string
    df_qs['Ques'] = df_qs['q']
    df_qs.ix[df_qs.n != 1, ['Ques']] = df_qs['Ques'] + '.' + df_qs['subq']

    DF.columns = ['ID'] + df_qs.Ques.values.tolist()

    return df_qs, DF

def ques_list(srvy):

    df_qs, DF = ques_recode(srvy)
    Qs = DataFrame(zip(DF.columns, srvy.columns), columns = [ "recoded", "desc"])[1:]
    return Qs

# df = dataframe of survey, sel = list of question numbers you want to extract free of DVT
def dvt(srvy, sel):

    """Function to select questions then remove extra dummy column (avoids dummy variable trap DVT)"""

    df_qs, DF = ques_recode(srvy)

    sel = [str(v) for v in sel]
    nms = DF.columns

    # extract selected columns
    indx = []
    for v in sel:
         l = df_qs.ix[df_qs['Ques'] == v, ['index']].values.tolist()
         if(len(l) == 0):
            print (bcolors.FAIL + bcolors.UNDERLINE +
            "\n\nERROR: Question %s not found. Please check CER documentation"
            " and choose a different question.\n" + bcolors.ENDC) % v
         indx =  indx + [i for sublist in l for i in sublist]

    # Exclude NAs Rows
    DF = DF.dropna(axis=0, how='any', subset=[nms[indx]])

    # get IDs
    dum = DF[['ID']]
    # get dummy matrix
    for i in indx:
        # drop the first dummy to avoid dvt
        temp = pd.get_dummies(DF[nms[i]], columns = [i], prefix = 'D_' + nms[i]).iloc[:, 1:]
        dum = pd.concat([dum, temp], axis = 1)
        # print dum

        # test for multicollineary

    return dum

def rm_perf_sep(y, X):

    dep = y.copy()
    indep = X.copy()
    yx = pd.concat([dep, indep], axis = 1)
    grp = yx.groupby(dep)

    nm_y = dep.name
    nm_dum = np.array([v for v in indep.columns if v.startswith('D_')])

    DFs = [yx.ix[v,:] for k, v in grp.groups.iteritems()]
    perf_sep0 = np.ndarray((2, indep[nm_dum].shape[1]),
        buffer = np.array([np.linalg.norm(DF[nm_y].values.astype(bool) - v.values) for DF in DFs for k, v in DF[nm_dum].iteritems()]))
    perf_sep1 = np.ndarray((2, indep[nm_dum].shape[1]),
        buffer = np.array([np.linalg.norm(~DF[nm_y].values.astype(bool) - v.values) for DF in DFs for k, v in DF[nm_dum].iteritems()]))

    check = np.vstack([perf_sep0, perf_sep1])==0.
    indx = np.where(check)[1] if np.any(check) else np.array([])

    if indx.size > 0:
        keep = np.all(np.array([indep.columns.values != i for i in nm_dum[indx]]), axis=0)
        nms = [i.encode('utf-8') for i in nm_dum[indx]]
        print (bcolors.FAIL + bcolors.UNDERLINE +
        "\nPerfect Separation produced by %s. Removed.\n" + bcolors.ENDC) % nms

        # return matrix with perfect predictor colums removed and obs where true
        indep1 = indep[np.all(indep[nm_dum[indx]]!=1, axis=1)].ix[:, keep]
        dep1 = dep[np.all(indep[nm_dum[indx]]!=1, axis=1)]
        return dep1, indep1
    else:
        return dep, indep


def rm_vif(X):

    import statsmodels.stats.outliers_influence as smso
    loop=True
    indep = X.copy()
    # print indep.shape
    while loop:
        vifs = np.array([smso.variance_inflation_factor(indep.values, i) for i in xrange(indep.shape[1])])
        max_vif = vifs[1:].max()
        # print max_vif, vifs.mean()
        if max_vif > 30 and vifs.mean() > 10:
            where_vif = vifs[1:].argmax() + 1
            keep = np.arange(indep.shape[1]) != where_vif
            nms = indep.columns.values[where_vif].encode('utf-8') # only ever length 1, so convert unicode
            print (bcolors.FAIL + bcolors.UNDERLINE +
            "\n%s removed due to multicollinearity.\n" + bcolors.ENDC) % nms
            indep = indep.ix[:, keep]
        else:
            loop=False
    # print indep.shape

    return indep


def do_logit(df, tar, stim, D = None):

    DF = df.copy()
    if D is not None:
        DF = pd.merge(DF, D, on = 'ID')
        kwh_cols = [v for v in DF.columns.values if v.startswith('kwh')]
        dum_cols = [v for v in D.columns.values if v.startswith('D_')]
        cols = kwh_cols + dum_cols
    else:
        kwh_cols = [v for v in DF.columns.values if v.startswith('kwh')]
        cols = kwh_cols

    # DF.to_csv("/Users/dnoriega/Desktop/" + "test.csv", index = False)
    # set up y and X
    indx = (DF.tariff == 'E') | ((DF.tariff == tar) & (DF.stimulus == stim))
    df1 = DF.ix[indx, :].copy() # `:` denotes ALL columns; use copy to create a NEW frame
    df1['T'] = 0 + (df1['tariff'] != 'E') # stays zero unless NOT of part of control
    # print df1

    y = df1['T']
    X = df1[cols] # extend list of kwh names
    X = sm.add_constant(X)

    msg = ("\n\n\n\n\n-----------------------------------------------------------------\n"
    "LOGIT where Treatment is Tariff = %s, Stimulus = %s"
    "\n-----------------------------------------------------------------\n") % (tar, stim)
    print msg

    print (bcolors.FAIL +
        "\n\n-----------------------------------------------------------------" + bcolors.ENDC)

    y, X = rm_perf_sep(y, X) # remove perfect predictors
    X = rm_vif(X) # remove multicollinear vars

    print (bcolors.FAIL +
        "-----------------------------------------------------------------\n\n\n" + bcolors.ENDC)

    ## RUN LOGIT
    logit_model = sm.Logit(y, X) # linearly prob model
    logit_results = logit_model.fit(maxiter=10000, method='newton') # get the fitted values
    print logit_results.summary() # print pretty results (no results given lack of obs)


#####################################################################
#                           SECTION 2                               #
#####################################################################

nas = ['', ' ', 'NA'] # set NA values so that we dont end up with numbers and text
srvy = pd.read_csv(root + 'Smart meters Residential pre-trial survey data.csv', na_values = nas)
df2 = pd.read_csv(root + 'data_section2.csv')

# list of questions
qs = ques_list(srvy)
qs.recoded.values

# get dummies
dummies = dvt(srvy, [200, 310, 405])

# run logit, optional dummies
tariffs = [v for v in pd.unique(df2['tariff']) if v != 'E']
stimuli = [v for v in pd.unique(df2['stimulus']) if v != 'E']
tariffs.sort() # make sure the order correct with .sort()
stimuli.sort()

for i in tariffs:
    for j in stimuli:
        do_logit(df2, i, j, D = dummies)
