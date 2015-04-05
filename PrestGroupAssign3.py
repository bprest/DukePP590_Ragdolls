from __future__ import division
from pandas import Series, DataFrame
import pandas as pd
import numpy as np
import statsmodels.api as sm

#main_dir = u'C:/Users/Brianprest/OneDrive/Grad School/2015-Spring/Big Data/data/raw/3_task_data/'
main_dir = u'C:/Users/bcp17/OneDrive/Grad School/2015-Spring/Big Data/data/raw/3_task_data/'
root = main_dir
allocation = "allocation_subsamp.csv"
kwh = "kwh_redux_pretrial.csv"
survey = "Smart meters Residential pre-trial survey data.csv"
data_sec2 = "data_section2.csv"

df_assign = pd.read_csv(root + allocation, header=0)

tariffs = [v for v in pd.unique(df_assign['tariff']) if v!='E']
stimuli = [v for v in pd.unique(df_assign['stimulus']) if v!='E']
tariffs.sort()
stimuli.sort()

control_ids = df_assign.ID[(df_assign['tariff']=='E') & (df_assign['stimulus']=='E')]
A1_ids = df_assign.ID[(df_assign['tariff']=='A') & (df_assign['stimulus']=='1')]
A3_ids = df_assign.ID[(df_assign['tariff']=='A') & (df_assign['stimulus']=='3')]
B1_ids = df_assign.ID[(df_assign['tariff']=='B') & (df_assign['stimulus']=='1')]
B3_ids = df_assign.ID[(df_assign['tariff']=='B') & (df_assign['stimulus']=='3')]

np.random.seed(seed=1789)

selection_control = np.random.choice(control_ids,size=300,replace=False).tolist()
selection_A1 = np.random.choice(A1_ids,size=150,replace=False).tolist()
selection_A3 = np.random.choice(A3_ids,size=150,replace=False).tolist()
selection_B1 = np.random.choice(B1_ids,size=50,replace=False).tolist()
selection_B3 = np.random.choice(B3_ids,size=50,replace=False).tolist()

selection_all = pd.DataFrame(selection_control+selection_A1+selection_A3+selection_B1+selection_B3, columns=["ID"])

df = pd.read_csv(root + kwh, header=0)
# Merge on selected IDs
df = pd.merge(selection_all,df)
# Generate Month/Year variable


# Sum monthly consumption
monthgrp = df.groupby(['ID','month']) # only 1 year in the data, so we can just group by month
df = monthgrp['kwh'].sum().reset_index()

# Pivot to wide on monthly kwh
df['kwh_mon'] = 'kwh_' + df['month'].apply(str)

df_piv = df.pivot('ID','kwh_mon','kwh') # i,j,v. i is rows. j is columns, v is values to go in j columns.
df_piv.reset_index(inplace=True)
df_piv.columns.name = None # gets rid of top left thing

# Merge with treatment assignment
df_piv = pd.merge(df_assign,df_piv)

# Note that "code" is a constant =1 here, so we don't need to add a constant. just rename code "constant".
df_piv.rename(columns ={'code':'constant'}, inplace=True)

# Set up X variables
X_E  = df_piv[(df_piv.tariff=='E') & (df_piv.stimulus=='E')][[1]+range(4,10)] # columns 1 plus 4 thru 9
X_A1 = df_piv[(df_piv.tariff=='A') & (df_piv.stimulus=='1')][[1]+range(4,10)]
X_A3 = df_piv[(df_piv.tariff=='A') & (df_piv.stimulus=='3')][[1]+range(4,10)]
X_B1 = df_piv[(df_piv.tariff=='B') & (df_piv.stimulus=='1')][[1]+range(4,10)]
X_B3 = df_piv[(df_piv.tariff=='B') & (df_piv.stimulus=='3')][[1]+range(4,10)]

# Set up Y variable: 0 if control, 1 if treatment
y_E  = pd.DataFrame([0]*len(X_E), columns=['trt'])
y_A1 = pd.DataFrame([1]*len(X_A1), columns=['trt'])
y_A3 = pd.DataFrame([1]*len(X_A3), columns=['trt'])
y_B1 = pd.DataFrame([1]*len(X_B1), columns=['trt'])
y_B3 = pd.DataFrame([1]*len(X_B3), columns=['trt'])

# Run logit.
logit_model_A1 = sm.Logit(y_E.append(y_A1).reset_index(drop=True),X_E.append(X_A1).reset_index(drop=True))
logit_model_A3 = sm.Logit(y_E.append(y_A3).reset_index(drop=True),X_E.append(X_A3).reset_index(drop=True))
logit_model_B1 = sm.Logit(y_E.append(y_B1).reset_index(drop=True),X_E.append(X_B1).reset_index(drop=True))
logit_model_B3 = sm.Logit(y_E.append(y_B3).reset_index(drop=True),X_E.append(X_B3).reset_index(drop=True))

logit_results_A1 = logit_model_A1.fit()
logit_results_A3 = logit_model_A3.fit()
logit_results_B1 = logit_model_B1.fit()
logit_results_B3 = logit_model_A3.fit()

# Print results
print(logit_results_A1.summary())
print(logit_results_A3.summary())
print(logit_results_B1.summary())
print(logit_results_B3.summary())

#####################################################################
#                    DEFINE FUNCTIONS                               #
#####################################################################


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

#    DF.to_csv("/Users/dnoriega/Desktop/" + "test.csv", index = False)
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
df = pd.read_csv(root + data_sec2)

# list of questions
qs = ques_list(srvy)

# get dummies
dummies = dvt(srvy, [200, 310, 405])

# run logit, optional dummies
for i in tariffs:
        for j in stimuli:
            do_logit(df, i, j, D = dummies)

# Questions to consider:
# 310 - employment
# 405 - Internet access (could affect treatment through technical constraints)
# 450 - Kind of house (apartment, detached, semi-deteached, etc.)
# 452 - own or rent?

