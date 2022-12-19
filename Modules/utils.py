import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt
from PyCausality.TransferEntropy import *

#FUNCTIONS FOR SYNTHETIC DATA GENERATION 

def AR_step(x,y, c = 0.01):
    """
    x,y : input variables for generating new observations (x_new, y_new)
    c   : coupling factor
    >> as c increases, the more x has a contribute on the evolution of y
    
    return x_new, y_new
    
    the function performs a single iteration of the AR model
    """
    y_new = 0.6*x + c*y + np.random.normal()
    x_new = 0.5*x + np.random.normal()
    
    return x_new, y_new
    
def AR_process(L, c = 0.01):
    """
    L : length of the time-series generated with the AR model from AR_step() function
    c : coupling factor
    
    return ts_process
    
    ts_process: time-series dataframe with L records with columns (x,y,z) 
    > y depends on x via coupling factor c 
    > z is a dummy variable generated from unit gaussian distribution; it has no relationship with x and z 
    """
    
    x_init = np.random.normal()
    y_init = np.random.normal() 
    
    L_process = [ [x_init, y_init] ]
    
    for i in range(L):
            
        x = L_process[-1][0]
        y = L_process[-1][1]
    
        x_new, y_new = AR_step(x,y,c)
        
        L_process.append( [x_new,y_new])
        
    ts_process = pd.DataFrame(L_process, columns= ["x", "y"])
    #joining dummy variable z 
    ts_process["z"] = np.random.normal(size = len(ts_process))
    
    return ts_process

def gen_synthetic_data(length = 1000, c = 0.5):
    """
    length : length of the synthetic dataframe
    c : coupling factor
    
    return data
    
    synthetic dataframe with 
    """

    #number of initial records not considered in the data generating process
    transient = 10000
    #time-series generated according to the AR model
    ts_process = AR_process(transient + length, c = c)
    
    #collecting the AR model generated data after the transient steps
    data = ts_process[-length:]
    data.index = range(len(data))
    
    return data

#FUNCTIONS FOR TRANSFER ENTROPY EXPERIMENTS 

#subsetting a dataframe according to its unique feature values
def subset_df_feature(df, 
                      feature):
    """
    df: input dataframe
    feature: feature of df for which unique values are computed
    
    return DICT_df_feature 

    > DICT_df_feature is a dictionary which assigns each unique feature value to the portion of the dataframe having that feature value
    """
    
    feature_values = df[feature].unique()
    
    DICT_df_feature = {f: df[df[feature] == f] for f in feature_values}
    
    return DICT_df_feature

#setting entropy component computation for a given time-series with X,Y features 
def entropy_components(ts, 
                       X, 
                       Y, 
                       lag = 1, 
                       direction = "xy"):
    """
    ts: time-series dataframe with columns X and Y 
    X   : feature included in TE computation 
    Y   : feature included in TE computation
    lag : lag parameter for TE computation 
    direction: default "xy" computes TE from X to Y feature; if "yx" computes TE from Y to X feature
               
    return TE_df, DICT_HDF
    
    > TE_df is dataframe with features X,Y and also lagged features corresponding to the selected lag value
    >> e.g. if lag=n TE_df features will be ('X','Y', 'X_lagn', 'Y_lagn') 
    
    > DICT_HDF is a dictionary having the entropy component names as keys (H1,H2,H3,H4)
    >> for each entropy component, the corresponding value is the subset of features employed for its computation. 
    >> the subset is selected from set ('X','Y', 'X_lagn', 'Y_lagn') 
    """

    #Object of TransferEntropy class in PyCausality.TransferEntropy module
    TE_object = TransferEntropy(ts[[X,Y]],
                                endog = Y,
                                exog  = X,
                                lag = lag)

    #tsset employed for TE computation
    #it includes X,Y and their corresponding lagged features
    TE_df = TE_object.df[0]

    #Dictionary which associates entropy component names to the feature subsets employed for their computation 
    DICT_HDF = {"H1" : [Y, Y + "_lag" + str(lag), X + "_lag" + str(lag)],
                "H2" : [Y + "_lag" + str(lag), X + "_lag" + str(lag)],
                "H3" : [Y, Y + "_lag" + str(lag)],
                "H4" : [Y + "_lag" + str(lag)]}

    
    if direction == "yx":

        DICT_HDF = {"H1" : [X, X + "_lag" + str(lag), Y + "_lag" + str(lag)],
                    "H2" : [X + "_lag" + str(lag), X + "_lag" + str(lag)],
                    "H3" : [X, X + "_lag" + str(lag)],
                    "H4" : [X + "_lag" + str(lag)]}


    return TE_df, DICT_HDF

#estimate Tranfer Entropy with Kernel Density Estimation
def TE_estimate_KDE(ts, 
                 X, 
                 Y, 
                 lag = 1, 
                 direction = "xy", 
                 bandwidth = "scott", 
                 gridpoints = 20):
    """
    ts : time-series dataframe with columns X and Y 
    X    : feature included in TE computation 
    Y    : feature included in TE computation
    lag  : lag parameter for TE computation 
    direction : default "xy" computes TE from X to Y feature; if "yx" computes TE from Y to X feature
    bandwidth   : method for inferring bandwidth for kernel density estimation (deafult is 'scott')
    gridpoints  : number of grid points per dimension employed in kernel density estimation 
    
    return TE, dict_Hterms 
    
    > TE : Tranfer Entropy estimation 
    >> TE is evaluated from X to Y if direction is "xy"; otherwise from Y to X)
    >> TE is evaluated as a combination of entropy component terms: (H3-H4) - (H1-H2)
    
    > dict_Hterms : dictionary with entropy component values
    >> the entropy component values are computed with kernel density estimation method
    >> the bandwidth of the kernel is estimated with "scott" method, according to parameter bandwidth = "scott"
    >> the gridpoints refer to the number of points by which each feature domain is uniformly discretized in order to compute the entropy from the discretized kernel density estimate
    """

    TE_df, DICT_HDF = entropy_components(ts, X, Y, lag = lag, direction = direction)

    estim = "kernel"

    H1 = get_entropy(TE_df[DICT_HDF["H1"]],
                     gridpoints = gridpoints,
                     bandwidth  = bandwidth,
                     estimator  = estim,  
                     covar = None)

    H2 = get_entropy(TE_df[DICT_HDF["H2"]],
                     gridpoints = gridpoints,
                     bandwidth  = bandwidth,
                     estimator = estim,
                     covar = None)

    H3 = get_entropy(TE_df[DICT_HDF["H3"]],
                     gridpoints = gridpoints,
                     bandwidth  = bandwidth,
                     estimator = estim,
                     covar = None)

    H4 = get_entropy(TE_df[DICT_HDF["H4"]],
                     gridpoints = gridpoints,
                     bandwidth  = bandwidth,
                     estimator  = estim,
                     covar = None)

    dict_Hterms = {"H1": H1,
                   "H2": H2,
                   "H3": H3,
                   "H4": H4}

    TE = (H3 - H4) - (H1 - H2)

    return TE, dict_Hterms


def TE_estimate_KDE_df(ts,
                       exog  = "X",
                       endog = "Y",
                       lag   = 1,
                       bandwidth  = "scott",
                       gridpoints = 20,
                       n_shuffles = 100,
                       half = True):
    """
    ts:  time-series dataframe including exog and endog features
    exog:  exogenous variable included in TE computation
    endog: exogenous variable included in TE computation
    lag  : lag parameter for TE computation 
    bandwidth   : method for inferring bandwidth for kernel density estimation (deafult is 'scott')
    gridpoints  : number of grid points per dimension employed in kernel density estimation 
    
    n_shuffles: parameter of nonlinearTE function of TransferEntropy class from PyCausality
    >> it is used for performing significance testing over the TE estimate
    half: if True computes TE only from exog to endog variable
   
    return df_result
    > with half=True 
    >> we compute TE in only one direction (exog to endog); therefore halving the computational time required fot its computation 
    >> the df_result dataframe has columns ('X','Y', 'TE_XY') and contains the estimate of TE from exog variable to endog variable  
    >> the df_result dataframe has a single row with (exog, endog, TE estimate from exog to endog) record

    > with half=False 
    >> TE is estimated in both directions with nonlinearTE function of TransferEntropy class from PyCausality
    >> it can be used to perform consistency check of the two procedures for TE estimation
    >> but for experiments we always use half = False 
    """

    if half:
        
        TE_XY, dict_Hterms = TE_estimate_KDE(ts,
                                             X=exog,
                                             Y=endog,
                                             lag = lag,
                                             direction = "xy",
                                             bandwidth = bandwidth,
                                             gridpoints = gridpoints)
   
        df_result = pd.DataFrame({"X":exog, "Y":endog, "TE_XY":TE_XY}, index=[''])
        return df_result

    else:
        causality = TransferEntropy(ts,
                                    endog = endog, 
                                    exog  = exog,  
                                    lag = lag)     

        causality.nonlinear_TE(pdf_estimator = 'kernel',
                               gridpoints = gridpoints,
                               bandwidth  = bandwidth,
                               n_shuffles = n_shuffles)

        df_result = causality.results

        df_result["X"] = exog  
        df_result["Y"] = endog 

        return df_result

def TE_estimate_shuffled_KDE_df(ts,
                                exog  = "X",
                                endog = "Y",
                                lag   = 1,
                                bandwidth  = "scott",
                                gridpoints = 20,
                                n_shuffles = 100,
                                ETE_shuffles = 100,
                                shuffle_y = False,
                                half = True):
    """
    ts:  time-series dataframe including exog and endog features
    exog:  exogenous variable included in TE computation
    endog: exogenous variable included in TE computation
    lag  : lag parameter for TE computation 
    bandwidth   : method for inferring bandwidth for kernel density estimation (deafult is 'scott')
    gridpoints  : number of grid points per dimension employed in kernel density estimation 
    n_shuffles: parameter of nonlinearTE function of TransferEntropy class from PyCausality
    >> it is not employed if half=True
    ETE_shuffles: number of times for which TE is iteratively estimated on suffled time-series.
    >> the shuffled estimates are subsequently used for ETE estimate
    shuffle_y: False
    >> in shuffling of ts time-series only exog variable is shuffled by default
    >> if shuffle_y=True also the endog variable is shuffled
    half: if True computes TE only from exog to endog variable
        
    return df_result_shuffle
    
    > df_result_shuffle: dataframe with columns ('X','Y', 'TE_XY') 
    >> the number of records is equal to ETE_shuffle
    >> each record corresponds to a TE estimate over the shuffled time-series 
    """
    
    result_shuffle = []

    for i in range(ETE_shuffles):

        ts_shuffled = pd.DataFrame(ts, copy = True)

        x = ts[exog].values
        np.random.shuffle(x)
        ts_shuffled[exog] = x

        if shuffle_y:
            y = ts[endog].values
            np.random.shuffle(y)
            ts_shuffled[exog] = y


        df_result = TE_estimate_KDE_df(ts_shuffled,
                                       exog  = exog,
                                       endog = endog,
                                       lag   = lag,
                                       bandwidth  = bandwidth,
                                       gridpoints = gridpoints,
                                       n_shuffles = n_shuffles,
                                       half = half)

        result_shuffle.append(df_result)

    df_result_shuffle = pd.concat(result_shuffle, axis = 0)

    return df_result_shuffle


def ETE_estimate_KDE_df(ts_,
                        exog  = "X",
                        endog = "Y",
                        lag   = 1,
                        bandwidth  = "scott",
                        gridpoints = 20,
                        n_shuffles = 100,
                        ETE_shuffles = 100,
                        shuffle_y = False,
                        half = True):
    """
    ts:  time-series dataframe including exog and endog features
    exog:  exogenous variable included in TE computation
    endog: exogenous variable included in TE computation
    lag  : lag parameter for TE computation 
    bandwidth   : method for inferring bandwidth for kernel density estimation (deafult is 'scott')
    gridpoints  : number of grid points per dimension employed in kernel density estimation 
    n_shuffles: parameter of nonlinearTE function of TransferEntropy class from PyCausality
    >> it is not employed if half=True
    ETE_shuffles: number of times for which TE is iteratively estimated on suffled time-series.
    >> the shuffled estimates are subsequently used for ETE estimate
    shuffle_y: False
    >> in shuffling of ts time-series only exog variable is shuffled by default
    >> if shuffle_y=True also the endog variable is shuffled
    half: if True computes TE only from exog to endog variable
    
    return df_TE, df_ETE
    
    > df_TE 
    >> dataframe with columns (X,Y,TE_XY); has a single record with TE estimate from exog to endog variable
    > df_TE_SHUFFLE
    >> dataframe with columns (X,Y,TE_XY); has a collection of records with TE estimates over shuffled time-series 
    > df_ETE 
    >> dataframe with column ETE_XY, has a single record with Effective Transfer Entropy estimate computed by subtracting TE estimate (from df_TE) from average of collection of shuffled TE estimates (df_TE_SHUFFLE)  
    
    >> these three data-sets are subsequently used for evaluating p-value of ETE estimate
    """

    ts = pd.DataFrame(ts_, copy=True)

    df_TE = TE_estimate_KDE_df(ts,
                               exog  = exog,
                               endog = endog,
                               lag   = lag,
                               bandwidth  = bandwidth,
                               gridpoints = gridpoints,
                               n_shuffles = n_shuffles,
                               half = half)

    df_TE_SHUFFLE = TE_estimate_shuffled_KDE_df(ts,
                                 exog  = exog,
                                 endog = endog,
                                 lag   = lag,
                                 bandwidth  = bandwidth,
                                 gridpoints = gridpoints,
                                 n_shuffles = n_shuffles,
                                 ETE_shuffles = ETE_shuffles,
                                 shuffle_y = shuffle_y,
                                 half = half)

    df_TE_SHUFFLE_MEAN = df_TE_SHUFFLE.mean(axis=0)
    df_TE_SHUFFLE_MEAN = pd.DataFrame(df_TE_SHUFFLE_MEAN).T

    #Effective Transfer Entropy 
    df_ETE = pd.DataFrame(df_TE, copy = True)[["TE_XY"]]
    df_ETE.iloc[0,0] = df_ETE.iloc[0,0] - df_TE_SHUFFLE_MEAN.loc[0,"TE_XY"]
    df_ETE.columns = ["ETE_XY"]
    
    return df_TE, df_TE_SHUFFLE, df_ETE


def EXP_TE(df_,                                              
           Input_features,                                   
           Output_features,    
           LAGS = [1],                                       
           gridpoints = 20,                                  
           Dates_select = None,
           zone_name = "zone",
           FOLD_save = "../Output_Experiments/fold_save/"):  
    """
    df_: input dataframe containing a collection of time-series 
    Input_features : collection of exogenous variables for which TE is estimated
    Output_features : collection of endogenous variables for which TE is estimated
    LAGS           : collection of lag parameters for TE computation   
    ETE_SHUFFLES : number of shuffled TE estimates employed for performing ETE estimation
    >> this function is part of the pipeline and ETE_SHUFFLES is set to 1 
    gridpoints   : number of grid points per dimension employed in kernel density estimation 
    Dates_select : list of dates to select from records of df_
    >> it can be useful if it is needed to perform experiments in a more narrow study period
    zone_name: name of the feature which defines the elements of the collection of time-series which make up df_
    FOLD_save: fold in which TE estimates are stored
    >> TE estimates are stored in FOLD_TE = FOLD_save + name_exp + "_" + "TE/"

   
    return df_results
    
    > this function saves iteratively for each zone the corresponding experiment results on zone time-series 
    
    > the experiment results collects in a dataframe the TE_XY estimates for each tuple (X,Y,L) where:
        - X : entry from Input_features list
        - Y : entry from Output_features list
        - L : entry from LAGS list
    
    > zone experiment results are collected into a single dataframe df_results which is returned by the function 
    
    > df_results collects results for each zone time-series and has columns:
    'X', 'Y' : features for which TE is evaluated from 'X' to 'Y'
    'TE_XY'  : Transfer Entropy 
    'zone' : time-series zone name 
    'LAG': lag values for which TE is estimated 
    'H1', 'H2', 'H3', 'H4' : entropy components of TE
    'C1': H3 - H4
    'C2': H1 - H2
    """

    df = pd.DataFrame(df_, copy = True)
    
    #filtering dates
    if Dates_select is not None:
        df  = df.loc[[d in Dates_select for d in df.index]]
    
    #subsetting dataframe by zone names; each zone is associated to a time-series 
    DICT_df_zone = subset_df_feature(df, feature = zone_name)
    Zones = list(DICT_df_zone.keys())
    
    if FOLD_save is not None: 
        #zones for which we already have experiments
        Zones_exp = os.listdir(FOLD_save)
        Zones_exp = [p.split(".csv")[0] for p in Zones_exp]
        Zones = np.setdiff1d(Zones,Zones_exp)
    
    #results dataframe containing results for al zone time-seires 
    df_results = []
   
    for zone in np.sort(Zones):

        df_zone = []
        
        print("\t computing TE for " + zone)

        for LAG in LAGS:

            print("\t\t setting lag value at " + str(LAG))
            
            for X in Input_features:

                for Y in Output_features:
                    
                    print("\t\t\t computing TE from " + X + " to " + Y)

                    ts = DICT_df_zone[zone][[X,Y]]

                    #in this procedure step we only need TE estimates
                    df_TE, _, _ = ETE_estimate_KDE_df(ts,
                                                      exog  = X,
                                                      endog = Y,
                                                      lag   = LAG,
                                                      bandwidth = "scott",
                                                      gridpoints = gridpoints,
                                                      n_shuffles = 0,
                                                      ETE_shuffles = 1, #shuffled estimates are not used
                                                      half = True)

                    
                    #joining experiment features
                    df_TE["zone"] = zone
                    df_TE["LAG"] = LAG

                    #JOINING TE_XY ENTROPY COMPONENTS
                    TE, dict_Hterms = TE_estimate_KDE(ts,
                                                      X, Y,
                                                      lag = LAG,
                                                      direction = "xy",
                                                      bandwidth = "scott",
                                                      gridpoints = 20)
   
                    H1 = dict_Hterms["H1"]
                    H2 = dict_Hterms["H2"]
                    H3 = dict_Hterms["H3"]
                    H4 = dict_Hterms["H4"]

                    dict_TE_xy_terms = {"H1" : H1,
                                        "H2" : H2,
                                        "H3" : H3,
                                        "H4" : H4,
                                        "C1" : H3-H4,
                                        "C2" : H1-H2}


                    for c in ["H1","H2","H3","H4","C1","C2"]:
                        c_val = dict_TE_xy_terms[c]
                        df_TE[c] = c_val

                    df_zone.append(df_TE)
                    df_results.append(df_TE)

        df_zone = pd.concat(df_zone, axis=0)
    
        #save zone results 
        if FOLD_save is not None: 
            df_zone.to_csv(FOLD_save +zone +".csv")
    
    df_results = pd.concat(df_results, axis=0)
    
    return df_results 

def EXP_TE_shuffle(df_,                                             
                   Input_features,                                  
                   Output_features,   
                   LAGS = [1],                                     
                   ETE_SHUFFLES = 20,                               
                   gridpoints = 20,                                 
                   Dates_select = None,
                   zone_name = "zone",
                   FOLD_save = "../Output_Experiments/fold_save/"):
    """
    df_: input dataframe containing a collection of time-series 
    Input_features : list of exogenous variables for which TE is estimated
    Output_features : list of endogenous variables for which TE is estimated
    LAGS           : list of lag parameters for TE computation   
    ETE_SHUFFLES : number of shuffled TE estimates employed for performing ETE estimation
    gridpoints   : number of grid points per dimension employed in kernel density estimation 
    Dates_select : list of dates to select from records of df_
    >> it can be useful if it is needed to perform experiments in a more narrow study period
    zone_name: name of the feature which defines the elements of the collection of time-series which make up df_
    FOLD_save: fold in which TE estimates are stored
    >> TE SHUFFLED estimates are stored in FOLD_TE = FOLD_save + name_exp + "_" + "TE_SHUFFLE/"

   
    return df_results
    
    > this function saves iteratively for each zone the corresponding experiment results on zone time-series 
    
    > the experiment results collects in a dataframe the TE_XY shuffled estimates for each tuple (X,Y,L) where:
        - X : entry from Input_features list
        - Y : entry from Output_features list
        - L : entry from LAGS list
        
    > for each (X,Y,L) tuple we have ETE_SHUFFLES TE_XY shuffled estimates
    
    > zone experiment results are collected into a single dataframe df_results which is returned by the function 
    
    > df_results collects results for each zone time-series and has columns:
    'X', 'Y' : features for which TE is evaluated from 'X' to 'Y'
    'TE_XY'  : Transfer Entropy 
    'zone' : time-series zone name 
    'LAG': lag values for which TE is estimated     
    """

    df = pd.DataFrame(df_,copy = True)
    if Dates_select is not None:
        df  = df.loc[[d in Dates_select for d in df.index]]

    DICT_df_zone = subset_df_feature(df, feature = zone_name)
    Zones = list(DICT_df_zone.keys())

    if FOLD_save is not None:
        #zones for which we already have experiments
        Zones_exp = os.listdir(FOLD_save)
        Zones_exp = [p.split(".csv")[0] for p in Zones_exp]
        #remove zones for which we already have experiments
        Zones = np.setdiff1d(Zones,Zones_exp)

    #results dataframe containing results for al zone time-seires 
    df_results = []    
    
    for zone in np.sort(Zones):

        df_zone = []
        
        print("\t computing SHUFFLED TE estimates for " + zone)

        for LAG in LAGS:

            print("\t\t setting lag value at " + str(LAG))
            
            for X in Input_features:

                for Y in Output_features:
                    
                    print("\t\t\t computing " +str(ETE_SHUFFLES)+ " TE SHUFFLED estimates from " + X + " to " + Y)
                    
                    ts = DICT_df_zone[zone][[X,Y]]

                    df_TE, df_TE_SHUFFLE, df_ETE = ETE_estimate_KDE_df(ts,
                                                                       exog  = X,
                                                                       endog = Y,
                                                                       lag   = LAG,
                                                                       bandwidth = "scott",
                                                                       gridpoints = gridpoints,
                                                                       n_shuffles = 0,
                                                                       ETE_shuffles = ETE_SHUFFLES)
                    
                    
                    df_TE_SHUFFLE["zone"] = zone
                    df_TE_SHUFFLE["LAG"] = LAG

                    df_TE_SHUFFLE["X"] = X
                    df_TE_SHUFFLE["Y"] = Y

                    df_zone.append(df_TE_SHUFFLE)
                    df_results.append(df_TE_SHUFFLE)

        df_zone = pd.concat(df_zone, axis=0)

        if FOLD_save is not None: 
            df_zone.to_csv(FOLD_save + zone +".csv")
            
    df_results = pd.concat(df_results, axis=0)
            
    return df_results

def update_shuffle(df_zone_TE_v0, 
                   df_zone_SHUFFLE, 
                   zone_name = "zone"):
    """
    df_zone_TE_v0   : zone result dataframe with TE estimates obtained from running EXP_TE() function
    df_zone_SHUFFLE : zone result dataframe with TE shuffled estimates obtained from running EXP_TE_SHUFFLE() function
    zone_name: zone name of time-series 
    
    return df_zone_TE
    
    this function updates df_zone_TE_v0 by adding columns:
    - ETE_XY : computed by employing shuffled TE estimates contained in dataframe df_zone_SHUFFLE
    - p_XY : p-value of ETE_XY estimate by computing the percentage of shuffled estimates having a value greater than the original TE_XY estimate contained in df_zone_TE_v0
    - NETE_XY: computed as the ratio ETE_XY/C1
    
    >> where C1 is a conditional entropy term exposed in EXP_TE() description
    """

    df_zone_TE = pd.DataFrame(df_zone_TE_v0, copy = True)

    List_exp = df_zone_TE[[zone_name,"X","Y","LAG"]].values

    for i, L in zip(df_zone_TE.index, List_exp):

        #Original TE estimates
        TE_XY = df_zone_TE.loc[i,"TE_XY"]

        #selecting TE shuffled estimates for (p,X,Y,L) tuple
        p, X, Y, Lag = tuple(L)
        ind_p = df_zone_SHUFFLE[zone_name] == p #zone selection
        ind_X = df_zone_SHUFFLE["X"]    == X    #X selection
        ind_Y = df_zone_SHUFFLE["Y"]    == Y    #Y selection
        ind_L = df_zone_SHUFFLE["LAG"]  == Lag  #Lag selection
        df_pXYL = df_zone_SHUFFLE[ ind_p & ind_X & ind_Y & ind_L][["TE_XY"]]#,"TE_YX"]]

        #Effective transfer entropy
        ETE_XY = TE_XY - df_pXYL.mean(axis=0)["TE_XY"]
        
        #p-value of ETE_XY estimate
        p_XY = np.sum(df_pXYL["TE_XY"] > TE_XY)/len(df_pXYL)

        df_zone_TE.loc[i,"ETE_XY"] = ETE_XY
        df_zone_TE.loc[i,"p_XY"]   = p_XY

    #Normalized effective transfer entropy
    df_zone_TE["NETE_XY"] = df_zone_TE["ETE_XY"]/df_zone_TE["C1"]

    return df_zone_TE

def pipeline_exp(df_,
                 EXP_step,
                 name_exp     = "new_exp",
                 Input_features = ['x'],
                 Output_features      = ['y'],
                 LAGS_        = [2,3,4,5,6,7,8],
                 N_shuffles   = 500,
                 Dates_select = None,
                 FOLD_save = "../Output_Experiments/"):
    """
    df_ : input dataframe 
    EXP_step         : step of the experiment pipeline
        - TE         : TE computation
        - TE_SHUFFLE : TE SHUFFLED computation (saved in a dataframe)
        - NETE       : NETE computation from TE and TE_SHUFFLE    
    name_exp         : name of experiment folder
    Input_features   : input signals for TE computation
    Output_features          : output signals for TE computation
    LAGS_            : lags values for computation
    N_shuffles       : used in "TE_SHUFFLE" step, number of iterations for shuffling the time-series
    Dates_select : list of dates to select from records of df_
    FOLD_save        : folder containing experiment results 
    """


    zones = df_["zone"].dropna().unique()
    
    if EXP_step == "TE":

        try:
            os.mkdir(FOLD_save + name_exp + "_TE")
        except:
            print("WARNING: TE estimate folder already existing or not constructed")
        
        df_results = EXP_TE(df_,                                           
                            Input_features,                                
                            Output_features,                                       
                            gridpoints = 20,                               
                            LAGS = LAGS_,                                    
                            Dates_select = Dates_select,
                            zone_name = "zone",
                            FOLD_save = FOLD_save + name_exp + "_TE/")

    if EXP_step == "TE_SHUFFLE":
        try:
            os.mkdir(FOLD_save + name_exp + "_TE_SHUFFLE/")
        except:
            print("WARNING: TE SHUFFLED estimate folder already existing or not constructed")

        df_results = EXP_TE_shuffle(df_,                                           
                                    Input_features,                                
                                    Output_features,                                       
                                    ETE_SHUFFLES = N_shuffles,                             
                                    gridpoints = 20,                               
                                    LAGS = LAGS_,                                    
                                    Dates_select = Dates_select,
                                    zone_name = "zone",
                                    FOLD_save = FOLD_save + name_exp + "_TE_SHUFFLE/")

    if EXP_step == "NETE":

        F = FOLD_save + name_exp + "_TE/"
        df_zone_TE_v0 = pd.concat( [pd.read_csv(F+f,index_col=0) for f in os.listdir(F) if ".ipynb" not in f], axis= 0)
        df_zone_TE_v0.index = range(len(df_zone_TE_v0.index))

        F = FOLD_save + name_exp + "_TE_SHUFFLE/"
        df_zone_SHUFFLE = pd.concat( [pd.read_csv(F+f,index_col=0) for f in os.listdir(F)], axis= 0)
        df_zone_SHUFFLE.index = range(len(df_zone_SHUFFLE.index))
        df_zone_SHUFFLE["TE_XY"] = df_zone_SHUFFLE["TE_XY"].astype(float)

        df_OUTPUT = update_shuffle(df_zone_TE_v0, df_zone_SHUFFLE)

        #SHUFFLED TE EXPERIMENT
        try:
            os.mkdir(FOLD_save + "Results/")
        except:
            print("")
        df_OUTPUT.to_csv(FOLD_save+ "Results/df_" + name_exp + ".csv")