import os
import inspect 
import sys 
import shutil

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt

from PyCausality.TransferEntropy import *

#importing functions for running the experiments
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 
from Modules import utils

#NAME OF THE EXPERIMENT

print("\nThis script takes in input a bivariate time-series in a csv format")
print("The columns of the csv must be: \n index: timestamp \n column1 : zone; can be a single zone \n or we can have multiple zones such that each one of them is associated to the same set of timestamps \n column2: x \n column3: y")
print("\nThe script will generate a dataframe of NETE estimates from x to y for the different zones of the dataset")

path = input('\nSelect path of bivariate time-series: ')
LAG = input('\nSelect lag of the experiment: ')
N_shuffles = input('\nSelect N_shuffles of the experiment: ')
name_exp = input('\nSelect name of the experiment folder which will be generated: ')

df_input = pd.read_csv(path, index_col = 0)
#fillnan values of input dataframe
df_input = df_input.fillna(0)


#PATHS CONTAINING EXPERIMENT RESULTS 

#folder containing the results for all the experiments 
FOLD_save = "../Output_Experiments/"
#folder containing TE estimates for each zone
FOLD_TE = FOLD_save + name_exp + "_" + "TE/"
#folder containing TE estimates on shuffled time-series for each zone
FOLD_TE_SHUFFLE = FOLD_save + name_exp + "_" + "TE_SHUFFLE/"
#path to final results
PATH_FINAL_RESULTS = FOLD_save + "Results/df_" + name_exp + ".csv"



#CHECK IF EXPERIMENT HAS BEEN ALREADY LAUNCHED
if os.path.exists(PATH_FINAL_RESULTS):
    print("WARNING: the experiment has been already performed and there is a result dataset in path: " + PATH_FINAL_RESULTS)
    print("If you want to eliminate the experiment results run this commands \n")
    print("shutil.rmtree('" + FOLD_TE + "')")
    print("shutil.rmtree('" + FOLD_TE_SHUFFLE + "')") 
    print("os.remove('" + PATH_FINAL_RESULTS + "')")
    sys.exit() 

#DESCRIPTION OF STEPS OF THE PIPELINE 
DICT_Exp_description = {"TE": "\n 1) Computation of TE estimates for each zone \n csv results for each zone will be saved in " + FOLD_TE + "\n",
                        "TE_SHUFFLE": "\n 2) Computation of TE estimates on shuffled time series for each zone \n csv results for each zone will be saved in " + FOLD_TE_SHUFFLE + "\n", 
                        "NETE": "\n 3) Computation of NETE estimates for each zone \n a single csv result for the whole collection of zones will be saved in " + PATH_FINAL_RESULTS}

#STEPS OF THE EXPERIMENT 
Exp_steps = ["TE", 
             "TE_SHUFFLE", 
             "NETE"]

#RUNNING THE EXPERIMENT PIPELINE
for exp_step in Exp_steps: 
    
    print(DICT_Exp_description[exp_step])
    
    utils.pipeline_exp(df_input,
                       name_exp        = name_exp,
                       Input_features  = ['x'],
                       Output_features = ['y'],
                       LAGS_           = [int(LAG)],
                       EXP_step        = exp_step,
                       N_shuffles      = int(N_shuffles),
                       zone_name       = 'zone',
                       FOLD_save       = FOLD_save)