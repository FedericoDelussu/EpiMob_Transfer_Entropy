## README EpiMob_Transfer_Entropy

### Libraries 

The libraries employed for the EpiMob_Transfer_Entropy Project are the ones imported in the script “Scripts/Experiment.py”: 

- os
- inspect 
- sys 
- shutil
- numpy
- matplotlib
- pandas
- datetime
- PyCausality

### Folders of EpiMob_Transfer_Entropy Project 

The EpiMob_Transfer_Entropy Project is subdivided in folders which contain code functions, a script for launching the EpiMob_Transfer_Entropy experiment and an output folder for collecting the experiment results. 
Here, I give an overview of the folder content, their use and meaning in the experiment pipeline. The subfolders of EpiMob_Transfer_Entropy project are: 

1. Input_Data
2. Modules
3. Scripts
4. Output_Experiments
5. Notebooks


### 1. Input_Data 

Input_Data folder contains two dataframes which record a collection of daily and weekly province level time-series containing mobility and COVID-19 indicators for the four countries France, Austria, Spain, Italy (for Italy we do not have the daily time-series):
- ts_mob_covid_weekly.csv
- ts_mob_covid_daily.csv

The weekly dataframe 'ts_mob_covid_weekly.csv' contains the columns:

- index : weekly date of the record
- prov : name of the province 
- N_pop : number of weekly province users from the population dataset
- N_coloc : number of weekly province users from the colocation dataset
- cases : number of weekly COVID-19 cases in prov
- deaths : number of weekly COVID-19 deaths in prov
- contact_rate : contact rate 
- short_range_movement : short range movement rate 
- mid_range_movement : mid range movement rate
- Country : Country to which the province belongs

The daily dataframe 'ts_mob_covid_daily.csv' contains the columns:

- index : daily date of the record
- prov : name of the province 
- cases : number of weekly COVID-19 cases in prov
- deaths : number of weekly COVID-19 deaths in prov
- short_range_movement : short range movement rate 
- mid_range_movement : mid range movement rate
- Country : Country to which the province belongs

N_pop and N_coloc columns in the weekly dataframe are employed for province sample selection for both weekly and daily experiments. 

### 2. Modules 

The Modules folder contains a module named utils.py which contains functions for launching the EpiMob_TransferEntropy experiment. 
The functions are provided with detailed description of input arguments and returned variables. 
  
### 3. Scripts

The Scripts folder contains the Experiment.py script. 

This Script performs Transfer Entropy experiment over the dataframe of weekly province time-series collection.
For reducing the time of computation the Script launches the experiment only on two provinces of Italy (Torino and Milano). 

With this experiment we can specify:
- the name of the experiment
- the collection of input features
- the collection of output features
- the collection of lag values for which Transfer Entropy can be evaluated
- the N_shuffles: number of shuffled TE estimates to compute in order to evaluate ETE

For each: 
- Z : zone time-series name 
- L : lag value
- (X,Y) : (input,output) pair 

We obtain a  **Z zone EpiMob_TransferEntropy estimate from input X to output Y at a given lag L**. 

The EpiMob_TransferEntropy estimate consists of: 
- entropy components: 
	- H1 : H(Y, X_L, Y_L)
		- where X_L,Y_L are lagged time-series of X and Y at lag L   
	- H2 : H(X_L, Y_L)
	- H3 : H(Y, Y_L)
	- H4 : H(Y_L)
- conditional entropy components:
	- C1 = H3 - H4 : H(Y|Y_L)
	- C2 = H1 - H2 : H(Y|X_L,Y_L)
- transfer entropy estimate:
	- TE = C1 - C2
- ETE   : effective transfer entropy estimate
- p_XY : p-value of ETE estimate
- NETE = ETE/C1 : normalized transfer entropy estimate

**NB**: the script can be edited in order to:
- select weekly or daily dataframe
- select different Countries
- select different lags, input and output features for evaluating the EpiMob_TransferEntropy estimates 

### 4. Output_Experiments

Output_Experiments is the destination folder of the results of experiments launched on the weekly dataframe of collection of time-series. 

NB: Launching the experiment requires specifying an experiment name which we refer to as **expname**.


The experiment consists of two intermediary phases in which csv results are saved for each zone:
- **computation of TE estimates**
	- destination folder is  **Output_Experiments/expname_TE/**
- computation of shuffled TE estimates
	- destination folder is  **Output_Experiments/expname_TE_SHUFFLE/**

Once these two phases are completed evaluation of NETE can be performed from TE and shuffled TE estimates. 

**Final result dataset** has path **“Output_Experiments/Results/df_expname.csv”**  


### 5. Notebooks

The Notebooks folder contains the notebook 'launching_experiment_script.ipynb' which:
- displays the two possible input dataframe; daily and weekly province-level time-series 
- launches EpiMob_Transfer_Entropy experiment described in **3. Scripts**.
- displays the results dataframe contained in Output_Experiments folder


