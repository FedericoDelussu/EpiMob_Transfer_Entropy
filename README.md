## README EpiMob_TransferEntropy

### Libraries 

The libraries employed for the EpiMob_TransferEntropy Project are the ones imported in the script “Scripts/Experiment.py”: 

- os
- inspect 
- sys 
- shutil
- numpy
- matplotlib
- pandas
- datetime
- PyCausality


### Folders of EpiMob_TransferEntropy Project 

The EpiMob_TransferEntropy Project is subdivided in folders which contain synthetic input data, code functions, a script for launching the EpiMob_TransferEntropy experiment and an output folder for collecting the experiment results. 
Here, I give an overview of the folder content, their use and meaning in the experiment pipeline. The subfolders of EpiMob_TransferEntropy project are: 

- 1. Input_Data
- 2. Modules
- 3. Notebooks
- 4. Scripts
- 5. Output_Experiment

#### 1. Input_Data 

Input_Data folder contains four datasets which can be employed as input for EpiMob_TransferEntropy experiments.

The datasets are generated in notebook "01_generate_synthetic_dataset.ipynb" contained in Notebooks folder

Each data-set contains features x and y which are generated with an AutoRegression model and a dummy variable z generated as gaussian noise. 
- y is dependent by x via a coupling factor c.
- As c increases, the dependency of y on x increases. 

We have both a single synthetic time-series and a dataframe of collection of synthetic time-series with low (c=0.1) and high (c=1) coupling factors. 

The data-frame collection is created in order to simulate the paper experiments which are performed on mobility and epidemiologic time-series for different provinces. 

**Single Synthetic Time Series**

The single synthetic time-series has 300 observations with a time-index from 2015-01-01 to 2020-09-24 having weekly frequency. 

The first two time-series datasets are: 

- ‘data_c01.csv’
	- single synthetic time-series of features (x,y,z)
	- the coupling factor c is 0.1 
- ‘data_c1.csv’ 
	- single synthetic time-series of features (x,y,z) 
	- the coupling factor c is 1 

**Collection of Synthetic Time Series**

Each time-series element of the collection is associated with a specific zone. 

In this synthetic data, we have three zones called: zone1, zone2, zone3. 

Each zone time-series is generated according to the same procedure followed for creating a single synthetic time-series. 

- ‘data_multi_c01.csv’
	- data with features (x, y, z, zone)
	- for each zone we have a single synthetic time-series (x,y,z) with x and y generated with coupling factor c= 0.1 
- ‘data_multi_c1.csv’
	- data with features (x, y, z, zone)
	- for each zone we have a single synthetic time-series (x,y,z) with x and y generated with coupling factor c = 1 

### 2. Modules 

The Modules folder contains a module named utils.py which contains functions employed for generating the synthetic dataset and launching the EpiMob_TransferEntropy experiment. 

The functions are provided with detailed description of input arguments and returned variables. 

### 3. Notebooks

The Notebooks folder contains two jupyter notebooks: 
- 01_generate_synthetic_dataset.ipynb: 
generates the synthetic data-sets described in **1. Input_Data**
- 02_launching_experiment_script.ipynb: launches EpiMob_TransferEntropy experiment described in **4. Scripts**.  

### 4. Scripts

The Scripts folder contains the Experiment.py script. 

This Script performs Transfer Entropy experiment over the dataframe of collection of synthetic time-series ‘data_multi_c1.csv’.

With these experiment we can specify:
- the name of the experiment
- the collection of input features
- the collection of output features
- the collection of lag values for which Transfer Entropy can be evaluated
- the N_shuffles: number of shuffled TE estimates to compute in order to evaluate ETE
- the Dates to be selected from the input dataframe
	- in this experiment we employ by default all the study period setting Dates_select = None 

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

### 5. Output_Experiments

Output_Experiments is the destination folder of the results of experiments launched on dataframe of collection of synthetic time-series. 

NB: Launching the experiment requires specifying an experiment name which we refer to as **expname**.


The experiment consists of two intermediary phases in which csv results are saved for each zone:
- **computation of TE estimates**
	- destination folder is  **Output_Experiments/expname_TE/**
- computation of shuffled TE estimates
	- destination folder is  **Output_Experiments/expname_TE_SHUFFLE/**

Once these two phases are completed evaluation of NETE can be performed from TE and shuffled TE estimates. 

**Final result dataset** has path **“Output_Experiments/Results/df_expname.csv”**  






