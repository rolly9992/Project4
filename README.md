# Data Science Project #4 – SPY vs SPY comparing investing directly in the SPY vs picking individual stocks that make up the SPY.


## The Dataset

For this project, I am pulling data from Alpha Vantage. 
While there is a free tier, it is limited to only 25 pulls per day. To pull the entire data set, there is a $50 per month charge. So to pull everything you’d have to sign up for at least 1 month. 


Their website is here: :https://www.alphavantage.co/



## Acknowledgements
 
I used multiple references in this project, some snippets from stack overflow, the alpha vantage website for snippets on pulling data in python, the corporate financial institute for their definitions of ratios, along with some other sources. The full list of references is located in the following file located in this repository: 

Project4/references/Sources.txt 

 

## Motivation

The motivation for this study was to fill a requirement for a Udacity course. A secondary reason was that I had personal interest to see if there was any value in attempting to pick individual stocks on a quarterly basis using an algorithm. Perhaps it could help over the long term with saving for retirement. 


## Files Used

There are quite a few files and directories in for this project. These are outlined by folder name, then files are subcategories under the directory. Important note: there are several output files in Excel format. 

A) The data folder 
	
	1) .ipynb_checkpoints
		these are automatically saved files. I left them alone
            
	2) BalanceSheets – the set of files pulled for individual company balance sheets. These are all 		 in cvs format and have the same structure.  One file per company. These are the companies 		 that make up the S&P 500. 
	
	3) CashFlowStatements – similar to the BalanceSheets folder, this folder has individual 	 	CashFlowStatements, one for each company making up the S&P 500. All are in csv
	format and have the same structure. 
   
	4) IncomeStatements – similar to the BalanceSheets and CashFlowStatements folders, this 	folder has individual IncomeStatements, one for each company making up the S&P 500.
	All are in csv format and have the same structure


	5) TimeSeriesAdjusted – this folder houses stock history of adjusted close price for the 	individual stocks that make up the S&P 500. These files are also in csv format, but have a 	different 		structure than the 3 folders with the financial data. We’ll wrangling these a bit later.



	exploring_data.ipynb – a jupyter notebook used to explore the data and use some visuals in the blog article about this project 

	pull_data.py – a python script that calculates which tickers currently make up the spy, then pulls the financial data for those ticker, along with the time series data. It also pulls the time series data for 	the SPY itself. 

	sp500_tickerlist.xlsx – this is an output file from the first script. It’s just a list of the tickers that make up the SPY. 


B) wrangling folder 
	
 	format_raw_data.py – this script takes all the data pulled in the data folder and consolidates it into quarterly data, add interactive fields, filling in NULL values and prepping for machine learning. 

	Consolidate_BalanceSheet_Data.xlsx – an output file from the above script. 

	Consolidate_CashFlowStatements_Data.xlsx – another output file from the format_raw_data.py script

	Consolidate_IncomeStatements_Data.xlsx – yet another output file from the script

	Consolidate_TimeSeries_Data.xlsx – the final interim output file from the format_raw_data.py script


C) model directory
	
	1) .ipynb checkpoints
	these are automatically saved files. I left them alone
	
	2) QC – REFERENCE DATA ONLY. for for interim QC work. None of these are used in any 	scripts.  
		a) QC_ml_data.xlsx    QC on the full machine learning data
		b) QC_new_df.xlsx    QC on getting X and y values for a quarter
		c) QC_walk_through_alg.py QC as I was working out some thoughts on the algorithm
		d) dfout_QC.xlsx – looking at an output file

	3) exploring directory – REFERENCE DATA ONLY. some jupyter notebooks used to explore the machine learning input data and workout some machine learning, some visuals from this were used in the blog article.  
	
		a) distributions.ipynb – looking at distributions of variables
	
 		b) explore_ML_data_and_models.ipynb
	
 		c) explore_ML_data_and_other_models.ipynb


	#NOTE these are in the main section of the models directory:
	
 	ml_data.xlsx the INPUT data for machine learning. This is an OUTPUT file from the format_raw_data.py script.  This has all the data we use for all the quarters for all the tickers, along with SPY prices and 		our target variable. This file is created AFTER running the format_raw_data.py script. 

	Model_Exploring_and_Evaluations.py – a python script to create various models, test them all and export the resulting stats. We are using precision as the most important metric for this project. 

	individual_picks_vs_SPY.py – the final script that does some backtesting. What would have happened if we picked 5 stocks based on the algorithm (with no human interfering). How would it have done? There is
 	a shuffle function in the models used here so there is some variation in the outcome. 


Individual files in the main directory

	requirements.txt file – standard file used to create the python directory 

	sp500_tickerlist_exclude list.xlsx – a list of tickers excluded from the algorithm due to falling outside the studied time period. In other words, these were more recent joins. One exception was ticker SNA 	  	which seems to not be available using Alpha Vantage after attempting numerous times. 

	Which python version.py – a script to print out the python version used for this project which is 3.11.0

	 
  


## Python Libraries Used
alpha_vantage,
datetime,
numpy,
os,
pandas, 
requests,
sklearn,
sys,
time,
warnings




## Method 

For a detailed analysis of the set up and implementation of this project, please see the following technical paper in this repository called “Project 4 Technical Paper” 




## Steps to run this code on your computer 

1) download this repo to your local computer

2) Extract the data from this repo to a folder. Name the folder whatever you like. For the instructions, I'm naming my folder Project4

3) Open the terminal

4) Navigate to the root folder of the directory that we just downloaded and extracted. If you also named your folder Project4, your terminal location should be in ../Project4. We will be staying here to run all our code to get the app up and running.

5) Create a new python environment. I am using conda and naming my new environment project4. You can name your environment whatever you like. Note that I am using python version 3.11. Type the following command in the terminal:

<pre>
	conda create –name project4 python==3.11
</pre>

6) It will ask you if you want to proceed with a y/n option. Press y

7) Activate the new python evironment you just created with the following code in the terminal:

<pre>
	conda activate project4
</pre>
	
8) Please enter the following to install the requirements.txt file: 

<pre>
	conda install --yes --file requirements.txt
</pre>

9) Now we can run the first script. You will need an AlphaVantage password as one of the input system arguments. You also have a choice to pull full data (all tickers that currently make up the S&P 500) or sample data (pulls only the first 2 tickers for each type of data). 

Choice 1 pull full data. This pulls time series data and financial data for all tickers in the SPY so it will take some time. 
In the terminal type: 

<pre>
python data/pull_data.py MyAlphaVantagePassword full
</pre>

Choice 2 pull only 2 tickers for each type of data
In the terminal type:

<pre>
python data/pull_data.py MyAlphaVantagePassword sample
</pre>

10) next, we run a script to consolidate the data and make input data for machine learning

In the terminal type: 

<pre>
python wrangling/format_raw_data.py
</pre>

11) now we run the input data through various groups of classification models: 5 generic models with no modifications, then models with feature reduction, then we do grid searching on these. For each, the precisions are printed out in the terminal. Typically, I found the generic set did better than either models with feature reduction or models with grid searching.

In the terminal type: 

<pre>
python model/Model_Exploring_and_Evaluations.py
</pre>

12) Finally, we run the individual_picks_vs_SPY.py

in the terminal type:

<pre>
python model/individual_picks_vs_SPY.py  
</pre>






## Summary


This was a very interesting project. It seems there may be some value in doing machine learning on market data as results in this project suggest it’s possible that algorithms might outperform the SPY. It certainly warrants further testing and study. 


For more details please either look at the code or read the technical paper mentioned above. 


Thanks. 


 



