===========================================================================
# Netflix Customer Retention using GPR
Forecasting Netflix Customer Retention based on Gaussian Process Regression.

<div align="center">
    <img width="80%" src="img/netflix (4).gif" alt="Netflix1.gif" >
</div>

### Introduction
The detailed activity of customers can be monitored to predict the rate of retention to stay subscribed for an extended duration of time, also known as the Customer lifetime value (CLV), which can be applied as a metric system that represents the total net profit a company can expect to generate from a customer throughout their entire relationship. Additionally, it takes into account the customer's initial purchase, repeat purchases, and the average duration of their relationship with the company. There are two ways of inspecting the customer lifetime value: historic customer lifetime value (how much each existing customer has already spent with the specified brand) and predictive customer lifetime value (how much customers could spend with the brand). Both measurements of customer lifetime value are useful for tracking business success.

Therefore, by leveraging the power of data analysis and engineering tools such as Matplotlib, Pandas, MisingNo, and Seaborn, in-depth and visual exploration is conducted to discover key insights about age demographics, age, gender distribution, subscription types, and so forth. Consequently, the processed data is then transformed and rescaled using multiple transformation algorithm and a comparison have been analyzed to discover which algorithm works the best and why. Additionally, GridSearchCV has been conducted in order to perform hyperparameter tuning in order to determine the optimal values for a given model, which, in this case, is the Gaussian Process Regression (GPR), which is a nonparametric, Bayesian approach to regression that is making waves in the area of machine learning. GPR has several advantages, including working adequately on small-scale datasets and having the ability to provide uncertainty measurements on the predictions. This study is also designed to be concise and clear to beginners in the field of Data Analyst. 

<br/><br/>

![alt text](https://github.com/shahriar-rahman/Netflix-Customer-Retention-using-GPR/blob/main/img/netflix%20(7).jpg)

<br/><br/>

### ◘ Objective
The primary incentive of this research is to:
* Process dataset by analyzing its integrity, missing values, duplicated values, and so forth.
* Perform various clean-ups, if required, and improve accessibility for more convenient exploratory analysis.
* Conduct exploratory analysis using a myriad of graphing tools to reach a conclusion.
* To reach a proper decision on which of the targeted model's hyperparameters to apply using the processed and transformed dataset.
* Scaled data using three types of algorithms: Robust Scaling, Standard Scaling, and Minmax Scaling.
* Compare and contrast the chosen set of transformation algorithms.
* Apply the concepts of GridSearchCV to reach a proper conclusion on the ideal type of variables that would ultimately boost the model's performance on the data.
* Apply the GPR Model and compare the results for different scaling algorithms.

<br/><br/>

### ◘ Approach
This research is classified into 2 steps:
1.	Data Wrangling: Where the dataset is extracted, tested, cleaned, processed, and stored in memory.
2.	Feature Analysis: Where the processed data is then explored thoroughly to acquire a viable insight.
3.	Feature Transformations: The data is rescaled using Robust, Standard, and Minmax Scaling.
4.	Compare to find the best set of parameters such as kernels and Radial Basis Function (RBF) parameters.
5.	Apply the GPR Model.
6.	Compare the results to reach an accurate assessment.	

<br/><br/>

### ◘ Study Flowcharts
* Feature Analysis & Exploration: 
![alt text](https://github.com/shahriar-rahman/Exploratory-Analysis-of-Netflix-Userbase/blob/main/img/edaFlowchart.png)

<br/>

* Model Workflow:
![alt text](https://github.com/shahriar-rahman/Netflix-Customer-Retention-using-GPR/blob/main/img/ModelFlowchart.JPG)

<br/><br/>

### ◘ Methodologies & Concepts applied
* Diagnose and fix structural errors
* Check and Clean data
* Address duplicates & outliers
* Logical feature amalgamation to construct a unique variable
* Univariate inspection
* Bivariate inspection
* Feature correlations using Multivariate Analysis
* Distributions and Pattern Recognition using Visualization tools 
* Model Hyperparameter analysis and tuning
* Gaussian Process Algorithm
* Model Fitting, Training, Saving, Loading, and Evaluation
* Loss Evaluation and Comparison

<br/><br/>

### ◘ Project Organization
------------
    ├─-- LICENSE
    |
    ├─-- README.md              # The top-level README for developers using this project
    |
    ├─-- dataset                # Different types of data derived from the original raw dataset
    |    └──  processed        
    |    └──  raw
    |    └──  scaled
    |    └──  test_set
    |
    |
    ├─-- models                 # Trained and serialized models for future model predictions  
    |    └── gpr_minmax.pkl
    |    └── gpr_robust.pkl
    |    └── gpr_standard.pkl
    |
    |
    ├─ graphs                    # Generated graphics and figures obtained from visualization.py
    |
    |
    ├─-- img                    # Project related files
    |
    ├─-- requirements.txt       # The requirements file for reproducing the analysis environments
    |                         
    |
    ├─-- setup.py               # makes project pip installable, so that src can be imported
    |
    |
    ├─-- src                    # Source code for use in this research
    |   └───-- __init__.py    
    |   |
    |   ├─-- features            # Scripts to modify the raw data into different types of features for modeling
    |   |   └── feature_preprocessing.py
    |   |   └── feature_exploration.py
    |   |   └── feature_transformation.py
    |   |
    |   ├─-- models                # Contains py filess for inspecting hyperparameters, training, and using trained models to make predictions         
    |   |   └─── predict_model.py
    |   |   └─── train_model.py
    |   |
    |   └───-- visualization        # Construct exploratory and result oriented visualizations to identify and reaffirm patterns
    |       └───-- visualize.py
    |
    ├─
--------

<br/><br/>

### ◘ Libraries & Technologies utilized
* Python 3.11
* PyCharm IDE (2023.1)
* pip 23.0.1
* setuptools 65.5.1
* scikit-learn 1.2.2
* seaborn 0.12.2
* matplotlib 3.7.1
* missingno 0.5.2
* numpy 1.24.2
* plotly 5.15.0
* joblib 1.2.0

<br/><br/>



<br/><br/>

