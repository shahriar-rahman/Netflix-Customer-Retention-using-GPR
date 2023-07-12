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

![alt text](https://github.com/shahriar-rahman/Netflix-Customer-Retention-using-GPR/blob/main/img/netflix%20(7).jpg)

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
* Feature Exploration: 
![alt text](https://github.com/shahriar-rahman/Exploratory-Analysis-of-Netflix-Userbase/blob/main/img/edaFlowchart.png)

<br/><br/>

### ◘ Methodologies & Technologies applied
* Diagnose and fix structural errors
* Check and Clean data
* Address duplicates & outliers
* Logical feature amalgamation to construct a unique variable
* Univariate inspection
* Bivariate inspection
* Feature correlations
* Seaborn & Matplotplib visualizations
* GridSearchCV
* GPR
* Saving and loading an ML model

<br/><br/>
