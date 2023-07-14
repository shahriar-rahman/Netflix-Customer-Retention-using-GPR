===========================================================================
# Netflix Customer Retention using GPR
Forecasting Netflix Customer Retention based on Gaussian Process Regression.

<div align="center">
    <img width="65%" src="img/netflix (4).gif" alt="Netflix1.gif" >
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

### ◘ Feature Transformations
Three types of Scaling methods are conducted in order to reach peak efficiency in the model's performance:
1. Robust Scaling
2. Standard Scaler
3. MinMax Scaling

<br/>

![al text](https://github.com/shahriar-rahman/Netflix-Customer-Retention-using-GPR/blob/main/graphs/Fig_23_refined.png)

<br/><br/>

### ◘ Evaluation 
Training Loss

| Type | RMSE | MAE | R-Squared |
|--|--|--|---|
| Robust GPR | 1.93e-11 | 3.26e-06 | 0.99 |
| Standard GPR | 2.30e-11 | 3.41e-06 | 0.99 |
| MinMax GPR | 3.79e-13 | 4.59e-07 | 0.99 |


Test Loss

| Type | RMSE | MAE | R-Squared |
|--|--|--|---|
| Robust GPR | 1.08e-09 | 1.20e-05 | 0.99 |
| Standard GPR | 1.20e-09 | 1.29e-05 | 0.99 |
| MinMax GPR | 1.89e-12 | 7.69e-07 | 0.99 |

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
* pandas 2.0.0
* scikit-learn 1.2.2
* seaborn 0.12.2
* matplotlib 3.7.1
* missingno 0.5.2
* numpy 1.24.2
* joblib 1.2.0

<br/><br/>

### ◘ Module Installation (setup.py)
1. To use the *setup.py* file in Python, the first objective is to have the *setuptools* module installed. It can be accomplished by running the following command:
```
pip install setuptools                                     
```
2. Once the setuptools module is installed, use the setup.py file to build and distribute the Python package by running the following command:
```
python setup.py sdist bdist_wheel
```
3. In order to install the my_package package, run the following command:
```
pip install my_package                                 
```
4. This will install the my_package package and any of its dependencies that are not already installed on your system. Once the package is installed, you can use it in your Python programs by importing it like any other module. For example:
```
import my_package                                
```

<br/><br/>

### ◘ Python Library Installation (using pip)
In order to *install* the required packages on the local machine, Open pip and run the following commands separately:
```
> pip install setuptools                    

> pip install pandas                                                          

> pip install scikit-learn                                      

> pip install seaborn

> pip install matplotlib

> pip install missingno

> pip install numpy

> pip install joblib                                  
```

<br/><br/>

### ◘ Supplementary Resources
For more details, visit the following links:
* https://pypi.org/project/setuptools/
* https://pypi.org/project/pandas/
* https://pypi.org/project/scikit-learn/
* https://pypi.org/project/seaborn/
* https://pypi.org/project/matplotlib/
* https://pypi.org/project/missingno/
* https://pypi.org/project/numpy/
* https://pypi.org/project/joblib/

<br/><br/>

### ◘ MIT License
Copyright (c) 2023 Shahriar Rahman

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

<br/>

===========================================================================

