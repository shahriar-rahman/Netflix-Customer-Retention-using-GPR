import os
import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import missingno as msn
path = '../../dataset/processed/netflix_processed.csv'
sys.path.append(os.path.abspath('../visualization'))
import visualize


class FeatureExploration:
    def __init__(self):
        self.df_netflix = pd.read_csv(path, index_col=False)
        self.visualize = visualize.Visualize()

    @staticmethod
    def action_listener(action, text):
        print('\n')
        print('-' * 150)
        print(text)
        print(action)
        print('-' * 150)
        print('\n')

    @staticmethod
    def display_df(df, contents, text):
        print('\n')
        print('-' * 150)
        print(text)
        print(df.head(contents).to_string())
        print('-' * 150)
        print('\n')

    def feature_exploration(self):
        # Monthly Revenues
        text = "Monthly Revenues: "
        self.display_df(self.df_netflix['monthly_revenue'], 25, text)

        text = "Revenue Frequency: "
        self.action_listener(self.df_netflix['monthly_revenue'].value_counts(), text)

        text = "Histogram: Monthly Revenue"
        self.visualize.plot_hist(self.df_netflix, 'monthly_revenue', text)

        # Gender Distribution
        text = "Countplot: Distribution of Genders"
        self.visualize.plot_count(self.df_netflix, 'gender', text)

        # Age Groups
        self.action_listener(self.df_netflix['age'].mean(), "Mean Age: ")
        self.action_listener(self.df_netflix['age'].median(), "Median Age: ")
        self.action_listener(self.df_netflix['age'].std(), "Age Deviation: ")

        text = "Countplot: Distribution of Age"
        self.visualize.plot_count(self.df_netflix, 'age', text)

        text = "KDE: Age Distribution"
        self.visualize.plot_kde(self.df_netflix, 'age', text)

        # Subscription Types
        text = "Unique values for Subscription:"
        self.action_listener(self.df_netflix['subscription_type'].unique(), text)
        self.action_listener(self.df_netflix['subscription_type'].value_counts(), text)

        text = "Pie Chart: Subscription Types"
        explode = [0.4, 0.0, 0.0]
        self.visualize.plot_pie(self.df_netflix, 'subscription_type', explode, text)

        # Monthly User Analysis
        customer_churn = self.df_netflix['clv'] > 30
        self.action_listener(customer_churn.value_counts(), "Customer Churn:")
        print(customer_churn.get(False, 0))

        self.action_listener(self.df_netflix['start_month'].value_counts(), "Monthly Stats")
        text = "Bar Plot: Monthly Starting Users"
        self.visualize.plot_bar(self.df_netflix, 'start_month', 'v', text)

        # User Count in Countries
        text = "Number of users in Countries: "
        self.display_df(self.df_netflix['country'].value_counts(), 25, text)
        self.visualize.plot_count(self.df_netflix, 'country', text)

        # Age Density
        action = self.df_netflix.groupby('subscription_type')['age'].mean().sort_values()
        self.action_listener(action, "Age density by Subscriptions: ")

        text = "Violin Plot: Age Density by Subscriptions"
        self.visualize.plot_violin(self.df_netflix, 'subscription_type', 'age', text)

        # Pearson Correlation Heatmap for revenue and age
        text = "Correlation Matrix: Revenue and Age"
        self.visualize.plot_correlation(self.df_netflix[['monthly_revenue', 'age']], text)

        # Pair Plotting all the data
        self.visualize.plot_pair(self.df_netflix)

        # Monthly Revenue by Genders
        action = self.df_netflix.groupby('gender')['monthly_revenue'].mean().sort_values()
        self.action_listener(action, "Monthly Revenue by Genders: ")

        text = "Boxplot: Monthly Revenue by Genders"
        self.visualize.plot_box(self.df_netflix, 'gender', 'monthly_revenue', text)

        # Most used Devices
        action = self.df_netflix['device'].value_counts()
        self.action_listener(action, "Most used Devices: ")

        text = "Countplot: Most used Devices "
        self.visualize.plot_count(self.df_netflix, 'device', text)

        # Monthly Revenue vs Subscription Types
        text = "Boxplot: Revenues by Subscription Types  "
        self.visualize.plot_box(self.df_netflix, 'monthly_revenue', 'subscription_type', text)

        text = "Scatterplot: Monthly Revenue vs Subscription Types  "
        self.visualize.plot_scatter(self.df_netflix, 'monthly_revenue', 'subscription_type', text)

        # Subscription Type vs Plan Duration
        text = "Scatterplot: Subscription Type vs. Plan Duration  "
        self.visualize.plot_scatter(self.df_netflix, 'plan_duration', 'subscription_type', text)

        # Plan Duration vs Monthly Revenues
        text = "Boxplot: Monthly Revenues by User Plans  "
        self.visualize.plot_box(self.df_netflix, 'monthly_revenue', 'plan_duration', text)

        text = "Scatterplot: Monthly Revenue vs Subscription Types  "
        self.visualize.plot_scatter(self.df_netflix, 'plan_duration', 'monthly_revenue', text)

        # Feature Correlation for all the continuous Features
        text = "Feature Correlation Heatmap"
        columns = ['monthly_revenue', 'age', 'subscribed_duration', 'start_month', 'clv']
        self.visualize.plot_correlation(self.df_netflix[columns], text)


if __name__ == "__main__":
    main = FeatureExploration()
    main.feature_exploration()
