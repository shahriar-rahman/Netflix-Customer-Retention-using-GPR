import os
import sys
import pandas as pd
import seaborn as sns
import missingno as msn
import matplotlib.pyplot as plt
df_path = '../../dataset/raw/netflix_userbase.csv'
sys.path.append(os.path.abspath('../visualization'))
import visualize


class FeatureProcessing:
    def __init__(self):
        self.df_netflix = pd.read_csv(df_path, index_col=False)
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

    def feature_processing(self):
        # Inquire structural integrity
        text_1 = "Netflix userbase dataframe:"
        self.df_netflix.reset_index(drop=True, inplace=True)
        self.display_df(self.df_netflix, 25, text_1)

        self.df_netflix.info()
        text_2 = "Dataframe structure:"
        self.action_listener(self.df_netflix.describe(), text_2)

        # Enhance data accessibility
        text_2 = "Pre-processing Columns:"
        self.action_listener(self.df_netflix.columns, text_2)

        self.df_netflix.rename(columns={'User ID': 'user_id', 'Subscription Type': 'subscription_type',
                                        'Monthly Revenue': 'monthly_revenue', 'Join Date': 'join_date',
                                        'Last Payment Date': 'last_payment_date', 'Country': 'country',
                                        'Age': 'age', 'Gender': 'gender', 'Device': 'device',
                                        'Plan Duration': 'plan_duration'}, inplace=True)

        text_2 = "Post-processing Columns:"
        self.action_listener(self.df_netflix.columns, text_2)

        # Analyze any Missing values
        self.visualize.plot_msn(self.df_netflix, 'bar')
        self.visualize.plot_msn(self.df_netflix, 'matrix')
        self.visualize.plot_msn(self.df_netflix, 'dendrogram')

        text_2 = "Dataframe Nullity table: "
        self.action_listener(self.df_netflix.isnull().sum(), text_2)

        # Data cleaning
        text_2 = "Shape of dataframe:"
        self.action_listener(self.df_netflix.shape, text_2)
        self.df_netflix.dropna(axis='columns')

        # Find duplicate Cell
        duplicated_cells = 0
        check_duplicate = self.df_netflix.duplicated()

        for row in check_duplicate:
            if row:
                duplicated_cells += 1

        duplicated_prc = (duplicated_cells / len(check_duplicate)) * 100
        print("• Total Cells:", len(check_duplicate), '\n', "• Duplicated Cells:", duplicated_cells, '\n',
              "• Duplicate %:", duplicated_prc)

        # Check for Outliers
        self.visualize.plot_outlier(self.df_netflix)

        # Ensure data consistency
        self.df_netflix['join_date'] = pd.to_datetime(self.df_netflix['join_date'])
        self.df_netflix['last_payment_date'] = pd.to_datetime(self.df_netflix['last_payment_date'])
        self.df_netflix.info()

        # Combine Features
        text_1 = "Duration of users subscribed: "
        self.df_netflix['subscribed_duration'] = self.df_netflix['last_payment_date'] - self.df_netflix['join_date']
        self.df_netflix['subscribed_duration'] = self.df_netflix['subscribed_duration'].dt.days
        self.display_df(self.df_netflix['subscribed_duration'], 25, text_1)

        text_1 = "Month in which the user joined: "
        self.df_netflix['start_month'] = self.df_netflix['join_date'].dt.month
        self.display_df(self.df_netflix['start_month'], 25, text_1)

        text_1 = "Customer Lifetime Value (clv): "
        self.df_netflix['clv'] = self.df_netflix['subscribed_duration'] * self.df_netflix['monthly_revenue']
        self.display_df(self.df_netflix['clv'], 25, text_1)

        # Truncate redundant features
        self.df_netflix = self.df_netflix.drop(['last_payment_date', 'join_date'], axis=1)
        self.df_netflix.info()

        # Save processed data
        try:
            self.df_netflix.to_csv('../../dataset/processed/netflix_processed.csv', sep=',', index=False)
            self.df_netflix.to_json('../../dataset/processed/netflix_processed.json')

        except Exception as exc:
            print("! Exception encountered", exc)

        else:
            print("Dataframe successfully saved")


if __name__ == "__main__":
    main = FeatureProcessing()
    main.feature_processing()

