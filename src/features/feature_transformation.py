import os
import sys
import pandas as pd
from sklearn import preprocessing
path = '../../dataset/processed/netflix_processed.csv'
sys.path.append(os.path.abspath('../visualization'))
import visualize


class FeatureTransformation:
    def __init__(self):
        # Object Instantiation
        self.df_netflix = pd.read_csv(path, index_col=False)
        self.visualize = visualize.Visualize()

    @staticmethod
    def display_dataframe(name, df, contents):
        table = df.head(contents)
        print('\n')
        print("=" * 150)
        print("◘ ", name, " Dataframe:")
        print(table.to_string())
        print("=" * 150)

    @staticmethod
    def debug_text(title, task):
        print('\n')
        print("=" * 150)
        print('◘ ', title)

        try:
            print(task)

        except Exception as exc:
            print("! ", exc)

        finally:
            print("=" * 150)

    def feature_transformation(self):
        self.df_netflix = self.df_netflix.drop(columns=['user_id', 'plan_duration'], axis=1)
        self.display_dataframe("Processed Netflix Data", self.df_netflix, 25)

        df_encoded = pd.get_dummies(self.df_netflix, columns=['subscription_type', 'country', 'gender', 'device'])
        self.display_dataframe("Processed Netflix Data", df_encoded, 25)

        df_columns = ['monthly_revenue', 'age', 'subscribed_duration', 'start_month', 'clv']

        super_title = "Original Scaling Distribution"
        self.visualize.plot_multi_kde(df_encoded, df_columns, super_title)

        updated_columns = df_encoded.columns.values.tolist()
        super_title = "Robust, Standard, and Min-Max Scaler distribution"
        # Robust Scaler
        scaler = preprocessing.RobustScaler()
        df_robust = scaler.fit_transform(df_encoded)
        df_robust = pd.DataFrame(df_robust, columns=updated_columns)

        # Standard Scaler
        scaler = preprocessing.StandardScaler()
        df_standard = scaler.fit_transform(df_encoded)
        df_standard = pd.DataFrame(df_standard, columns=updated_columns)

        # Min-Max Scaler
        scaler = preprocessing.MinMaxScaler()
        df_minmax = scaler.fit_transform(df_encoded)
        df_minmax = pd.DataFrame(df_minmax, columns=updated_columns)

        self.visualize.plot_compare_kde(df_robust, df_standard, df_minmax, df_columns, super_title)

        self.display_dataframe("Robust Scaler", df_robust, 20)
        self.display_dataframe("Standard Scaler", df_standard, 20)
        self.display_dataframe("Min-Max Scaler", df_minmax, 20)

        self.data_storage(df_robust, 'robust')
        self.data_storage(df_standard, 'standard')
        self.data_storage(df_minmax, 'minmax')

    def data_storage(self, df, process):
        # Save the transformed-intermediate data to storage
        try:
            df.to_csv(f'../../dataset/scaled/netflix_scaled_{process}.csv', sep=',', index=False)

        except Exception as exc:
            self.debug_text("! Exception encountered", exc)

        else:
            text = f"{process.capitalize()} scaled Dataframe successfully saved."
            self.debug_text(text, '...')


if __name__ == "__main__":
    main = FeatureTransformation()
    main.feature_transformation()
