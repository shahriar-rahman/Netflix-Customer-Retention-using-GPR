import os
import sys
import joblib as jb
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae
sys.path.append(os.path.abspath('../visualization'))
import visualize

data_type_index = 2
data_type = ['robust', 'standard', 'minmax']
path_test = [f'../../dataset/test_set/netflix_test_x_{data_type[data_type_index]}.csv',
             f'../../dataset/test_set/netflix_test_y_{data_type[data_type_index]}.csv']
path_model = f'../../models/gpr_{data_type[data_type_index]}.pkl'


class PredictModel:
    def __init__(self):
        # Object Instantiation
        self.visualize = visualize.Visualize()
        self.test_x = pd.read_csv(path_test[0], index_col=False)
        self.test_y = pd.read_csv(path_test[1], index_col=False)
        self.gpr = jb.load(f'../../models/gpr_{data_type[data_type_index]}.pkl')

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

    def predict_model(self):
        pred_y = self.gpr.predict(self.test_x)

        # Loss Functions
        loss_mse = mse(self.test_y, pred_y, squared=False)
        loss_r_mse = mse(self.test_y, pred_y, squared=True)
        loss_r2 = r2_score(self.test_y, pred_y)
        loss_mae = mae(self.test_y, pred_y)

        self.debug_text(f"MSE (Test {data_type[data_type_index]} data):", loss_mse)
        self.debug_text(f"RMSE (Test {data_type[data_type_index]} data):", loss_r_mse)
        self.debug_text(f"R-Squared (Test {data_type[data_type_index]} data):", loss_r2)
        self.debug_text(f"MAE (Test {data_type[data_type_index]} data):", loss_mae)

        title = f"Residual Loss (Test {data_type[data_type_index]} data)"
        self.visualize.plot_residual(self.test_y, pred_y, title, "Actual Values", "Predicted Values")

        bins = 10
        x_label = 'Errors'
        y_label = 'Frequency'
        error_list = pred_y - self.test_y.to_numpy().ravel()
        title = f"Prediction error distribution (Test {data_type[data_type_index]} data)"
        self.visualize.plot_dist(error_list, bins, title, x_label, y_label)


if __name__ == "__main__":
    main = PredictModel()
    main.predict_model()
