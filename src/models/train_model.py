import os
import sys
import joblib as jb
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.gaussian_process.kernels import RBF
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae
from sklearn.gaussian_process import GaussianProcessRegressor as Gpr
sys.path.append(os.path.abspath('../visualization'))
import visualize

data_type = ['robust', 'standard', 'minmax']
data_type_index = 2
test_size = 0.25
random_state = 48
path = f'../../dataset/scaled/netflix_scaled_{data_type[data_type_index]}.csv'


class TrainModel:
    def __init__(self):
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

    def train_model(self):
        df_columns = self.df_netflix.columns.values.tolist()
        self.debug_text("Dataframe Columns", df_columns)

        df_x = self.df_netflix.copy()
        df_x = df_x.drop(columns=['clv'])

        df_y = self.df_netflix[['clv']]
        self.display_dataframe("Features", df_x, 10)
        self.display_dataframe("Label", df_y, 10)

        train_x, test_x, train_y, test_y = train_test_split(df_x, df_y, test_size=test_size, random_state=random_state)
        self.debug_text("Shape of Training set:", train_x.shape)
        self.debug_text("Shape of Testing set:", test_x.shape)

        # Grid CV Search
        kernels = [1 * RBF(1), 2 * RBF(1), 1 * RBF(2), 2 * RBF(2)]
        gpr_params = {'kernel': kernels}

        gpr = Gpr(optimizer='fmin_l_bfgs_b', random_state=48)
        gpr_grid = GridSearchCV(gpr, gpr_params, scoring='neg_mean_squared_error', cv=5)
        grid_search = gpr_grid
        grid_search.fit(train_x, train_y)

        self.debug_text("Ideal parameters: ", grid_search.best_params_)
        self.debug_text("Ideal Score: ", grid_search.best_score_)
        ideal_param = grid_search.best_params_['kernel']

        k = 1
        rbf = 1
        for kernel in kernels:
            if kernel == ideal_param:
                self.debug_text(f"Ideal Raw parameters:\nk={k}, rbf={rbf}", kernel)

            k += 1
            if k > 2:
                k = 1
                rbf += 1

            if rbf > 2:
                break

        # GPR Model with ideal hyperparameters
        gpr = Gpr(kernel=ideal_param, optimizer='fmin_l_bfgs_b', random_state=48)
        gpr.fit(train_x, train_y)
        pred_y = gpr.predict(train_x)

        # Loss Functions
        loss_mse = mse(train_y, pred_y, squared=False)
        loss_r_mse = mse(train_y, pred_y, squared=True)
        loss_r2 = r2_score(train_y, pred_y)
        loss_mae = mae(train_y, pred_y)

        self.debug_text(f"MSE (Training {data_type[data_type_index]} data):", loss_mse)
        self.debug_text(f"RMSE (Training {data_type[data_type_index]} data):", loss_r_mse)
        self.debug_text(f"R-Squared (Training {data_type[data_type_index]} data):", loss_r2)
        self.debug_text(f"MAE (Training {data_type[data_type_index]} data):", loss_mae)

        title = f"Residual Loss (Training {data_type[data_type_index]} data)"
        self.visualize.plot_residual(train_y, pred_y, title, "Actual Values", "Predicted Values")

        bins = 10
        x_label = 'Errors'
        y_label = 'Frequency'
        error_list = pred_y - train_y.to_numpy().ravel()
        title = f"Prediction error distribution (Training {data_type[data_type_index]} data)"
        self.visualize.plot_dist(error_list, bins, title, x_label, y_label)

        # Store Model
        try:
            jb.dump(gpr, f'../../models/gpr_{data_type[data_type_index]}.pkl')

        except Exception as exc:
            self.debug_text("! Exception encountered", exc)

        else:
            self.debug_text("• Model saved successfully", '')

    def data_storage(self, df, name):
        # Save partitioned data to storage
        try:
            df.to_csv(f'../../dataset/test_set/netflix_{str(name)}.csv', sep=',', index=False)

        except Exception as exc:
            self.debug_text("! Exception encountered", exc)

        else:
            text = "Dataframe successfully saved"
            self.debug_text(text, '...')


if __name__ == "__main__":
    main = TrainModel()
    main.train_model()
