import math
import numpy as np
import pandas as pd
import seaborn as sb
import missingno as msn
from matplotlib import pyplot as plt
from matplotlib.font_manager import FontProperties


class Visualize:
    def __init__(self):
        pass

    @staticmethod
    def graph_settings():
        # Customizable Set-ups
        plt.figure(figsize=(13, 15))
        font = FontProperties()
        font.set_family('serif bold')
        font.set_style('oblique')
        font.set_weight('bold')
        ax = plt.axes()
        ax.set_facecolor("#e6eef1")

    def plot_msn(self, df, kind):
        self.graph_settings()

        if kind == 'matrix':
            try:
                msn.matrix(df, color=(1, 0.30, 0.20), figsize=[13, 18], fontsize=12)

            except Exception as exc:
                print("! ", exc)

            else:
                plt.title("Missingno Nullity Matrix for the raw data", fontsize=15, fontweight='bold')
                plt.show()

        elif kind == 'bar':
            try:
                msn.bar(df, color="dodgerblue", sort="ascending", figsize=(13, 18), fontsize=12)

            except Exception as exc:
                print("! ", exc)

            else:
                plt.title("Missingno Bar Plot for the raw data", fontsize=15, fontweight='bold')
                plt.show()

        elif kind == 'heatmap':
            try:
                msn.heatmap(df, cmap="RdYlGn", figsize=(13, 18), fontsize=12)

            except Exception as exc:
                print("! ", exc)

            else:
                plt.title("Missingno Nullity Heatmap for the raw data", fontsize=15, fontweight='bold')
                plt.show()

        elif kind == 'dendrogram':
            try:
                msn.dendrogram(df, figsize=(13, 18), fontsize=11)

            except Exception as exc:
                print("! ", exc)

            else:
                plt.title("Missingno Nullity Dendrogram for the raw data", fontsize=15, fontweight='bold')
                plt.show()

    def plot_outlier(self, df):
        self.graph_settings()

        sb.boxplot(df)
        plt.title("Boxplot for the Numerical data", fontsize=16, fontweight='bold')
        plt.show()

    def plot_hist(self, df, x, text):
        self.graph_settings()

        plt.hist(df[x], bins=20, color='indigo', edgecolor='#0d0103', linewidth=1.2)
        plt.title(text, fontsize=16, fontweight='bold')
        plt.xlabel(x.capitalize().replace('_', ' '), fontsize=12, fontweight='bold')
        plt.ylabel("Frequency", fontsize=12, fontweight='bold')
        plt.show()

    def plot_count(self, df, x, text):
        self.graph_settings()
        palette = ['#038754', '#432371']
        palette_a = ['#6b2115', '#693113', '#694313', '#436913', '#12750e', '#066b41', '#02736f', '#133863',
                     '#2d2152', '#48275c', '#661f40', '#661f29']

        if x == 'age':
            sb.countplot(x=x, data=df, palette=palette_a)

        elif x == 'country' or x == 'device':
            sb.countplot(x=x, data=df)

        else:
            sb.countplot(x=x, data=df, palette=palette)
        plt.title(text, fontsize=16, fontweight='bold')
        plt.xlabel(x.capitalize().replace('_', ' '), fontsize=12, fontweight='bold')
        plt.ylabel('Frequency', fontsize=12, fontweight='bold')
        plt.show()

    def plot_kde(self, df, x, text):
        self.graph_settings()
        df[x].plot(kind='kde', color='#343063', linewidth=1.2)
        plt.title(text, fontsize=16, fontweight='bold')
        plt.xlabel(x.capitalize().replace('_', ' '), fontsize=12, fontweight='bold')
        plt.ylabel("Frequency", fontsize=12, fontweight='bold')
        plt.show()

    def plot_pie(self, df, x, explode, text):
        self.graph_settings()
        sub_count = df[x].value_counts()
        plt.pie(sub_count, labels=sub_count.index, autopct='%1.1f%%', shadow=True, explode=explode, startangle=85)
        plt.title(text, fontsize=16, fontweight='bold')
        plt.legend(loc="upper right")
        plt.show()

    def plot_bar(self, df, x, orientation, text):
        self.graph_settings()

        if orientation == 'h':
            plt.barh(df[x], df[x].index, color='#0b4d80')

        else:
            plt.bar(df[x], df[x].index, color='#0b4d80')
            plt.title(text, fontsize=16, fontweight='bold')
            plt.xlabel(x.capitalize().replace('_', ' '), fontsize=12, fontweight='bold')
            plt.ylabel('Count', fontsize=12, fontweight='bold')
            plt.show()

    def plot_violin(self, df, x, y, text):
        self.graph_settings()

        sb.violinplot(x=x, y=y, data=df)
        plt.title(text, fontsize=16, fontweight='bold')
        plt.xlabel(x.capitalize().replace('_', ' '), fontsize=12, fontweight='bold')
        plt.ylabel(y.capitalize().replace('_', ' '), fontsize=12, fontweight='bold')
        plt.show()

    def plot_correlation(self, df, text):
        self.graph_settings()

        corr_matrix = df.corr()
        sb.heatmap(corr_matrix, annot=True, cmap=sb.cubehelix_palette(as_cmap=True))
        plt.title(text, fontsize=16, fontweight='bold')
        plt.show()

    @staticmethod
    def plot_pair(df):

        sb.set(style="ticks", color_codes=True)
        sb.pairplot(df)
        plt.show()

    def plot_box(self, df, x, y, text):
        self.graph_settings()

        sb.boxplot(x=x, y=y, data=df, palette="Blues")
        plt.title(text, fontsize=16, fontweight='bold')
        plt.xlabel(x.capitalize().replace('_', ' '), fontsize=12, fontweight='bold')
        plt.ylabel(y.capitalize().replace('_', ' '), fontsize=12, fontweight='bold')
        plt.show()

    def plot_scatter(self, df, x, y, text):
        self.graph_settings()

        plt.scatter(x=x, y=y, data=df, color='#0b2470', linewidth=1.2)
        plt.title(text, fontsize=16, fontweight='bold')
        plt.xlabel(x.capitalize().replace('_', ' '), fontsize=12, fontweight='bold')
        plt.ylabel(y.capitalize().replace('_', ' '), fontsize=12, fontweight='bold')
        plt.grid(True)
        plt.show()

    @staticmethod
    def plot_multi_kde(df, columns, super_title):
        row = 2
        column = 3
        i = 0
        j = 0

        fig, ax = plt.subplots(row, column, figsize=(19, 23))
        for feature in columns:
            x_label = str(feature).replace('_', ' ').capitalize().title()

            if i < row and j < column:
                sb.kdeplot(df[feature], ax=ax[i, j], color='r')
                ax[i, j].set_xlabel(x_label, fontsize=10, fontweight='bold')
                ax[i, j].set_ylabel('Density', fontsize=10, fontweight='bold')

            j += 1
            if j > column-1:
                i += 1
                j = 0

        fig.delaxes(ax[1, 2])
        fig.suptitle(super_title, fontsize=18, fontweight='bold')
        plt.show()

    @staticmethod
    def plot_compare_kde(df1, df2, df3, columns, super_title):
        i = 0
        j = 0
        row = 2
        column = 3

        fig, ax = plt.subplots(row, column, figsize=(19, 23))
        for feature in columns:
            x_label = str(feature).replace('_', ' ').capitalize().title()

            if i < row and j < column:
                sb.kdeplot(df1[feature], ax=ax[i, j], color='r')
                sb.kdeplot(df2[feature], ax=ax[i, j], color='g')
                sb.kdeplot(df3[feature], ax=ax[i, j], color='b')

                ax[i, j].set_xlabel(x_label, fontsize=10, fontweight='bold')
                ax[i, j].set_ylabel('Density', fontsize=10, fontweight='bold')
                ax[i, j].legend(["Robust", "Standard", "MinMax"], fontsize="10", loc="upper right")

            j += 1
            if j > column - 1:
                i += 1
                j = 0

        fig.delaxes(ax[1, 2])
        fig.suptitle(super_title, fontsize=18, fontweight='bold')
        plt.show()

    def plot_residual(self, x, y, title, x_label, y_label):
        self.graph_settings()
        sb.residplot(x=x, y=y, lowess=True)

        plt.title(title, fontsize=17, fontweight='bold')
        plt.xlabel(x_label, fontsize=12, fontweight='bold')
        plt.xticks()
        plt.ylabel(y_label, fontsize=12, fontweight='bold')
        plt.show()

    def plot_loss_curve(self, model, title, x_label, y_label):
        self.graph_settings()

        plt.plot(np.array(model.loss_curve_[15:]))
        plt.title(title, fontsize=16)
        plt.xlabel(x_label, fontsize=12, fontweight='bold')
        plt.ylabel(y_label, fontsize=12, fontweight='bold')
        plt.show()

    def plot_dist(self, df, bins, title, x_label, y_label):
        self.graph_settings()

        sb.histplot(data=df, kde=True, bins=bins)
        plt.title(title, fontsize=17, fontweight='bold')
        plt.xlabel(x_label, fontsize=12, fontweight='bold')
        plt.ylabel(y_label, fontsize=12, fontweight='bold')
        plt.show()


if __name__ == "__main__":
    main = Visualize()
