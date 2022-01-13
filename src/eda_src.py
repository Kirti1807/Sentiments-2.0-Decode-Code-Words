import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from data_ingestion import DataUtils
import plotly
from plotly import tools
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

init_notebook_mode(connected=True)
import plotly.offline as offline
import plotly.graph_objs as go
from data_ingestion import DataUtils


class EDA:
    def __init__(self, data) -> None:
        self.data = data

    def check_class_distributions(self):
        logging.info("Checking class distributions")
        try:
            # check the distribution of the classes except column Review and ID, make a pie chart in plotly
            data_utils = DataUtils()
            train_data, test_data = data_utils.read_data()
            # plot a pie chart in plotly for every column except Review and ID
            for column in train_data.columns:
                if column not in ["Id", "Review"]:
                    # create a pie chart in plotly
                    labels = train_data[column].unique()
                    values = train_data[column].value_counts()
                    fig = go.Figure(data=[go.Pie(labels=labels, values=values)])
                    fig.update_layout(title_text=column)
                    fig.show()
        except Exception as e:
            logging.error("Error in checking class distributions")
            raise e

            # check the distribution of the classes except column Review and ID, make a pie chart in plotly

    def cross_tabulation(self):
        logging.info("Cross tabulation")
        try:
            # cross tabulation for two columns in pandas
            data_utils = DataUtils()
            train_data, test_data = data_utils.read_data()
            #train_data
            # cross tabulation for two columns in pandas
            pd.crosstab(
                train_data["Components"], train_data["Delivery and Customer Support"]
            ).plot(kind="bar")
            pd.crosstab(
                train_data["Design and Aesthetics"], train_data["Dimensions"]
            ).plot(kind="bar")

            plt.show()
        except Exception as e:
            logging.error("Error in checking class distributions")
            raise e

    def basic_data_exploration(self):
        logging.info("Exploring data")
        try:
            logging.info("Showing the head of the data")
            print(self.data.head())
            print("\n Shape of the data \n", self.data.shape)
            print("\n DataTYpes of Every Column \n", self.data.dtypes)
            print("\n Describing the data:- \n ", self.data.describe())
            print("\n Information about the data:- \n", self.data.info())
            print("\n No. of Null Values in your data:- \n", self.data.isnull().sum())
        except Exception as e:
            logging.error("Error in exploring data")
            raise e

    def check_missing_values(self):
        logging.info("Checking missing values")
        try:
            logging.info("Checking missing values")
            # get the percentage of every column and plot it in a bar chart
            missing_values = self.data.isnull().sum()
            missing_values_percentage = 100 * self.data.isnull().sum() / len(self.data)
            missing_values_table = pd.concat(
                [missing_values, missing_values_percentage], axis=1
            )
            missing_values_table_ren_columns = missing_values_table.rename(
                columns={0: "Missing Values", 1: "% of Total Values"}
            )
            # plot the missing values
            missing_values_table_ren_columns.plot(kind="bar", figsize=(20, 10))

            plt.show()
        except Exception as e:
            logging.error("Error in checking missing values")
            raise e


if __name__ == "__main__":
    data_utils = DataUtils()
    train_data, test_data = data_utils.read_data()
    eda = EDA(train_data)
    eda.check_class_distributions()
    eda.cross_tabulation()
    eda.basic_data_exploration()
    eda.check_missing_values()
