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
from application_logger import CustomApplicationLogger

class EDA:
    def __init__(self, data) -> None:
        self.data = data
        self.file_object = open(
            r"D:\ML_Projects\MultiClassClassification\Sentiments-2.0-Decode-Code-Words\logs\EdaLogs.txt",
            "a+",
        )
        self.logging = CustomApplicationLogger()

    def check_class_distributions(self):
        self.logging.log(
            self.file_object,
            "In check_class_distributions method in EDA class: checking class distributions"
        )
        
        try:
            # check the distribution of the classes except column Review and ID, make a pie chart in plotly
            for column in self.data.columns:
                if column not in ["Id", "Review"]: 
                    labels = self.data[column].unique()
                    values = self.data[column].value_counts()
                    fig = go.Figure(data=[go.Pie(labels=labels, values=values)])
                    fig.update_layout(title_text=column)
                    fig.show()
        except Exception as e:
            self.logging.log(
                self.file_object,
                f"In check_class_distributions method in EDA class: Error in class distribution: {e}"
            )
            raise e


    def cross_tabulation(self):
        self.logging.log(
            self.file_object,
            "In cross_tabulation method in EDA class: cross tabulation"
        )
        
        try:
            # cross tabulation for two columns in pandas
            pd.crosstab(
                self.data["Components"], self.data["Delivery and Customer Support"]
            ).plot(kind="bar")
            pd.crosstab(
                self.data["Design and Aesthetics"], self.data["Dimensions"]
            ).plot(kind="bar")

            plt.show()
        except Exception as e:
            self.logging.log(
                self.file_object,
                f"In cross_tabulation method in EDA class: Error in cross tabulation: {e}"
            )
            raise e

    def basic_data_exploration(self):
        self.logging.log(
            self.file_object,
            "In basic_data_exploration method in EDA class: showing basic data exploration"
        )
        try:
            logging.info("Showing the head of the data")
            print(self.data.head())
            print("\n Shape of the data \n", self.data.shape)
            print("\n DataTYpes of Every Column \n", self.data.dtypes)
            print("\n Describing the data:- \n ", self.data.describe())
            print("\n Information about the data:- \n", self.data.info())
            print("\n No. of Null Values in your data:- \n", self.data.isnull().sum())
        except Exception as e:
            self.logging.log(
                self.file_object,
                f"In basic_data_exploration method in EDA class: Error in data exploration: {e}"
            )
            raise e

    def check_missing_values(self):
        self.logging.log(
            self.file_object,
            "In check_missing_values method in EDA class: checking missing values"
        )
        try:
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
            self.logging.log(
                self.file_object,
                f"In check_missing_values method in EDA class: Error in checking missing values: {e}"
            )
            raise e


if __name__ == "__main__":
    data_utils = DataUtils()
    train_data , test_data = data_utils.read_data(
        train_path=r"D:\ML_Projects\MultiClassClassification\Sentiments-2.0-Decode-Code-Words\data\trainmulticlass.csv",
        test_path=r"D:\ML_Projects\MultiClassClassification\Sentiments-2.0-Decode-Code-Words\data\testmulticlass.csv"
    )
    eda = EDA(train_data)
    eda.check_class_distributions()
    eda.cross_tabulation()
    eda.basic_data_exploration()
    eda.check_missing_values()
