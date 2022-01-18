import numpy as np
import pandas as pd
import nltk
import re
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.tokenize import RegexpTokenizer
import logging
from application_logger import CustomApplicationLogger
from data_ingestion import DataUtils
from sklearn.model_selection import train_test_split


class DataProcessing:
    def __init__(self, train_data) -> None:
       self.train_data = train_data
       self.logging = CustomApplicationLogger()
       self.file_object = open(
            r"D:\ML_Projects\MultiClassClassification\Sentiments-2.0-Decode-Code-Words\logs\DataProcessingLogs.txt",
            "a+",
        )

    
    def apply_all_processing_on_train_test_data(self):
        self.logging.log(
            self.file_object,
            "In apply_all_processing_on_train_test_data method in DataProcessing class: Started data processing"
        )
        
        try:
            self.train_data["Review"] = self.train_data["Review"].apply(
                lambda x: self.Review_processing(x)
            )
            self.train_data["Review"] = self.train_data["Review"].apply(
                lambda x: self.remove_punctuation(x)
            )
            self.train_data["Review"] = self.train_data["Review"].apply(
                lambda x: self.remove_numbers(x)
            )
            self.train_data["Review"] = self.train_data["Review"].apply(
                lambda x: self.remove_special_characters(x)
            )
            self.train_data["Review"] = self.train_data["Review"].apply(
                lambda x: self.remove_short_words(x)
            )
            self.train_data["Review"] = self.train_data["Review"].apply(
                lambda x: self.remove_stopwords(x)
            )
            self.train_data["Review"] = self.train_data["Review"].apply(
                lambda x: self.lemmatization(x)
            )

            self.logging.log(
                self.file_object,
                "In apply_all_processing_on_train_test_data method in DataProcessing class: data processing completed"
            )
            return self.train_data

        except Exception as e:
            self.logging.log(
                self.file_object,
                f"In apply_all_processing_on_train_test_data method in DataProcessing class: Error in data processing: {e}"
            )
            raise e
            return None

    def Review_processing(self, Review):
        try:
            Review = Review.lower()
            Review = Review.replace("\n", " ")
            Review = Review.replace("\r", " ")
            Review = Review.replace("\t", " ")
            Review = Review.replace("\xa0", " ")
            Review = Review.replace("\u200b", " ")
            Review = Review.replace("\u200c", " ")
            Review = Review.replace("\u200d", " ")
            Review = Review.replace("\ufeff", " ")
            Review = Review.replace("\ufeef", " ")
        except Exception as e:
            self.logging.log(
                self.file_object,
                f"In review_processing method in DataProcessing class: Error in review processing {e}"
            )
            raise e
            return None
        return Review

    def stemming(self, Review):
        logging.info("Applying stemming methods on train and test data")
        try:
            Review = Review.split()
            ps = PorterStemmer()
            Review = [ps.stem(word) for word in Review]
            Review = " ".join(Review)
        except Exception as e:
            logging.error(
                "Error in applying stemming methods on train and test data")
            logging.error(e)
            return None
        return Review

    def lemmatization(self, Review):
        Review = Review.split()
        lem = WordNetLemmatizer()
        Review = [lem.lemmatize(word) for word in Review]
        Review = " ".join(Review)
        return Review

    def remove_stopwords(self, Review):
        Review = Review.split()
        stop_words = set(stopwords.words("english"))
        Review = [word for word in Review if not word in stop_words]
        Review = " ".join(Review)
        return Review

    def remove_punctuation(self, Review):
        # remove all punctuation except full stop, exclaimation mark and question mark
        Review = Review.split()
        Review = [word for word in Review if word.isalpha()]
        Review = " ".join(Review)
        return Review

    def remove_numbers(self, Review):
        Review = Review.split()
        Review = [word for word in Review if not word.isnumeric()]
        Review = " ".join(Review)
        return Review

    def remove_special_characters(self, Review):
        Review = Review.split()
        Review = [word for word in Review if word.isalpha()]
        Review = " ".join(Review)
        return Review

    def remove_short_words(self, Review):
        Review = Review.split()
        Review = [word for word in Review if len(word) > 2]
        Review = " ".join(Review)
        return Review

    def remove_stopwords_and_punctuation(self, Review):
        Review = Review.split()
        stop_words = set(stopwords.words("english"))
        Review = [word for word in Review if not word in stop_words]
        Review = [word for word in Review if word.isalpha()]
        Review = " ".join(Review)
        return Review

    def remove_stopwords_and_punctuation_and_numbers(self, Review):
        Review = Review.split()
        stop_words = set(stopwords.words("english"))
        Review = [word for word in Review if not word in stop_words]
        Review = [word for word in Review if word.isalpha()]
        Review = [word for word in Review if not word.isnumeric()]
        Review = " ".join(Review)
        return Review

    def remove_nan_values(self, df):
        # fill nan values with UNKOWN and return the dataframe
        df = df.fillna("UNKNOWN", inplace=True)
        


class DataDevelopment:
    def __init__(self) -> None:
        
        pass

    def divide_data(self, column_name, data):
        try:
            # ==============================================================================
            if column_name == "Components":
                X = data.drop(["Components"], axis=1)
                y = data["Components"]
                (
                    x_train_component,
                    x_test_component,
                    y_train_component,
                    y_test_component,
                ) = train_test_split(X, y, test_size=0.2, random_state=0)
                return (
                    x_train_component,
                    x_test_component,
                    y_train_component,
                    y_test_component,
                )
            # ==============================================================================
            if column_name == "Delivery and Customer Support":
                X = data.drop(
                    ["Delivery and Customer Support"], axis=1
                )
                y = data["Delivery and Customer Support"]
                (
                    x_train_DeliveryandCustomerSupport,
                    x_test_DeliveryandCustomerSupport,
                    y_train_DeliveryandCustomerSupport,
                    y_test_DeliveryandCustomerSupport,
                ) = train_test_split(X, y, test_size=0.2, random_state=0)
                return (
                    x_train_DeliveryandCustomerSupport,
                    x_test_DeliveryandCustomerSupport,
                    y_train_DeliveryandCustomerSupport,
                    y_test_DeliveryandCustomerSupport,
                )
            # ==============================================================================
            if column_name == "Design and Aesthetics":
                X = data.drop(["Design and Aesthetics"], axis=1)
                y = data["Design and Aesthetics"]
                (
                    x_train_DesignandAesthetics,
                    x_test_DesignandAesthetics,
                    y_train_DesignandAesthetics,
                    y_test_DesignandAesthetics,
                ) = train_test_split(X, y, test_size=0.2, random_state=0)
                return (
                    x_train_DesignandAesthetics,
                    x_test_DesignandAesthetics,
                    y_train_DesignandAesthetics,
                    y_test_DesignandAesthetics,
                )
            # ==============================================================================
            if column_name == "Dimensions":
                X = data.drop(["Dimensions"], axis=1)
                y = data["Dimensions"]
                (
                    x_train_Dimensions,
                    x_test_Dimensions,
                    y_train_Dimensions,
                    y_test_Dimensions,
                ) = train_test_split(X, y, test_size=0.2, random_state=0)
                return (
                    x_train_Dimensions,
                    x_test_Dimensions,
                    y_train_Dimensions,
                    y_test_Dimensions,
                )
            # ==============================================================================
            if column_name == "Features":
                X = data.drop(["Features"], axis=1)
                y = data["Features"]
                (
                    x_train_Features,
                    x_test_Features,
                    y_train_Features,
                    y_test_Features,
                ) = train_test_split(X, y, test_size=0.2, random_state=0)
                return x_train_Features, x_test_Features, y_train_Features, y_test_Features
            # ==============================================================================
            if column_name == "Functionality":
                X = data.drop(["Functionality"], axis=1)
                y = data["Functionality"]
                (
                    x_train_Functionality,
                    x_test_Functionality,
                    y_train_Functionality,
                    y_test_Functionality,
                ) = train_test_split(X, y, test_size=0.2, random_state=0)
                return (
                    x_train_Functionality,
                    x_test_Functionality,
                    y_train_Functionality,
                    y_test_Functionality,
                )
            # ==============================================================================
            if column_name == "Installation":
                X = data.drop(["Installation"], axis=1)
                y = data["Installation"]
                (
                    x_train_Installation,
                    x_test_Installation,
                    y_train_Installation,
                    y_test_Installation,
                ) = train_test_split(X, y, test_size=0.2, random_state=0)
                return (
                    x_train_Installation,
                    x_test_Installation,
                    y_train_Installation,
                    y_test_Installation,
                )
            # ==============================================================================
            if column_name == "Material":
                X = data.drop(["Material"], axis=1)
                y = data["Material"]
                (
                    x_train_Material,
                    x_test_Material,
                    y_train_Material,
                    y_test_Material,
                ) = train_test_split(X, y, test_size=0.2, random_state=0)
                return x_train_Material, x_test_Material, y_train_Material, y_test_Material
            # ==============================================================================
            if column_name == "Price":
                X = data.drop(["Price"], axis=1)
                y = data["Price"]
                x_train_Price, x_test_Price, y_train_Price, y_test_Price = train_test_split(
                    X, y, test_size=0.2, random_state=0
                )
                return x_train_Price, x_test_Price, y_train_Price, y_test_Price
            # ==============================================================================
            if column_name == "Quality":
                X = data.drop(["Quality"], axis=1)
                y = data["Quality"]
                (
                    x_train_Quality,
                    x_test_Quality,
                    y_train_Quality,
                    y_test_Quality,
                ) = train_test_split(X, y, test_size=0.2, random_state=0)
                return x_train_Quality, x_test_Quality, y_train_Quality, y_test_Quality
            # ==============================================================================
            if column_name == "Usability":
                X = data.drop(["Usability"], axis=1)
                y = data["Usability"]
                (
                    x_train_Usability,
                    x_test_Usability,
                    y_train_Usability,
                    y_test_Usability,
                ) = train_test_split(X, y, test_size=0.2, random_state=0)
                return (
                    x_train_Usability,
                    x_test_Usability,
                    y_train_Usability,
                    y_test_Usability,
                )
            # ==============================================================================
            if column_name == "Polarity":
                X = data.drop(["Polarity"], axis=1)
                y = data["Polarity"]
                (
                    x_train_Polarity,
                    x_test_Polarity,
                    y_train_Polarity,
                    y_test_Polarity,
                ) = train_test_split(X, y, test_size=0.2, random_state=0)
                return x_train_Polarity, x_test_Polarity, y_train_Polarity, y_test_Polarity
            # ==============================================================================
        except Exception as e:
            raise e


if __name__ == "__main__":
    data_utils = DataUtils()
    train_data, test_data = data_utils.read_data(
        train_path=r"D:\ML_Projects\MultiClassClassification\Sentiments-2.0-Decode-Code-Words\data\trainmulticlass.csv",
        test_path=r"D:\ML_Projects\MultiClassClassification\Sentiments-2.0-Decode-Code-Words\data\testmulticlass.csv"
    )

    data_process = DataProcessing(train_data)
    processed_train_data = data_process.apply_all_processing_on_train_test_data()
    data_process.remove_nan_values(train_data)

    print(processed_train_data.shape)
    print(processed_train_data.head())

    # data_utils = DataUtils()
    # train_data, test_data = data_utils.read_data(
    #     train_path=r"E:\Hackathon\UGAM\Participants_Data_DCW\train.csv",
    #     test_path=r"E:\Hackathon\UGAM\Participants_Data_DCW\test.csv",
    # )
    # Review_preprocessing = DataProcessing(test_data, train_data)
    # (
    #     train_data,
    #     test_data,
    # ) = Review_preprocessing.apply_all_processing_on_train_test_data()
    # train_data
    # train_data.to_csv(
    #     r"E:\Hackathon\UGAM\Participants_Data_DCW\processed_data\train_data_preprocessed.csv",
    #     index=False,
    # )
    # test_data.to_csv(
    #     r"E:\Hackathon\UGAM\Participants_Data_DCW\processed_data\test_data_preprocessed.csv",
    #     index=False,
    # )
    # # ==============================================================================
    # data_dev = DataDevelopment()
    # (
    #             x_train_component,
    #             x_test_component,
    #             y_train_component,
    #             y_test_component,
    #         ) = data_dev.divide_data(df="df_componenet")
    # x_train_component
    pass
