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

from data_ingestion import DataUtils
from sklearn.model_selection import train_test_split


class DataProcessing:
    def __init__(self, test_data, train_data) -> None:
        self.test_data = test_data
        self.train_data = train_data

    # different different methods for Review processing like stemming lemmatization, words removal and etc

    def apply_all_processing_on_train_test_data(self):
        # apply all the Review processing methods on train and test data
        logging.info(
            "Applying all the Review processing methods on train and test data"
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

            self.test_data["Review"] = self.test_data["Review"].apply(
                lambda x: self.remove_numbers(x)
            )

            self.test_data["Review"] = self.test_data["Review"].apply(
                lambda x: self.remove_short_words(x)
            )
            self.test_data["Review"] = self.test_data["Review"].apply(
                lambda x: self.remove_stopwords(x)
            )
            self.test_data["Review"] = self.test_data["Review"].apply(
                lambda x: self.lemmatization(x)
            )
            return self.train_data, self.test_data

        except Exception as e:
            logging.error(
                "Error in applying all the Review processing methods on train and test data"
            )
            logging.error(e)
            return None

    def Review_processing(self, Review):
        logging.info("Applying Review processing methods on train and test data")
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
            logging.error(
                "Error in applying Review processing methods on train and test data"
            )
            logging.error(e)
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
            logging.error("Error in applying stemming methods on train and test data")
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
        df = df.fillna("UNKOWN")     
        return df 
 

class DataDevelopment:
    def __init__(self) -> None:
        pass

    def divide_data(self, df, data):
        data_utils = DataUtils()
        (
            df_componenet,
            df_DeliveryandCustomerSupport,
            df_DesignandAesthetics,
            df_Dimensions,
            df_Features,
            df_Functionality,
            df_Installation,
            df_Material,
            df_Price,
            df_Quality,
            df_Usability,
            df_Polarity,
        ) = data_utils.divide_data_in_divisions()

        # ==============================================================================
        if df == "df_componenet":
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
        if df == "df_DeliveryandCustomerSupport":
            X = df_DeliveryandCustomerSupport.drop(
                ["Delivery and Customer Support"], axis=1
            )
            y = df_DeliveryandCustomerSupport["Delivery and Customer Support"]
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
        if df == "df_DesignandAesthetics":
            X = df_DesignandAesthetics.drop(["Design and Aesthetics"], axis=1)
            y = df_DesignandAesthetics["Design and Aesthetics"]
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
        if df == "df_Dimensions":
            X = df_Dimensions.drop(["Dimensions"], axis=1)
            y = df_Dimensions["Dimensions"]
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
        if df == "df_Features":
            X = df_Features.drop(["Features"], axis=1)
            y = df_Features["Features"]
            (
                x_train_Features,
                x_test_Features,
                y_train_Features,
                y_test_Features,
            ) = train_test_split(X, y, test_size=0.2, random_state=0)
            return x_train_Features, x_test_Features, y_train_Features, y_test_Features
        # ==============================================================================
        if df == "df_Functionality":
            X = df_Functionality.drop(["Functionality"], axis=1)
            y = df_Functionality["Functionality"]
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
        if df == "df_Installation":
            X = df_Installation.drop(["Installation"], axis=1)
            y = df_Installation["Installation"]
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
        if df == "df_Material":
            X = df_Material.drop(["Material"], axis=1)
            y = df_Material["Material"]
            (
                x_train_Material,
                x_test_Material,
                y_train_Material,
                y_test_Material,
            ) = train_test_split(X, y, test_size=0.2, random_state=0)
            return x_train_Material, x_test_Material, y_train_Material, y_test_Material
        # ==============================================================================
        if df == "df_Price":
            X = df_Price.drop(["Price"], axis=1)
            y = df_Price["Price"]
            x_train_Price, x_test_Price, y_train_Price, y_test_Price = train_test_split(
                X, y, test_size=0.2, random_state=0
            )
            return x_train_Price, x_test_Price, y_train_Price, y_test_Price
        # ==============================================================================
        if df == "df_Quality":
            X = df_Quality.drop(["Quality"], axis=1)
            y = df_Quality["Quality"]
            (
                x_train_Quality,
                x_test_Quality,
                y_train_Quality,
                y_test_Quality,
            ) = train_test_split(X, y, test_size=0.2, random_state=0)
            return x_train_Quality, x_test_Quality, y_train_Quality, y_test_Quality
        # ==============================================================================
        if df == "df_Usability":
            X = df_Usability.drop(["Usability"], axis=1)
            y = df_Usability["Usability"]
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
        if df == "df_Polarity":
            X = df_Polarity.drop(["Polarity"], axis=1)
            y = df_Polarity["Polarity"]
            (
                x_train_Polarity,
                x_test_Polarity,
                y_train_Polarity,
                y_test_Polarity,
            ) = train_test_split(X, y, test_size=0.2, random_state=0)
            return x_train_Polarity, x_test_Polarity, y_train_Polarity, y_test_Polarity
        # ==============================================================================


if __name__ == "__main__":
    data_utils = DataUtils()
    train_data , test_data = data_utils.read_data()
    
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