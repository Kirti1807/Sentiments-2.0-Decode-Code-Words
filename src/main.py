from data_ingestion import DataUtils
from data_processing import DataProcessing, DataDevelopment
from feature_engineering import FeatureEngineering, Vectorization
import pandas as pd
from model_development import TrainMLModel


def main():
    # =====================================================================================
    data_utils = DataUtils()
    train_data, test_data = data_utils.read_data(
        # E:\Hackathon\UGAM\Participants_Data_DCW\processed_data\train_data_with_most_similar_words_processed.csv
        train_path=r"C:\Users\shreyas.dhaware2301\Documents\antern\Sentiments-2.0-Decode-Code-Words\dataset\trainmulticlass.csv",
        # E:\Hackathon\UGAM\Participants_Data_DCW\processed_data\test_data_with_most_similar_words_processed.csv
        test_path=r"C:\Users\shreyas.dhaware2301\Documents\antern\Sentiments-2.0-Decode-Code-Words\dataset\testmulticlass.csv",
    )
    # =====================================================================================
    train_data.head()
    test_data.head()
    # train_data = train_data.iloc[:100]
    # test_data = test_data.iloc[:100]
    data_preprocessing = DataProcessing(train_data, test_data)
    # train_data, test_data = data_preprocessing.apply_all_processing_on_train_test_data()
    train_data.to_csv("cleaned_train_data.csv")
    test_data.to_csv("cleaned_test_data.csv")
    train_data_preprocessed = pd.read_csv(
        r"E:\Hackathon\UGAM\Participants_Data_DCW\processed_data\train_data_preprocessed.csv"
    )
    train_data["Review"] = train_data_preprocessed["Review"]
    test_data_preprocessed = pd.read_csv(
        r"E:\Hackathon\UGAM\Participants_Data_DCW\processed_data\test_data_preprocessed.csv"
    )
    test_data["Review"] = test_data_preprocessed["Review"]
    train_data = data_preprocessing.remove_nan_values(train_data)
    test_data = data_preprocessing.remove_nan_values(test_data)
    # =====================================================================================

    feature_engineering = Vectorization(train_data, test_data)
    # Final_Training_data, Final_Test = feature_engineering.tfidf_extract_features()
    # =====================================================================================
    Final_Training_data, Final_Test = feature_engineering.fast_text_extract_features()

    feature_engineering_obj = FeatureEngineering(train_data, test_data)
    model = feature_engineering_obj.train_a_gensim_model()
    # model.wv.most_similar("good")

    train_data, test_data = feature_engineering_obj.make_acolumn(model)
    # test_data.to_csv(
    #     r"E:\Hackathon\UGAM\Participants_Data_DCW\processed_data\test_data_with_most_similar_words.csv",
    #     index=False,
    # )
    # train_data.to_csv(
    #     r"E:\Hackathon\UGAM\Participants_Data_DCW\processed_data\train_data_with_most_similar_words.csv",
    #     index=False,
    # )
    train_data["most_similar_words"]
    train_data["most_similar_words"] = train_data["most_similar_words"].apply(
        str)
    test_data["most_similar_words"] = test_data["most_similar_words"].apply(
        str)

    train_data["most_similar_words"] = train_data["most_similar_words"].apply(
        lambda x: feature_engineering_obj.process_most_similar_words(x)
    )
    test_data["most_similar_words"] = test_data["most_similar_words"].apply(
        lambda x: feature_engineering_obj.process_most_similar_words(x)
    )
    # train_data.to_csv(
    #     r"E:\Hackathon\UGAM\Participants_Data_DCW\processed_data\train_data_with_most_similar_words_processed.csv",
    #     index=False,
    # )
    # test_data.to_csv(
    #     r"E:\Hackathon\UGAM\Participants_Data_DCW\processed_data\test_data_with_most_similar_words_processed.csv",
    #     index=False,
    # )

    train_data, test_data = feature_engineering_obj.add_features()
    train_data.to_csv(
        r"E:\Hackathon\UGAM\Participants_Data_DCW\processed_data\train_data_with_most_similar_words_processed_with_features.csv",
        index=False,
    )
    test_data.to_csv(
        r"E:\Hackathon\UGAM\Participants_Data_DCW\processed_data\test_data_with_most_similar_words_processed_with_features.csv",
        index=False,
    )

    (Final_Training_data_similar,
     Final_Test_similar) = feature_engineering.extract_features_most_similar_words()
    # drop the first 14 columns as they are not useful
    Final_Training_data_similar.drop(
        Final_Training_data_similar.columns[:14], axis=1, inplace=True
    )
    Final_Test_similar.drop(
        Final_Test_similar.columns[:14], axis=1, inplace=True)

    # =====================================================================================

    # concatenate the two dataframes Final_Training_data and Final_Training_data_similar
    Final_Training_data.drop(["Id"], axis=1, inplace=True)
    Final_Training_data = pd.concat(
        [Final_Training_data, Final_Training_data_similar], axis=1
    )
    Final_Test.drop(["Id"], axis=1, inplace=True)
    Final_Test = pd.concat([Final_Test, Final_Test_similar], axis=1)

    # =====================================================================================

    data_dev = DataDevelopment()
    (x_train_component, x_test_component, y_train_component,
     y_test_component) = data_dev.divide_data("df_componenet", Final_Training_data)
    # drop the first 11 columns as they are not useful
    x_train_component.drop(
        x_train_component.columns[:11], axis=1, inplace=True)
    x_test_component.drop(x_test_component.columns[:11], axis=1, inplace=True)
    x_train_component.drop(["most_similar_words"], axis=1, inplace=True)
    x_test_component.drop(["most_similar_words"], axis=1, inplace=True)

    # =====================================================================================

    model_dev = TrainMLModel(
        x_train_component, x_test_component, y_train_component, y_test_component)
    decision_tree = model_dev.random_forest()

    x_train_component.reshape(-1, 1)
    x_test_component.shape[0]
    x_test_component
    y_train_component.shape[0]
    y_test_component.shape[0]
