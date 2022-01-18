from data_ingestion import DataUtils
from data_processing import DataProcessing, DataDevelopment
from eda_src import EDA
from feature_engineering import FeatureEngineering, Vectorization
import pandas as pd
from model_development import TrainMLModel
from application_logger import CustomApplicationLogger

def main():
    # ======================================Data Ingestion===============================================
    data_utils = DataUtils()
    train_data, test_data = data_utils.read_data(
        train_path=r"D:\ML_Projects\MultiClassClassification\Sentiments-2.0-Decode-Code-Words\data\trainmulticlass.csv",
        test_path=r"D:\ML_Projects\MultiClassClassification\Sentiments-2.0-Decode-Code-Words\data\testmulticlass.csv"
    )
    # ======================================Data Processing==============================================
    data_preprocessing = DataProcessing(train_data)
    train_data = data_preprocessing.apply_all_processing_on_train_test_data()
    data_preprocessing.remove_nan_values(train_data)
    print(train_data.head())
    # ===========================================EDA===============================================
    # eda = EDA(train_data)
    # eda.check_class_distributions()
    # eda.cross_tabulation()
    # eda.basic_data_exploration()
    # eda.check_missing_values()
    # ======================================Feature Engineering===============================================
    feature_engineering = FeatureEngineering(train_data)
    train_data = feature_engineering.add_features()
    print(train_data.shape)

    model = feature_engineering.train_a_gensim_model()
    train_data = feature_engineering.make_acolumn(model)
    train_data["most_similar_words"] = train_data["most_similar_words"].apply(str)
    train_data["most_similar_words"] = train_data["most_similar_words"].apply(
        lambda x: feature_engineering.process_most_similar_words(x)
    )

    vectorization = Vectorization(train_data)
    final_train_data_with_similar_word_features = vectorization.extract_features_most_similar_words()
    fast_text_features = vectorization.fast_text_extract_features()

    print(final_train_data_with_similar_word_features.shape)
    print(final_train_data_with_similar_word_features.head())
    print("fast text : " , fast_text_features.shape)
    print(fast_text_features.head())
    #print(final_train_data_with_similar_word_features.columns[:30])

    # =====================================Data merging ===========================================
    final_train_data_with_similar_word_features.drop(["Id" , "Review"] , axis=1 , inplace=True)
    merged_train_data = pd.concat(
        [final_train_data_with_similar_word_features , fast_text_features],
        axis=1
    )
    print(merged_train_data.shape)
    print(merged_train_data.head())

    merged_train_data.to_csv("D:\ML_Projects\MultiClassClassification\Sentiments-2.0-Decode-Code-Words\data\merged_train_data.csv" , index=False)

    #====================================Data reduction====================================
    merged_train_data.dropna(inplace=True)
    train_target = merged_train_data.iloc[: , 0:12]
    train_features = merged_train_data.drop(merged_train_data.columns[:12] , axis=1)
    truncated_train_features = vectorization.reduce_features(train_features)
    # print(train_target.shape )
    # print(train_target.head())
    # print(truncated_train_features.shape)
    # print(truncated_train_features.head())
    #=================================Data divison into 12 Datasets======================
    
    ( df_Components,
    df_DeliveryandCustomerSupport,
    df_DesignAndAesthetics,
    df_Dimensions,
    df_Features,
    df_Functionality,
    df_Installation,
    df_Material,
    df_Price,
    df_Quality,
    df_Usability,
    df_Polarity) = data_utils.divide_data_in_divisions(truncated_train_features , train_target)
    print(df_Components.shape)
    print(df_Components.head())
    # ===============================Train Test split=================================
    data_dev = DataDevelopment()
    (
    x_train_component,
    x_test_component,
    y_train_component,
    y_test_component,
    ) = data_dev.divide_data("Components" , df_Components)
    print(x_train_component.shape, x_test_component.shape, y_train_component.shape, y_test_component.shape)
    
if __name__ == "__main__":
    main()