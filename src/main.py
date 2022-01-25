from numpy import matrix
from data_ingestion import DataUtils
from data_processing import DataProcessing, DataDevelopment
from eda_src import EDA
from feature_engineering import FeatureEngineering, Vectorization
from evaluation import EvaluateModel
import pandas as pd
from model_development import TrainMLModel
from application_logger import CustomApplicationLogger
from model_training import TrainedModel
import joblib

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
    # print(df_Components.shape)
    # print(df_Components.head())
    # ===============================Model Training=================================
    model_training = TrainedModel()
    (
        model_components,
        model_DeliveryandCustomerSupport,
        model_DesignandAesthetics,
        model_Dimensions,
        model_Features,
        model_Functionality,
        model_Installation,
        model_Material,
        model_Price,
        model_Quality,
        model_Usability,
        model_Polarity
    ) = model_training.getting_trained_model()
    
    # =========================================== Model Training And Evaluation ===============================================
    # evaluate_xg = EvaluateModel(x_test, y_test.values, xg_model)
    # evaluate_xg.evaluate_model()
    

def predict(msg):
    df = pd.DataFrame({'Review':[msg]} , index=[0])
    data_preprocessing = DataProcessing(df)
    processed_df = data_preprocessing.apply_all_processing_on_train_test_data()
    feature_engineering = FeatureEngineering(processed_df)
    processed_df = feature_engineering.add_features()

    # model = feature_engineering.train_a_gensim_model()
    from gensim.models import Word2Vec
    gensim_model = Word2Vec.load(r"D:\ML_Projects\MultiClassClassification\Sentiments-2.0-Decode-Code-Words\saved_model\ugam_reviews.model")
    # model.train([["hello", "world"]], total_examples=1, epochs=1)
    # gensim_model = joblib.load("D:\ML_Projects\MultiClassClassification\Sentiments-2.0-Decode-Code-Words\saved_model\ugam_reviews.model")
    train_data = feature_engineering.make_acolumn(gensim_model)
    train_data["most_similar_words"] = train_data["most_similar_words"].apply(str)
    train_data["most_similar_words"] = train_data["most_similar_words"].apply(
        lambda x: feature_engineering.process_most_similar_words(x)
    )
    
    vectorizer_similar_word = joblib.load(r"D:\ML_Projects\MultiClassClassification\Sentiments-2.0-Decode-Code-Words\vectorizer_similarFeatures.pkl")
    extracted_data = list(
        vectorizer_similar_word.transform(
            train_data["most_similar_words"]).toarray()
    )
    extracted_data = pd.DataFrame(extracted_data)
    #extracted_data.head()
    extracted_data.columns = vectorizer_similar_word.get_feature_names()

    from gensim.models import FastText
    vectorizer_fast_text = FastText.load(r"D:\ML_Projects\MultiClassClassification\Sentiments-2.0-Decode-Code-Words\ft_model.model")
    # vectorizer_fast_text = joblib.load("D:\ML_Projects\MultiClassClassification\Sentiments-2.0-Decode-Code-Words\ft_model.model")

    tokenized_docs_train = [doc.split()
        for doc in list(train_data['Review'])
    ]
    
    vectorizer_obj = Vectorization(train_data)
    doc_vecs_ft_train = vectorizer_obj.averaged_word2vec_vectorizer(
        tokenized_docs_train, vectorizer_fast_text, 300)
    doc_vecs_ft_train = pd.DataFrame(doc_vecs_ft_train)
    
    train_data.drop(["Review" , "most_similar_words"] , axis=1 , inplace=True)
    merged_data = pd.concat([train_data , extracted_data , doc_vecs_ft_train] , axis=1)

    svd_model = joblib.load("D:\ML_Projects\MultiClassClassification\Sentiments-2.0-Decode-Code-Words\saved_model\SVD.sav")
    truncated_merged_data = svd_model.transform(merged_data)
    truncated_merged_data=pd.DataFrame(truncated_merged_data)
    
    # model loading
    model_components = joblib.load("D:\ML_Projects\MultiClassClassification\Sentiments-2.0-Decode-Code-Words\saved_model\XGB_Components.pkl")
    model_DeliveryandCustomerSupport = joblib.load("D:\ML_Projects\MultiClassClassification\Sentiments-2.0-Decode-Code-Words\saved_model\XGB_DeliveryandCustomerSupport.pkl")
    model_DesignandAesthetics = joblib.load("D:\ML_Projects\MultiClassClassification\Sentiments-2.0-Decode-Code-Words\saved_model\XGB_DesignandAesthetics.pkl")    
    model_Dimensions = joblib.load("D:\ML_Projects\MultiClassClassification\Sentiments-2.0-Decode-Code-Words\saved_model\XGB_Dimensions.pkl")
    model_Features = joblib.load("D:\ML_Projects\MultiClassClassification\Sentiments-2.0-Decode-Code-Words\saved_model\XGB_Features.pkl")
    model_Functionality = joblib.load("D:\ML_Projects\MultiClassClassification\Sentiments-2.0-Decode-Code-Words\saved_model\XGB_Functionality.pkl")
    model_Installation = joblib.load("D:\ML_Projects\MultiClassClassification\Sentiments-2.0-Decode-Code-Words\saved_model\XGB_Installation.pkl")
    model_Material = joblib.load("D:\ML_Projects\MultiClassClassification\Sentiments-2.0-Decode-Code-Words\saved_model\XGB_Material.pkl")
    model_Price = joblib.load("D:\ML_Projects\MultiClassClassification\Sentiments-2.0-Decode-Code-Words\saved_model\XGB_Price.pkl")
    model_Quality = joblib.load("D:\ML_Projects\MultiClassClassification\Sentiments-2.0-Decode-Code-Words\saved_model\XGB_Quality.pkl")
    model_Usability = joblib.load("D:\ML_Projects\MultiClassClassification\Sentiments-2.0-Decode-Code-Words\saved_model\XGB_Usability.pkl")
    model_Polarity = joblib.load("D:\ML_Projects\MultiClassClassification\Sentiments-2.0-Decode-Code-Words\saved_model\XGB_Polarity.pkl")
    
    # predictions
    components = model_components.predict(truncated_merged_data)
    DeliveryandCustomerSupport = model_DeliveryandCustomerSupport.predict(truncated_merged_data)
    DesignandAesthetics = model_DesignandAesthetics.predict(truncated_merged_data)
    Dimensions = model_Dimensions.predict(truncated_merged_data)
    Features = model_Features.predict(truncated_merged_data)
    Functionality = model_Functionality.predict(truncated_merged_data)
    Installation = model_Installation.predict(truncated_merged_data)
    Material = model_Material.predict(truncated_merged_data)
    Polarity = model_Polarity.predict(truncated_merged_data)
    Price = model_Price.predict(truncated_merged_data)
    Quality = model_Quality.predict(truncated_merged_data)
    Usability = model_Usability.predict(truncated_merged_data)
    
    columns = ["Components", "Delivery and Customer Support" , "Design and Aesthetics", "Dimensions","Features","Functionality","Installation","Material","Price","Quality","Usability","Polarity"]
    prediction = [components , DeliveryandCustomerSupport , DesignandAesthetics , Dimensions , Features , Functionality, Installation , Material , Price , Quality , Usability , Polarity]
    result=pd.DataFrame([prediction], columns=columns)
    
    return result

    

if __name__ == "__main__":
    # main()
    df = predict("I am the new family plumber. Works well. No problems changing out valves in the rental houses.")
    print(df.shape)
    print(df.head())