from model_development import TrainMLModel
from application_logger import CustomApplicationLogger
from data_ingestion import DataUtils
from data_processing import DataDevelopment
import joblib

class TrainedModel:
    def __init__(self):
        
        self.file_object = open(
            r"D:\ML_Projects\MultiClassClassification\Sentiments-2.0-Decode-Code-Words\logs\ModelTrainingLogs.txt",
            "a+",
        )
        self.logging = CustomApplicationLogger()

    def getting_trained_model(self):
        self.logging.log(
            self.file_object,
            "In getting_trained_model method in TrainedModel class: starting the training of models"
        )

        try:
            data_utils = DataUtils()
            (
                df_Components,
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
                df_Polarity
            ) = data_utils.load_different_divisions_data()
            
            data_dev = DataDevelopment()
            # training model on df_Components data
            df_Components.dropna(inplace=True)
            (
                x_train_component,
                x_test_component,
                y_train_component,
                y_test_component,
            ) = data_dev.divide_data("Components" , df_Components)
            model_train_1 = TrainMLModel(x_train_component , x_test_component , y_train_component, y_test_component)
            xgb_components = model_train_1.xgboost(fine_tuning=True)

            joblib.dump(
                xgb_components,
                "D:\ML_Projects\MultiClassClassification\Sentiments-2.0-Decode-Code-Words\saved_model\XGB_Components.pkl"
            )

            # training model on df_DeliveryandCustomerSupport data
            df_DeliveryandCustomerSupport.dropna(inplace=True)
            (
                x_train_DeliveryandCustomerSupport,
                x_test_DeliveryandCustomerSupport,
                y_train_DeliveryandCustomerSupport,
                y_test_DeliveryandCustomerSupport,
            ) = data_dev.divide_data("Delivery and Customer Support" , df_DeliveryandCustomerSupport)
            model_train_2 = TrainMLModel(x_train_DeliveryandCustomerSupport , x_test_DeliveryandCustomerSupport , y_train_DeliveryandCustomerSupport , y_test_DeliveryandCustomerSupport)
            xgb_DeliveryandCustomerSupport = model_train_2.xgboost(fine_tuning=True)

            joblib.dump(
                xgb_DeliveryandCustomerSupport,
                "D:\ML_Projects\MultiClassClassification\Sentiments-2.0-Decode-Code-Words\saved_model\XGB_DeliveryandCustomerSupport.pkl"
            )

            # training model of df_DesignAndAesthetics data
            df_DesignAndAesthetics.dropna(inplace=True)
            (
                x_train_DesignandAesthetics,
                x_test_DesignandAesthetics,
                y_train_DesignandAesthetics,
                y_test_DesignandAesthetics,
            ) = data_dev.divide_data("Design and Aesthetics" , df_DesignAndAesthetics)
            model_train_3 = TrainMLModel(x_train_DesignandAesthetics , x_test_DesignandAesthetics , y_train_DesignandAesthetics , y_test_DesignandAesthetics )
            xgb_DesignandAesthetics = model_train_3.xgboost(fine_tuning=True)

            joblib.dump(
                xgb_DesignandAesthetics,
                "D:\ML_Projects\MultiClassClassification\Sentiments-2.0-Decode-Code-Words\saved_model\XGB_DesignandAesthetics.pkl"
            )

            # training model on df_Dimensions data
            df_Dimensions.dropna(inplace=True)
            (
                x_train_Dimensions,
                x_test_Dimensions,
                y_train_Dimensions,
                y_test_Dimensions,
            ) = data_dev.divide_data("Dimensions" , df_Dimensions)
            model_train_4 = TrainMLModel(x_train_Dimensions , x_test_Dimensions , y_train_Dimensions , y_test_Dimensions)
            xgb_Dimensions = model_train_4.xgboost(fine_tuning=True)

            joblib.dump(
                xgb_Dimensions,
                "D:\ML_Projects\MultiClassClassification\Sentiments-2.0-Decode-Code-Words\saved_model\XGB_Dimensions.pkl"
            )

            # training model on df_Features data
            df_Features.dropna(inplace=True)
            (
                x_train_Features,
                x_test_Features,
                y_train_Features,
                y_test_Features,
            ) = data_dev.divide_data("Features" , df_Features)
            model_train_5 = TrainMLModel(x_train_Features , x_test_Features , y_train_Features , y_test_Features)
            xgb_Features = model_train_5.xgboost(fine_tuning=True)

            joblib.dump(
                xgb_Features,
                "D:\ML_Projects\MultiClassClassification\Sentiments-2.0-Decode-Code-Words\saved_model\XGB_Features.pkl"
            )

            
            # training model on df_functionality data
            df_Functionality.dropna(inplace=True)
            (
                x_train_Functionality,
                x_test_Functionality,
                y_train_Functionality,
                y_test_Functionality
            ) = data_dev.divide_data("Functionality" , df_Functionality)
            model_train_6 = TrainMLModel(x_train_Functionality , x_test_Functionality , y_train_Functionality , y_test_Functionality)
            xgb_Functionality = model_train_6.xgboost(fine_tuning=True)

            joblib.dump(
                xgb_Functionality,
                "D:\ML_Projects\MultiClassClassification\Sentiments-2.0-Decode-Code-Words\saved_model\XGB_Functionality.pkl"
            )
            
            # training model on df_Installation data
            df_Installation.dropna(inplace=True)
            (
                x_train_Installation,
                x_test_Installation,
                y_train_Installation,
                y_test_Installation
            ) = data_dev.divide_data("Installation" , df_Installation)
            model_train_7 = TrainMLModel(x_train_Installation,
                x_test_Installation,
                y_train_Installation,
                y_test_Installation)
            xgb_Installation = model_train_7.xgboost(fine_tuning=True)

            joblib.dump(
                xgb_Installation,
                "D:\ML_Projects\MultiClassClassification\Sentiments-2.0-Decode-Code-Words\saved_model\XGB_Installation.pkl"
            )
            
            # training model on df_Material data
            df_Material.dropna(inplace=True)
            (
                x_train_Material,
                x_test_Material,
                y_train_Material,
                y_test_Material
            ) = data_dev.divide_data("Material" , df_Material)
            model_train_8 = TrainMLModel(x_train_Material,
                x_test_Material,
                y_train_Material,
                y_test_Material)
            xgb_Material = model_train_8.xgboost(fine_tuning=True)

            joblib.dump(
                xgb_Material,
                "D:\ML_Projects\MultiClassClassification\Sentiments-2.0-Decode-Code-Words\saved_model\XGB_Material.pkl"
            )
            
            # training model on df_Price data
            df_Price.dropna(inplace=True)
            (
                x_train_Price, x_test_Price, y_train_Price, y_test_Price
            ) = data_dev.divide_data("Price" , df_Price)
            model_train_9 = TrainMLModel(x_train_Price, x_test_Price, y_train_Price, y_test_Price)
            xgb_Price = model_train_9.xgboost(fine_tuning=True)

            joblib.dump(
                xgb_Price,
                "D:\ML_Projects\MultiClassClassification\Sentiments-2.0-Decode-Code-Words\saved_model\XGB_Price.pkl"
            )

            # training model on df_Quality data
            df_Quality.dropna(inplace=True)
            (
                x_train_Quality,
                x_test_Quality,
                y_train_Quality,
                y_test_Quality
            ) = data_dev.divide_data("Quality" , df_Quality)
            model_train_10 = TrainMLModel(x_train_Quality,
                x_test_Quality,
                y_train_Quality,
                y_test_Quality)
            xgb_Quality = model_train_10.xgboost(fine_tuning=True)

            joblib.dump(
                xgb_Quality,
                "D:\ML_Projects\MultiClassClassification\Sentiments-2.0-Decode-Code-Words\saved_model\XGB_Quality.pkl"
            )

            # training model on df_Usability data
            df_Usability.dropna(inplace=True)
            (
                x_train_Usability,
                x_test_Usability,
                y_train_Usability,
                y_test_Usability,
            ) = data_dev.divide_data("Usability" , df_Usability)
            model_train_11 = TrainMLModel(x_train_Usability , x_test_Usability , y_train_Usability , y_test_Usability)
            xgb_Usability = model_train_11.xgboost(fine_tuning=True)

            joblib.dump(
                xgb_Usability,
                "D:\ML_Projects\MultiClassClassification\Sentiments-2.0-Decode-Code-Words\saved_model\XGB_Usability.pkl"
            )

            # training model on df_Polarity data
            df_Polarity.dropna(inplace=True)
            (
                x_train_Polarity,
                x_test_Polarity,
                y_train_Polarity,
                y_test_Polarity,
            ) = data_dev.divide_data("Polarity" , df_Polarity)
            model_train_12 = TrainMLModel(x_train_Polarity , x_test_Polarity , y_train_Polarity , y_test_Polarity)
            xgb_Polarity = model_train_12.xgboost(fine_tuning=True)

            joblib.dump(
                xgb_Polarity,
                "D:\ML_Projects\MultiClassClassification\Sentiments-2.0-Decode-Code-Words\saved_model\XGB_Polarity.pkl"
            )

            self.logging.log(
                self.file_object,
                "In getting_trained_model method in TrainedModel class: Successfully completed model training"
            )

            return (
                xgb_components,
                xgb_DeliveryandCustomerSupport,
                xgb_DesignandAesthetics,
                xgb_Dimensions,
                xgb_Features,
                xgb_Functionality,
                xgb_Installation,
                xgb_Material,
                xgb_Price,
                xgb_Quality,
                xgb_Usability,
                xgb_Polarity
            )

        except Exception as e:
            self.logging.log(
                self.file_object,
                f"In getting_trained_model method in TrainedModel class: Error in model training: {e}"
            )
            raise e