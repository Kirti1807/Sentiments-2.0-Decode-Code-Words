import numpy as np
import pandas as pd
import logging
from application_logger import CustomApplicationLogger

class DataUtils:
    def __init__(self) -> None:
        self.file_object = open(
            r"D:\ML_Projects\MultiClassClassification\Sentiments-2.0-Decode-Code-Words\logs\DataIngestionLogs.txt",
            "a+",
        )
        self.logging = CustomApplicationLogger()
        

    def read_data(self, train_path, test_path):
        self.logging.log(
            self.file_object,
            "In read_data method in DataUtils class : Started data loading"
        )
        try:

            train_data = pd.read_csv(train_path)
            test_data = pd.read_csv(test_path)
            self.logging.log(
                self.file_object,
                "In read_data method in DataUtils class : Data read successfully"
            )
            return train_data, test_data
        except Exception as e:
            self.logging.log(
                self.file_object,
                f"In read_data method in DataUtils class : Error in loading data: {e}",
            )
            raise e
    
    def divide_data_in_divisions(self , features , target):
        self.logging.log(
            self.file_object,
            "In divide_data_in_divisions method in DataUtils class: started data divison into 12 dataset"
        )
        try:
            df_Components = pd.concat([features , target["Components"]] , axis=1)
            df_Components.to_csv(
                r"D:\ML_Projects\MultiClassClassification\Sentiments-2.0-Decode-Code-Words\divided_dataset\df_Components.csv",
                index=False)

            df_DeliveryandCustomerSupport = pd.concat([features , target["Delivery and Customer Support"]], axis=1)
            df_DeliveryandCustomerSupport.to_csv(
                r"D:\ML_Projects\MultiClassClassification\Sentiments-2.0-Decode-Code-Words\divided_dataset\DeliveryandCustomerSupport.csv",
                index=False)

            df_DesignAndAesthetics = pd.concat([features , target["Design and Aesthetics"]], axis=1)
            df_DesignAndAesthetics.to_csv(
                r"D:\ML_Projects\MultiClassClassification\Sentiments-2.0-Decode-Code-Words\divided_dataset\DesignAndAesthetics.csv",
                index=False)
            
            df_Dimensions = pd.concat([features , target["Dimensions"]], axis=1)
            df_Dimensions.to_csv(
                r"D:\ML_Projects\MultiClassClassification\Sentiments-2.0-Decode-Code-Words\divided_dataset\Dimensions.csv",
                index=False)

            df_Features = pd.concat([features , target["Features"]], axis=1)
            df_Features.to_csv(
                r"D:\ML_Projects\MultiClassClassification\Sentiments-2.0-Decode-Code-Words\divided_dataset\Features.csv",
                index=False)

            df_Functionality = pd.concat([features , target["Functionality"]], axis=1)
            df_Functionality.to_csv(
                r"D:\ML_Projects\MultiClassClassification\Sentiments-2.0-Decode-Code-Words\divided_dataset\Functionality.csv",
                index=False)

            df_Installation = pd.concat([features , target["Installation"]], axis=1)
            df_Installation.to_csv(
                r"D:\ML_Projects\MultiClassClassification\Sentiments-2.0-Decode-Code-Words\divided_dataset\Installation.csv",
                index=False)

            df_Material = pd.concat([features , target["Material"]], axis=1)
            df_Material.to_csv(
                r"D:\ML_Projects\MultiClassClassification\Sentiments-2.0-Decode-Code-Words\divided_dataset\Material.csv",
                index=False)

            df_Price = pd.concat([features , target["Price"]], axis=1)
            df_Price.to_csv(
                r"D:\ML_Projects\MultiClassClassification\Sentiments-2.0-Decode-Code-Words\divided_dataset\Price.csv",
                index=False)

            df_Quality = pd.concat([features , target["Quality"]], axis=1)
            df_Quality.to_csv(
                r"D:\ML_Projects\MultiClassClassification\Sentiments-2.0-Decode-Code-Words\divided_dataset\Quality.csv",
                index=False)

            df_Usability = pd.concat([features , target["Usability"]], axis=1)
            df_Usability.to_csv(
                r"D:\ML_Projects\MultiClassClassification\Sentiments-2.0-Decode-Code-Words\divided_dataset\Usability.csv",
                index=False)

            df_Polarity = pd.concat([features , target["Polarity"]], axis=1)
            df_Polarity.to_csv(
                r"D:\ML_Projects\MultiClassClassification\Sentiments-2.0-Decode-Code-Words\divided_dataset\Polarity.csv",
                index=False)
            
            self.logging.log(
                self.file_object,
                "In divide_data_in_divisions method in DataUtils class: Completed data divison"
            )
            
            return (
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
            )
        except Exception as e:
            self.logging.log(
                self.file_object,
                f"In divide_data_in_divisions method in DataUtils class: Error in data divison: {e}"
            )
            raise e

    def load_different_divisions_data(self):
        self.logging.log(
            self.file_object,
            "In load_different_divisions_data method in DataUtils class: started loading 12 datasets"
        )
        try:
            df_Components = pd.read_csv(
                r"D:\ML_Projects\MultiClassClassification\Sentiments-2.0-Decode-Code-Words\divided_dataset\df_Components.csv"
                )

            df_DeliveryandCustomerSupport = pd.read_csv(
                r"D:\ML_Projects\MultiClassClassification\Sentiments-2.0-Decode-Code-Words\divided_dataset\DeliveryandCustomerSupport.csv"
                )

            df_DesignAndAesthetics = pd.read_csv(
                r"D:\ML_Projects\MultiClassClassification\Sentiments-2.0-Decode-Code-Words\divided_dataset\DesignAndAesthetics.csv"
                )
            
            df_Dimensions = pd.read_csv(
                r"D:\ML_Projects\MultiClassClassification\Sentiments-2.0-Decode-Code-Words\divided_dataset\Dimensions.csv"
                )

            df_Features = pd.read_csv(
                r"D:\ML_Projects\MultiClassClassification\Sentiments-2.0-Decode-Code-Words\divided_dataset\Features.csv"
                )

            df_Functionality = pd.read_csv(
                r"D:\ML_Projects\MultiClassClassification\Sentiments-2.0-Decode-Code-Words\divided_dataset\Functionality.csv"
                )

            df_Installation = pd.read_csv(
                r"D:\ML_Projects\MultiClassClassification\Sentiments-2.0-Decode-Code-Words\divided_dataset\Installation.csv"
                )

            df_Material = pd.read_csv(
                r"D:\ML_Projects\MultiClassClassification\Sentiments-2.0-Decode-Code-Words\divided_dataset\Material.csv"
                )

            df_Price = pd.read_csv(
                r"D:\ML_Projects\MultiClassClassification\Sentiments-2.0-Decode-Code-Words\divided_dataset\Price.csv"
                )

            df_Quality = pd.read_csv(
                r"D:\ML_Projects\MultiClassClassification\Sentiments-2.0-Decode-Code-Words\divided_dataset\Quality.csv"
                )

            df_Usability = pd.read_csv(
                r"D:\ML_Projects\MultiClassClassification\Sentiments-2.0-Decode-Code-Words\divided_dataset\Usability.csv"
                )

            df_Polarity = pd.read_csv(
                r"D:\ML_Projects\MultiClassClassification\Sentiments-2.0-Decode-Code-Words\divided_dataset\Polarity.csv"
                )
            
            self.logging.log(
                self.file_object,
                r"In load_different_divisions_data method in DataUtils class: Completed loading 12 datasets"
            )
            
            return (
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
            )
        except Exception as e:
            self.logging.log(
                self.file_object,
                f"In divide_data_in_divisions method in DataUtils class: Error in data divison: {e}"
            )
            raise e

    
if __name__ == "__main__":
    data_utils = DataUtils()
    train_data, test_data = data_utils.read_data()
    print(train_data.head())
    print(train_data.shape)
    print(test_data.head())
    print(test_data.shape)
    # (
    #     df_Components,
    #     df_DeliveryandCustomerSupport,
    #     df_DesignandAesthetics,
    #     df_Dimensions,
    #     df_Features,
    #     df_Functionality,
    #     df_Installation,
    #     df_Material,
    #     df_Price,
    #     df_Quality,
    #     df_Usability,
    #     df_Polarity,
    # ) = data_utils.divide_data_in_divisions()
    # # convert every df in csv file
    # (
    #     df_componenet,
    #     df_DeliveryandCustomerSupport,
    #     df_Designand_Aesthetics,
    #     df_Dimensions,
    #     df_Features,
    #     df_Functionality,
    #     df_Installation,
    #     df_Material,
    #     df_Price,
    #     df_Quality,
    #     df_Usability,
    #     df_Polarity,
    # ) = data_utils.load_different_divisions_data()
    # df_componenet.head()
