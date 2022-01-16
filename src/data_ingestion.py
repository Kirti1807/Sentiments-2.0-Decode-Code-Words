import numpy as np
import pandas as pd
import logging


class DataUtils:
    def __init__(self) -> None:
        pass

    def read_data(self, train_path, test_path):
        logging.info("Reading training and testing data")
        train_data = pd.read_csv(train_path)
        test_data = pd.read_csv(test_path)
        logging.info("Data read successfully")
        return train_data, test_data

    # def divide_data_in_divisions(self):
    #     logging.info("Divide data into divisions")
    #     try:
    #         train_data, test_data = self.read_data(
    #             train_path=r"D:\ML_Projects\MultiClassClassification\Sentiments-2.0-Decode-Code-Words\data\trainmulticlass.csv",
    #             test_path=r"D:\ML_Projects\MultiClassClassification\Sentiments-2.0-Decode-Code-Words\data\testmulticlass.csv",
    #         )
    #         # select ID, Review, Component columns
    #         last7cols = train_data.iloc[:, -7:]

    #         logging.info("Dividing data into 12 divisions")
    #         df_componenet = pd.concat(
    #             [train_data[["Review", "Components"]], last7cols], axis=1
    #         )
    #         df_DeliveryandCustomerSupport = pd.concat(
    #             [train_data[["Review", "Delivery and Customer Support"]], last7cols],
    #             axis=1,
    #         )
    #         df_DesignandAesthetics = pd.concat(
    #             [train_data[["Review", "Design and Aesthetics"]], last7cols], axis=1
    #         )
    #         df_Dimensions = pd.concat(
    #             [train_data[["Review", "Dimensions"]], last7cols], axis=1
    #         )
    #         df_Features = pd.concat(
    #             [train_data[["Review", "Features"]], last7cols], axis=1
    #         )
    #         df_Functionality = pd.concat(
    #             [train_data[["Review", "Functionality"]], last7cols], axis=1
    #         )
    #         df_Installation = pd.concat(
    #             [train_data[["Review", "Installation"]], last7cols], axis=1
    #         )
    #         df_Material = pd.concat(
    #             [train_data[["Review", "Material"]], last7cols], axis=1
    #         )
    #         df_Quality = pd.concat(
    #             [train_data[["Review", "Quality"]], last7cols], axis=1
    #         )
    #         df_Price = pd.concat([train_data[["Review", "Price"]], last7cols], axis=1)
    #         df_Usability = pd.concat(
    #             [train_data[["Review", "Usability"]], last7cols], axis=1
    #         )
    #         df_Polarity = pd.concat(
    #             [train_data[["Review", "Polarity"]], last7cols], axis=1
    #         )

    #         logging.info("Data divided successfully")

    #         return (
    #             df_componenet,
    #             df_DeliveryandCustomerSupport,
    #             df_DesignandAesthetics,
    #             df_Dimensions,
    #             df_Features,
    #             df_Functionality,
    #             df_Installation,
    #             df_Material,
    #             df_Price,
    #             df_Quality,
    #             df_Usability,
    #             df_Polarity,
    #         )
    #         # convert every df in csv file
    #         # for df in df_list:
    #         #     df.to_csv(
    #         #         r"E:\Hackathon\UGAM\Participants_Data_DCW\data_divisons\data_divisons{}.csv".format(
    #         #             df.columns[2]
    #         #         ),
    #         #         index=False,
    #         #     )

    #         logging.info("Data divided successfully")

    #     except Exception as e:
    #         logging.error("Error while dividing data")
    #         logging.error(e)

    # def load_different_divisions_data(self):
    #     logging.info("Loading data from different divisions")
    #     try:
    #         logging.info("Loading data from different divisions")
    #         df_componenet = pd.read_csv(
    #             r"E:\Hackathon\UGAM\Participants_Data_DCW\data_divisons\data_divisonsComponents.csv"
    #         )
    #         df_DeliveryandCustomerSupport = pd.read_csv(
    #             r"E:\Hackathon\UGAM\Participants_Data_DCW\data_divisons\data_divisonsDelivery and Customer Support.csv"
    #         )
    #         df_Designand_Aesthetics = pd.read_csv(
    #             r"E:\Hackathon\UGAM\Participants_Data_DCW\data_divisons\data_divisonsDesign and Aesthetics.csv"
    #         )
    #         df_Dimensions = pd.read_csv(
    #             r"E:\Hackathon\UGAM\Participants_Data_DCW\data_divisons\data_divisonsDimensions.csv"
    #         )
    #         df_Features = pd.read_csv(
    #             r"E:\Hackathon\UGAM\Participants_Data_DCW\data_divisons\data_divisonsFeatures.csv"
    #         )
    #         df_Functionality = pd.read_csv(
    #             r"E:\Hackathon\UGAM\Participants_Data_DCW\data_divisons\data_divisonsFunctionality.csv"
    #         )
    #         df_Installation = pd.read_csv(
    #             r"E:\Hackathon\UGAM\Participants_Data_DCW\data_divisons\data_divisonsInstallation.csv"
    #         )
    #         df_Material = pd.read_csv(
    #             r"E:\Hackathon\UGAM\Participants_Data_DCW\data_divisons\data_divisonsMaterial.csv"
    #         )
    #         df_Price = pd.read_csv(
    #             r"E:\Hackathon\UGAM\Participants_Data_DCW\data_divisons\data_divisonsPrice.csv"
    #         )
    #         df_Quality = pd.read_csv(
    #             r"E:\Hackathon\UGAM\Participants_Data_DCW\data_divisons\data_divisonsQuality.csv"
    #         )
    #         df_Usability = pd.read_csv(
    #             r"E:\Hackathon\UGAM\Participants_Data_DCW\data_divisons\data_divisonsUsability.csv"
    #         )
    #         df_Polarity = pd.read_csv(
    #             r"E:\Hackathon\UGAM\Participants_Data_DCW\data_divisons\data_divisonsPolarity.csv"
    #         )
    #         logging.info("Data loaded successfully")
    #         return (
    #             df_componenet,
    #             df_DeliveryandCustomerSupport,
    #             df_Designand_Aesthetics,
    #             df_Dimensions,
    #             df_Features,
    #             df_Functionality,
    #             df_Installation,
    #             df_Material,
    #             df_Price,
    #             df_Quality,
    #             df_Usability,
    #             df_Polarity,
    #         )
    #     except Exception as e:
    #         logging.error("Error while loading data")
    #         logging.error(e)


if __name__ == "__main__":
    data_utils = DataUtils()
    train_data, test_data = data_utils.read_data()
    print(train_data.head())
    print(train_data.shape)
    print(test_data.head())
    print(test_data.shape)
    # (
    #     df_componenet,
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
