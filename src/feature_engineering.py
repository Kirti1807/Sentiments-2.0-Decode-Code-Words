import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import re
import logging
from nltk import sent_tokenize, word_tokenize
import gensim
from data_ingestion import DataUtils
from data_processing import DataProcessing
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib


class FeatureEngineering:
    def __init__(self, train_data, test_data) -> None:
        self.train_data = train_data
        self.test_data = test_data

    def get_count_of_words(self):
        # get count of words in train and test data
        logging.info("Get count of words in train and test data")
        train_data_count = self.train_data["Review"].apply(
            lambda x: len(x.split()))
        test_data_count = self.test_data["Review"].apply(
            lambda x: len(x.split()))
        return train_data_count, test_data_count

    def get_count_of_sentences(self):
        # get count of sentences in train and test data
        logging.info("Get count of sentences in train and test data")
        train_data_sentences = self.train_data["Review"].apply(
            lambda x: len(sent_tokenize(x))
        )
        test_data_sentences = self.test_data["Review"].apply(
            lambda x: len(sent_tokenize(x))
        )
        return train_data_sentences, test_data_sentences

    def get_average_word_length(self):
        # get average word length in train and test data
        logging.info("Get average word length in train and test data")
        train_data_average_word_length = self.train_data["Review"].apply(
            lambda x: np.mean([len(word) for word in x.split()])
        )
        test_data_average_word_length = self.test_data["Review"].apply(
            lambda x: np.mean([len(word) for word in x.split()])
        )
        return train_data_average_word_length, test_data_average_word_length

    def get_average_sentence_length(self):
        # get average sentence length in train and test data
        logging.info("Get average sentence length in train and test data")
        train_data_average_sentence_length = self.train_data["Review"].apply(
            lambda x: np.mean([len(sentence) for sentence in sent_tokenize(x)])
        )
        test_data_average_sentence_length = self.test_data["Review"].apply(
            lambda x: np.mean([len(sentence) for sentence in sent_tokenize(x)])
        )
        return train_data_average_sentence_length, test_data_average_sentence_length

    def get_average_sentence_complexity(self):
        # get average sentence complexity in train and test data
        logging.info("Get average sentence complexity in train and test data")
        train_data_average_sentence_complexity = self.train_data["Review"].apply(
            lambda x: np.mean(
                [len(word_tokenize(sentence)) for sentence in sent_tokenize(x)]
            )
        )
        test_data_average_sentence_complexity = self.test_data["Review"].apply(
            lambda x: np.mean(
                [len(word_tokenize(sentence)) for sentence in sent_tokenize(x)]
            )
        )
        return (
            train_data_average_sentence_complexity,
            test_data_average_sentence_complexity,
        )

    def get_average_word_complexity(self):
        # get average word complexity in train and test data
        logging.info("Get average word complexity in train and test data")
        train_data_average_word_complexity = self.train_data["Review"].apply(
            lambda x: np.mean([len(word_tokenize(word)) for word in x.split()])
        )
        test_data_average_word_complexity = self.test_data["Review"].apply(
            lambda x: np.mean([len(word_tokenize(word)) for word in x.split()])
        )
        return train_data_average_word_complexity, test_data_average_word_complexity

    def train_a_gensim_model(self):
        # train a gensim model
        logging.info("Train a gensim model")

        review_text = self.train_data.Review.apply(
            gensim.utils.simple_preprocess)
        model = gensim.models.Word2Vec(window=10, min_count=2, workers=4)
        model.build_vocab(review_text, progress_per=1000)
        model.train(review_text, total_examples=model.corpus_count,
                    epochs=model.epochs)
        model.save(r"E:\Hackathon\UGAM\src\saved_model\ugam_reviews.model")
        return model

    def get_word_embeddings(self, model):
        # get word embeddings
        logging.info("Get word embeddings")
        word_embeddings = model.wv
        return word_embeddings

    def get_similar(self, word, model):
        if word in model.wv:
            return model.wv.most_similar(word)[0]  # try runnign again
        else:
            return None

    def make_acolumn(self, model):
        # make a new column "most similar words" and get the most similar words for every word in review text
        logging.info(
            "Make a new column 'most similar words' and get the most similar words for every word in review text and leave the word whic is not present in the model"
        )
        #
        self.train_data["most_similar_words"] = self.train_data["Review"].apply(
            lambda x: [
                self.get_similar(word, model) for word in word_tokenize(x)
            ]  # get the most similar words for every word in review text
        )
        self.test_data["most_similar_words"] = self.test_data["Review"].apply(
            lambda x: [
                self.get_similar(word, model) for word in word_tokenize(x)
            ]  # get the most similar words for every word in review text
        )
        return self.train_data, self.test_data

    def process_most_similar_words(self, text):

        # process most similar words
        logging.info("Process most similar words")
        # process the column most similar words row by row
        # tokenize the word
        text = word_tokenize(text)
        for j in text:
            if j.isalpha() == False:
                text.remove(j)
            if j == "None":
                text.remove(j)
            if j == "":
                text.remove(j)
                # convert str to int
            if j.isdigit():
                text.remove(j)
        text = " ".join(text)
        # remove punctuation
        text = re.sub(r"[^\w\s]", "", text)
        # remove numbers, None and empty strings
        text = re.sub(r"\d+", "", text)
        # remove None from text
        text = re.sub(r"None", "", text)
        # remove extra spaces
        text = re.sub(r"\s+", " ", text)
        # remove stop words
        from nltk.corpus import stopwords

        stop_words = set(stopwords.words("english"))
        text = [word for word in text.split() if word not in stop_words]
        # convert list to str
        text = " ".join(text)
        return text

    def add_features(self):
        logging.info("Add features")
        train_data_count, test_data_count = self.get_count_of_words()
        train_data_sentences, test_data_sentences = self.get_count_of_sentences()
        (
            train_data_average_word_length,
            test_data_average_word_length,
        ) = self.get_average_word_length()
        (
            train_data_average_sentence_length,
            test_data_average_sentence_length,
        ) = self.get_average_sentence_length()
        (
            train_data_average_sentence_complexity,
            test_data_average_sentence_complexity,
        ) = self.get_average_sentence_complexity()
        (
            train_data_average_word_complexity,
            test_data_average_word_complexity,
        ) = self.get_average_word_complexity()

        self.train_data["count_of_words"] = train_data_count
        self.test_data["count_of_words"] = test_data_count
        self.train_data["count_of_sentences"] = train_data_sentences
        self.test_data["count_of_sentences"] = test_data_sentences
        self.train_data["average_word_length"] = train_data_average_word_length
        self.test_data["average_word_length"] = test_data_average_word_length
        self.train_data["average_sentence_length"] = train_data_average_sentence_length
        self.test_data["average_sentence_length"] = test_data_average_sentence_length
        self.train_data[
            "average_sentence_complexity"
        ] = train_data_average_sentence_complexity
        self.test_data[
            "average_sentence_complexity"
        ] = test_data_average_sentence_complexity
        self.train_data["average_word_complexity"] = train_data_average_word_complexity
        self.test_data["average_word_complexity"] = test_data_average_word_complexity

        return self.train_data, self.test_data


class Vectorization:
    def __init__(self, train_data, test_data) -> None:
        self.train_data = train_data
        self.test_data = test_data

    def extract_features(self):
        vectorizer = TfidfVectorizer()
        self.train_data["Review"].head()

        extracted_data = list(
            vectorizer.fit_transform(self.train_data["Review"]).toarray()
        )
        extracted_data = pd.DataFrame(extracted_data)
        extracted_data.head()
        extracted_data.columns = vectorizer.get_feature_names()

        vocab = vectorizer.vocabulary_
        mapping = vectorizer.get_feature_names()
        keys = list(vocab.keys())

        extracted_data.shape
        Modified_df = extracted_data.copy()
        print(Modified_df.shape)
        Modified_df.head()
        Modified_df.reset_index(drop=True, inplace=True)
        self.train_data.reset_index(drop=True, inplace=True)

        Final_Training_data = pd.concat([self.train_data, Modified_df], axis=1)

        Final_Training_data.head()
        print(Final_Training_data.shape)
        Final_Training_data.drop(["Review"], axis=1, inplace=True)
        Final_Training_data.head()
        Final_Training_data.to_csv(
            "Final_Training_vectorized.csv", index=False)

        dff_test = list(vectorizer.transform(
            self.test_data["Review"]).toarray())
        vocab_test = vectorizer.vocabulary_
        keys_test = list(vocab_test.keys())
        dff_test_df = pd.DataFrame(dff_test, columns=keys_test)
        dff_test_df.reset_index(drop=True, inplace=True)
        self.test_data.reset_index(drop=True, inplace=True)
        Final_Test = pd.concat([self.test_data, dff_test_df], axis=1)
        Final_Test.drop(["Review"], axis=1, inplace=True)
        Final_Test.to_csv("Final_Test_vectorized.csv", index=False)

        # save the vectorizer to disk
        joblib.dump(vectorizer, "vectorizer.pkl")
        return Final_Training_data, Final_Test

    def extract_features_most_similar_words(self):
        vectorizer = TfidfVectorizer()
        self.train_data["most_similar_words"].head()

        extracted_data = list(
            vectorizer.fit_transform(
                self.train_data["most_similar_words"]).toarray()
        )
        extracted_data = pd.DataFrame(extracted_data)
        extracted_data.head()
        extracted_data.columns = vectorizer.get_feature_names()

        vocab = vectorizer.vocabulary_
        mapping = vectorizer.get_feature_names()
        keys = list(vocab.keys())

        extracted_data.shape
        Modified_df = extracted_data.copy()
        print(Modified_df.shape)
        Modified_df.head()
        Modified_df.reset_index(drop=True, inplace=True)
        self.train_data.reset_index(drop=True, inplace=True)

        Final_Training_data = pd.concat([self.train_data, Modified_df], axis=1)

        Final_Training_data.head()
        print(Final_Training_data.shape)
        Final_Training_data.drop(["most_similar_words"], axis=1, inplace=True)
        Final_Training_data.head()
        Final_Training_data.to_csv(
            "Final_Training_vectorized_SimilarFeatures.csv", index=False)

        dff_test = list(vectorizer.transform(
            self.test_data["most_similar_words"]).toarray())
        vocab_test = vectorizer.vocabulary_
        keys_test = list(vocab_test.keys())
        dff_test_df = pd.DataFrame(dff_test, columns=keys_test)
        dff_test_df.reset_index(drop=True, inplace=True)
        self.test_data.reset_index(drop=True, inplace=True)
        Final_Test = pd.concat([self.test_data, dff_test_df], axis=1)
        Final_Test.drop(["most_similar_words"], axis=1, inplace=True)
        Final_Test.to_csv(
            "Final_Test_vectorized_SimilarFeatures.csv", index=False)

        # save the vectorizer to disk
        joblib.dump(vectorizer, "vectorizer_similarFeatures.pkl")
        return Final_Training_data, Final_Test


if __name__ == "__main__":
    data_utils = DataUtils()
    train_data, test_data = data_utils.read_data(
        train_path=r"E:\Hackathon\UGAM\Participants_Data_DCW\processed_data\train_data_with_most_similar_words_processed.csv",
        test_path=r"E:\Hackathon\UGAM\Participants_Data_DCW\processed_data\test_data_with_most_similar_words_processed.csv",
    )

    feature_engineering_obj = FeatureEngineering(train_data, test_data)
    # model = feature_engineering_obj.train_a_gensim_model()
    # model.wv.most_similar("Key")

    # train_data, test_data = feature_engineering_obj.make_acolumn(model)
    # test_data.to_csv(
    #     r"E:\Hackathon\UGAM\Participants_Data_DCW\processed_data\test_data_with_most_similar_words.csv",
    #     index=False,
    # )
    # train_data.to_csv(
    #     r"E:\Hackathon\UGAM\Participants_Data_DCW\processed_data\train_data_with_most_similar_words.csv",
    #     index=False,
    # )

    # train_data["most_similar_words"] = train_data["most_similar_words"].apply(
    #     lambda x: feature_engineering_obj.process_most_similar_words(x)
    # )
    # test_data["most_similar_words"] = test_data["most_similar_words"].apply(
    #     lambda x: feature_engineering_obj.process_most_similar_words(x)
    # )
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
