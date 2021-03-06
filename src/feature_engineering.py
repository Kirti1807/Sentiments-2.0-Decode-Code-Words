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
from gensim.models.fasttext import FastText
from application_logger import CustomApplicationLogger
from sklearn.decomposition import TruncatedSVD

class FeatureEngineering:
    def __init__(self, data) -> None:
        self.train_data = data
        self.file_object = open(
            r"D:\ML_Projects\MultiClassClassification\Sentiments-2.0-Decode-Code-Words\logs\FeatureEngineeringLogs.txt",
            "a+",
        )
        self.logging = CustomApplicationLogger()
        
    def get_count_of_words(self):
        self.logging.log(
            self.file_object,
            "In get_count_of_words method in FeatureEngineering class: Get count of words in data"
        )
        train_data_count = self.train_data["Review"].apply(
            lambda x: len(x.split()))
        return train_data_count

    def get_count_of_sentences(self):
        self.logging.log(
            self.file_object,
            "In get_count_of_sentences method in FeatureEngineering class: Get count of sentences in data"
        )
        train_data_sentences = self.train_data["Review"].apply(
            lambda x: len(sent_tokenize(x))
        )
        return train_data_sentences

    def get_average_word_length(self):
        self.logging.log(
            self.file_object,
            "In get_average_word_length method in FeatureEngineering class: Get average word length in data"
        )
        train_data_average_word_length = self.train_data["Review"].apply(
            lambda x: np.mean([len(word) for word in x.split()])
        )
        return train_data_average_word_length

    def get_average_sentence_length(self):
        self.logging.log(
            self.file_object,
            "In get_average_sentence_length method in FeatureEngineering class: Get average sentence length in data"
        )
        train_data_average_sentence_length = self.train_data["Review"].apply(
            lambda x: np.mean([len(sentence) for sentence in sent_tokenize(x)])
        )
        return train_data_average_sentence_length

    def get_average_sentence_complexity(self):
        self.logging.log(
            self.file_object,
            "In get_average_sentence_complexity method in FeatureEngineering class: Get average sentence complaxity in data"
        )
        train_data_average_sentence_complexity = self.train_data["Review"].apply(
            lambda x: np.mean(
                [len(word_tokenize(sentence)) for sentence in sent_tokenize(x)]
            )
        )
        return train_data_average_sentence_complexity

    def get_average_word_complexity(self):
        self.logging.log(
            self.file_object,
            "In get_average_word_complexity method in FeatureEngineering class: Get average word complaxity in data"
        )
        train_data_average_word_complexity = self.train_data["Review"].apply(
            lambda x: np.mean([len(word_tokenize(word)) for word in x.split()])
        )
        return train_data_average_word_complexity

    def train_a_gensim_model(self):
        self.logging.log(
            self.file_object,
            "In train_a_gensim_model method in FeatureEngineering class: train a gensim model"
        )
        review_text = self.train_data.Review.apply(
            gensim.utils.simple_preprocess)
        model = gensim.models.Word2Vec(window=10, min_count=2, workers=4)
        model.build_vocab(review_text, progress_per=1000)
        model.train(review_text, total_examples=model.corpus_count,
                    epochs=model.epochs)
        # model.save(r"E:\Hackathon\UGAM\src\saved_model\ugam_reviews.model")
        # model.save(r"ugam_reviews.model")

        return model

    def get_word_embeddings(self, model):
        # get word embeddings
        self.logging.log(
            self.file_object,
            "In get_word_embeddings method in FeatureEngineering class: getting word embeddings"
        )
        word_embeddings = model.wv
        return word_embeddings

    def get_similar(self, word, model):
        if word in model.wv:
            return model.wv.most_similar(word)[0][0]  # try runnign again
        else:
            return None

    def make_acolumn(self, model):
        # make a new column "most similar words" and get the most similar words for every word in review text
        self.logging.log(
            self.file_object,
            "In make_acolumn method in FeatureEngineering class: making new column for similar words"
        )
        self.train_data["most_similar_words"] = self.train_data["Review"].apply(
            lambda x: [
                self.get_similar(word, model) for word in word_tokenize(x)
            ] 
        )
        return self.train_data

    def process_most_similar_words(self, text):

        # self.logging.log(
        #     self.file_object,
        #     "In process_most_similar_words method in FeatureEngineering class: processing most similar word"
        # )
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
        self.logging.log(
            self.file_object,
            "In add_features method in FeatureEngineering class: started adding feature in dataset"
        )
        try:
            train_data_count = self.get_count_of_words()
            train_data_sentences = self.get_count_of_sentences()
            train_data_average_word_length = self.get_average_word_length()
            train_data_average_sentence_length = self.get_average_sentence_length()
            train_data_average_sentence_complexity = self.get_average_sentence_complexity()
            train_data_average_word_complexity = self.get_average_word_complexity()

            self.train_data["count_of_words"] = train_data_count
            self.train_data["count_of_sentences"] = train_data_sentences
            self.train_data["average_word_length"] = train_data_average_word_length
            self.train_data["average_sentence_length"] = train_data_average_sentence_length
            self.train_data["average_sentence_complexity"] = train_data_average_sentence_complexity
            self.train_data["average_word_complexity"] = train_data_average_word_complexity

            self.logging.log(
                self.file_object,
                "In add_features method in FeatureEngineering class: features added successfully"
            )

            return self.train_data

        except Exception as e:
            self.logging.log(
                self.file_object,
                f"In add_features method in FeatureEngineering class: Error in adding features: {e}"
            )
            raise e


class Vectorization:
    def __init__(self, data) -> None:
        self.train_data = data
        self.file_object = open(
            r"D:\ML_Projects\MultiClassClassification\Sentiments-2.0-Decode-Code-Words\logs\VectorizationLogs.txt",
            "a+",
        )
        self.logging = CustomApplicationLogger()

    # def tfidf_extract_features(self):
    #     vectorizer = TfidfVectorizer()
    #     self.train_data["Review"].head()

    #     extracted_data = list(
    #         vectorizer.fit_transform(self.train_data["Review"]).toarray()
    #     )
    #     extracted_data = pd.DataFrame(extracted_data)
    #     extracted_data.head()
    #     extracted_data.columns = vectorizer.get_feature_names()

    #     vocab = vectorizer.vocabulary_
    #     mapping = vectorizer.get_feature_names()
    #     keys = list(vocab.keys())

    #     extracted_data.shape
    #     Modified_df = extracted_data.copy()
    #     print(Modified_df.shape)
    #     Modified_df.head()
    #     Modified_df.reset_index(drop=True, inplace=True)
    #     self.train_data.reset_index(drop=True, inplace=True)

    #     Final_Training_data = pd.concat([self.train_data, Modified_df], axis=1)

    #     Final_Training_data.head()
    #     print(Final_Training_data.shape)
    #     Final_Training_data.drop(["Review"], axis=1, inplace=True)
    #     Final_Training_data.head()
    #     Final_Training_data.to_csv(
    #         "Final_Training_vectorized.csv", index=False)

    #     dff_test = list(vectorizer.transform(
    #         self.test_data["Review"]).toarray())
    #     vocab_test = vectorizer.vocabulary_
    #     keys_test = list(vocab_test.keys())
    #     dff_test_df = pd.DataFrame(dff_test, columns=keys_test)
    #     dff_test_df.reset_index(drop=True, inplace=True)
    #     self.test_data.reset_index(drop=True, inplace=True)
    #     Final_Test = pd.concat([self.test_data, dff_test_df], axis=1)
    #     Final_Test.drop(["Review"], axis=1, inplace=True)
    #     Final_Test.to_csv("Final_Test_vectorized.csv", index=False)

    #     # save the vectorizer to disk
    #     joblib.dump(vectorizer, "vectorizer.pkl")
    #     return Final_Training_data, Final_Test

    def fast_text_extract_features(self):
        self.logging.log(
            self.file_object,
            "In fast_text_extract_features method In Vectorization class: adding fast-text features"
        )
        try:
            def averaged_word2vec_vectorizer(corpus, model, num_features):
                vocabulary = set(model.wv.index_to_key)

                def average_word_vectors(words, model, vocabulary, num_features):
                    feature_vector = np.zeros((num_features,), dtype="float64")
                    nwords = 0.

                    for word in words:
                        if word in vocabulary:
                            nwords = nwords + 1.
                            feature_vector = np.add(feature_vector, model.wv[word])
                    if nwords:
                        feature_vector = np.divide(feature_vector, nwords)

                    return feature_vector
                features = [average_word_vectors(tokenized_sentence, model, vocabulary, num_features)
                            for tokenized_sentence in corpus]
                return np.array(features)

            # ft_model = FastText.load("ft_model")

            tokenized_docs_train = [doc.split()
                                    for doc in list(self.train_data['Review'])]
            ft_model = FastText(tokenized_docs_train, min_count=2,
                                vector_size=300, workers=4, window=40, sg=1, epochs=100)
            doc_vecs_ft_train = averaged_word2vec_vectorizer(
                tokenized_docs_train, ft_model, 300)
            doc_vecs_ft_train = pd.DataFrame(doc_vecs_ft_train)

            # tokenized_docs_test = [doc.split()
            #                     for doc in list(self.test_data['Review'])]
            # doc_vecs_ft_test = averaged_word2vec_vectorizer(
            #     tokenized_docs_test, ft_model, 300)
            # doc_vecs_ft_test = pd.DataFrame(doc_vecs_ft_test)
            ft_model.save("ft_model.model")

            self.logging.log(
                self.file_object,
                "In fast_text_extract_features method In Vectorization class: successfully added fast-text features"
            )
            return doc_vecs_ft_train

        except Exception as e:
            self.logging.log(
                self.file_object,
                f"In fast_text_extract_features method In Vectorization class: Error in adding fast-text features: {e}"
            )
            raise e

    def extract_features_most_similar_words(self):
        self.logging.log(
            self.file_object,
            "In extract_features_most_similar_words method In Vectorization class: starting adding most similar word features"
        )
        try:
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
            #Final_Training_data.head()
            Final_Training_data.to_csv(
                "D:\ML_Projects\MultiClassClassification\Sentiments-2.0-Decode-Code-Words\data\Final_Training_vectorized_SimilarFeatures.csv", index=False)

            # dff_test = list(vectorizer.transform(
            #     self.test_data["most_similar_words"]).toarray())
            # vocab_test = vectorizer.vocabulary_
            # keys_test = list(vocab_test.keys())
            # dff_test_df = pd.DataFrame(dff_test, columns=keys_test)
            # dff_test_df.reset_index(drop=True, inplace=True)
            # self.test_data.reset_index(drop=True, inplace=True)
            # Final_Test = pd.concat([self.test_data, dff_test_df], axis=1)
            # Final_Test.drop(["most_similar_words"], axis=1, inplace=True)
            # Final_Test.to_csv(
            #     "Final_Test_vectorized_SimilarFeatures.csv", index=False)

            # save the vectorizer to disk
            joblib.dump(vectorizer, "vectorizer_similarFeatures.pkl")
            return Final_Training_data

        except Exception as e:
            self.logging.log(
                self.file_object,
                f"In extract_features_most_similar_words method In Vectorization class: Error in adding similar word features: {e}"  
            )
            raise e

    def reduce_features(self, train_data):
        self.logging.log(
            self.file_object,
            "In reduce_features method in Vectorization class: started reducing features"
        )
        try:
            filename = 'svd.sav'
            #svd = joblib.load(filename)
            svd = TruncatedSVD(n_components=20, n_iter=7, random_state=42)
            tuncated_train_data = svd.fit_transform(train_data)
            tuncated_train_data=pd.DataFrame(tuncated_train_data)
            #joblib.dump(svd, filename)
            return tuncated_train_data
            
        except Exception as e:
            self.logging.log(
                self.file_object,
                f"In reduce_features method in Vectorization class: Error in feature reduction: {e}"
            )
            raise e

if __name__ == "__main__":
    pass
    data_utils = DataUtils()
    train_data, test_data = data_utils.read_data(
        train_path=r"D:\ML_Projects\MultiClassClassification\Sentiments-2.0-Decode-Code-Words\data\trainmulticlass.csv",
        test_path=r"D:\ML_Projects\MultiClassClassification\Sentiments-2.0-Decode-Code-Words\data\testmulticlass.csv"
    )

    fe = FeatureEngineering(train_data)
    process_train_data = fe.add_features()
    print(process_train_data.shape)
    print(process_train_data.head())

    # feature_engineering_obj = FeatureEngineering(train_data, test_data)
    # feature_engineering_obj = FeatureEngineering(dummy_train, dummy_test)

    # model = feature_engineering_obj.train_a_gensim_model()
    # model.wv.most_similar("good")

    # train_data, test_data = feature_engineering_obj.make_acolumn(model)
    # # test_data.to_csv(
    # #     r"E:\Hackathon\UGAM\Participants_Data_DCW\processed_data\test_data_with_most_similar_words.csv",
    # #     index=False,
    # # )
    # # train_data.to_csv(
    # #     r"E:\Hackathon\UGAM\Participants_Data_DCW\processed_data\train_data_with_most_similar_words.csv",
    # #     index=False,
    # # )
    # train_data["most_similar_words"]
    # train_data["most_similar_words"]=train_data["most_similar_words"].apply(str)
    # test_data["most_similar_words"]=test_data["most_similar_words"].apply(str)

    # train_data["most_similar_words"] = train_data["most_similar_words"].apply(
    #     lambda x: feature_engineering_obj.process_most_similar_words(x)
    # )
    # test_data["most_similar_words"] = test_data["most_similar_words"].apply(
    #     lambda x: feature_engineering_obj.process_most_similar_words(x)
    # )
    # # train_data.to_csv(
    # #     r"E:\Hackathon\UGAM\Participants_Data_DCW\processed_data\train_data_with_most_similar_words_processed.csv",
    # #     index=False,
    # # )
    # # test_data.to_csv(
    # #     r"E:\Hackathon\UGAM\Participants_Data_DCW\processed_data\test_data_with_most_similar_words_processed.csv",
    # #     index=False,
    # # )

    # train_data, test_data = feature_engineering_obj.add_features()
    # train_data.to_csv(
    #     r"E:\Hackathon\UGAM\Participants_Data_DCW\processed_data\train_data_with_most_similar_words_processed_with_features.csv",
    #     index=False,
    # )
    # test_data.to_csv(
    #     r"E:\Hackathon\UGAM\Participants_Data_DCW\processed_data\test_data_with_most_similar_words_processed_with_features.csv",
    #     index=False,
    # ).
