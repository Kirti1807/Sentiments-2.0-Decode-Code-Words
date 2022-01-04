import numpy as np
import pandas as pd
import xgboost as xgb
import logging
import optuna

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from catboost import CatBoostClassifier
import catboost
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import StackingClassifier
import xgboost as xgb
from xgboost import XGBClassifier
# import lightgbm as lgb

class MLHyperparameterOpt:
    def __init__(self, x_train, x_test, y_train, y_test) -> None:
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test

    def optimize_decisiontrees(self, trial):
        criterion = trial.suggest_categorical("criterion", ("gini", "entropy"))
        max_depth = trial.suggest_int("max_depth", 1, 20)
        min_samples_split = trial.suggest_int("min_samples_split", 2, 20)
        clf = DecisionTreeClassifier(
            criterion=criterion,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
        )
        clf.fit(self.x_train, self.y_train)
        val_accuracy = clf.score(self.x_test, self.y_test)
        return val_accuracy

    def optimize_randomforest(self, trial):
        logging.info("optimize_randomforest")
        n_estimators = trial.suggest_int("n_estimators", 1, 200)
        max_depth = trial.suggest_int("max_depth", 1, 20)
        min_samples_split = trial.suggest_int("min_samples_split", 2, 20)
        clf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
        )
        clf.fit(self.x_train, self.y_train)
        val_accuracy = clf.score(self.x_test, self.y_test)
        return val_accuracy

    def optimize_adaboost(self, trial):
        logging.info("optimize_adaboost")
        n_estimators = trial.suggest_int("n_estimators", 1, 200)
        learning_rate = trial.suggest_loguniform("learning_rate", 1e-7, 10.0)
        clf = AdaBoostClassifier(n_estimators=n_estimators, learning_rate=learning_rate)
        clf.fit(self.x_train, self.y_train)
        val_accuracy = clf.score(self.x_test, self.y_test)
        return val_accuracy

    def optimize_gradientboosting(self, trial):
        logging.info("optimize_gradientboosting")
        n_estimators = trial.suggest_int("n_estimators", 1, 200)
        learning_rate = trial.suggest_loguniform("learning_rate", 1e-7, 10.0)
        max_depth = trial.suggest_int("max_depth", 1, 20)
        min_samples_split = trial.suggest_int("min_samples_split", 2, 20)
        clf = GradientBoostingClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
        )
        clf.fit(self.x_train, self.y_train)
        val_accuracy = clf.score(self.x_test, self.y_test)
        return val_accuracy

    def optimize_xgboost(self, trial):
        logging.info("optimize_xgboost")
        param = {
            "max_depth": trial.suggest_int("max_depth", 1, 30),
            "learning_rate": trial.suggest_loguniform("learning_rate", 1e-7, 10.0),
            "n_estimators": trial.suggest_int("n_estimators", 1, 200),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
            "subsample": trial.suggest_uniform("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_uniform("colsample_bytree", 0.5, 1.0),
            "colsample_bylevel": trial.suggest_uniform("colsample_bylevel", 0.5, 1.0),
            "reg_alpha": trial.suggest_loguniform("reg_alpha", 1e-7, 10.0),
            "reg_lambda": trial.suggest_loguniform("reg_lambda", 1e-7, 10.0),
            "scale_pos_weight": trial.suggest_loguniform(
                "scale_pos_weight", 1e-7, 10.0
            ),
        }
        clf = xgb.XGBClassifier(**param)
        clf.fit(self.x_train, self.y_train)
        val_accuracy = clf.score(self.x_test, self.y_test)
        return val_accuracy

    def optimize_catboost(self, trial):
        logging.info("optimize_catboost")
        train_data = catboost.Pool(self.x_train, label=self.y_train)
        test_data = catboost.Pool(self.x_test, label=self.y_test)

        param = {
            "loss_function": trial.suggest_categorical(
                "loss_function", ("Logloss", "CrossEntropy")
            ),
            "l2_leaf_reg": trial.suggest_loguniform("l2_leaf_reg", 1e-3, 10.0),
            "max_bin": trial.suggest_int("max_bin", 200, 400),
            "subsample": trial.suggest_uniform("bagging_fraction", 0.4, 1.0),
            "learning_rate": trial.suggest_uniform("learning_rate", 0.006, 0.018),
            "n_estimators": trial.suggest_int("n_estimators", 1, 2000),
            "max_depth": trial.suggest_categorical("max_depth", [7, 10, 14, 16]),
            "random_state": trial.suggest_categorical("random_state", [24, 48, 2020]),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 300),
        }

        clf = catboost.CatBoostClassifier(**param)
        clf.fit(self.x_train, self.y_train)
        val_accuracy = clf.score(self.x_test, self.y_test)
        return val_accuracy


class TrainMLModel:
    def __init__(self, x_train, x_test, y_train, y_test) -> None:
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test

    def decision_trees(self, fine_tuning=True):
        logging.info("Entered for training Decision Trees model")
        if fine_tuning:
            hyper_opt = MLHyperparameterOpt(
                    self.x_train, self.y_train, self.x_test, self.y_test
                )
            study = optuna.create_study(direction="maximize")
            study.optimize(hyper_opt.optimize_decisiontrees, n_trials=100)
            trial = study.best_trial
            criterion = trial.params["criterion"]
            max_depth = trial.params["max_depth"]
            min_samples_split = trial.params["min_samples_split"]
            print("Best parameters : ", trial.params)
            clf = DecisionTreeClassifier(
                    criterion=criterion,
                    max_depth=max_depth,
                    min_samples_split=min_samples_split,
            )
            clf.fit(self.x_train, self.y_train)
            return clf
        else:
            model = DecisionTreeClassifier(
                    criterion="entropy", max_depth=1, min_samples_split=7
            )

            model.fit(self.x_train, self.y_train)
            return model
        # except Exception as e:
        #     logging.error("Error in training Decision Trees model")
        #     logging.error(e)
        #     return None

    def random_forest(self, fine_tuning=True):
        logging.info("Entered for training Random Forest model")
        try:
            if fine_tuning:
                hyper_opt = MLHyperparameterOpt(
                    self.x_train, self.y_train, self.x_test, self.y_test
                )
                study = optuna.create_study(direction="maximize")
                study.optimize(hyper_opt.optimize_randomforest, n_trials=100)
                trial = study.best_trial
                n_estimators = trial.params["n_estimators"]
                max_depth = trial.params["max_depth"]
                min_samples_split = trial.params["min_samples_split"]
                print("Best parameters : ", trial.params)
                clf = RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    min_samples_split=min_samples_split,
                )
                clf.fit(self.x_train, self.y_train)
                return clf
            else:
                model = RandomForestClassifier(
                    n_estimators=92, max_depth=19, min_samples_split=4
                )
                model.fit(self.x_train, self.y_train)
                return model
        except Exception as e:
            logging.error("Error in training Random Forest model")
            logging.error(e)
            return None

    def adaboost(self, fine_tuning=True):
        logging.info("Entered for training AdaBoost model")
        try:
            if fine_tuning:
                hyper_opt = MLHyperparameterOpt(
                    self.x_train, self.y_train, self.x_test, self.y_test
                )
                study = optuna.create_study(direction="maximize")
                study.optimize(hyper_opt.optimize_adaboost, n_trials=100)
                trial = study.best_trial
                n_estimators = trial.params["n_estimators"]
                learning_rate = trial.params["learning_rate"]
                print("Best parameters : ", trial.params)
                clf = AdaBoostClassifier(
                    n_estimators=n_estimators, learning_rate=learning_rate
                )
                clf.fit(self.x_train, self.y_train)
                return clf
            else:
                model = AdaBoostClassifier(
                    n_estimators=143, learning_rate=0.2374269674908056
                )
                model.fit(self.x_train, self.y_train)
                return model
        except Exception as e:
            logging.error("Error in training AdaBoost model")
            logging.error(e)
            return None

    def gradient_boosting(self, fine_tuning=True):
        logging.info("Entered for training Gradient Boosting model")
        try:
            if fine_tuning:
                hyper_opt = MLHyperparameterOpt(
                    self.x_train, self.y_train, self.x_test, self.y_test
                )
                study = optuna.create_study(direction="maximize")
                study.optimize(hyper_opt.optimize_gradientboosting, n_trials=100)
                trial = study.best_trial
                n_estimators = trial.params["n_estimators"]
                learning_rate = trial.params["learning_rate"]
                clf = GradientBoostingClassifier(
                    n_estimators=n_estimators, learning_rate=learning_rate
                )
                clf.fit(self.x_train, self.y_train)
                return clf
        except Exception as e:
            logging.error("Error in training Gradient Boosting model")
            logging.error(e)
            return None
    def xgboost(self, fine_tuning=True):
        logging.info("Entered for training XGBoost model")
        try:
            if fine_tuning:
                hy_opt = MLHyperparameterOpt(
                    self.x_train, self.y_train, self.x_test, self.y_test
                )
                study = optuna.create_study(direction="maximize")
                study.optimize(hy_opt.optimize_xgboost, n_trials=100)
                trial = study.best_trial
                n_estimators = trial.params["n_estimators"]
                learning_rate = trial.params["learning_rate"]
                max_depth = trial.params["max_depth"]
                min_child_weight = trial.params["min_child_weight"]
                subsample = trial.params["subsample"]
                colsample_bytree = trial.params["colsample_bytree"]
                reg_alpha = trial.params["reg_alpha"]
                reg_lambda = trial.params["reg_lambda"]
                clf = XGBClassifier(
                    n_estimators=n_estimators,
                    learning_rate=learning_rate,
                    max_depth=max_depth,
                    min_child_weight=min_child_weight,
                    subsample=subsample,
                    colsample_bytree=colsample_bytree,
                    reg_alpha=reg_alpha,
                    reg_lambda=reg_lambda,
                )
                print("Best parameters : ", trial.params)
                clf.fit(self.x_train, self.y_train)
                return clf
            else:
                params = {
                    "objective": "binary:logistic",
                    "use_label_encoder": True,
                    "base_score": 0.5,
                    "booster": "gbtree",
                    "colsample_bylevel": 1,
                    "colsample_bynode": 1,
                    "colsample_bytree": 0.9865465799558366,
                    "enable_categorical": False,
                    "gamma": 0,
                    "gpu_id": -1,
                    "importance_type": None,
                    "interaction_constraints": "",
                    "learning_rate": 0.1733839701849005,
                    "max_delta_step": 0,
                    "max_depth": 6,
                    "min_child_weight": 1,
                    "n_estimators": 73,
                    "n_jobs": 8,
                    "num_parallel_tree": 1,
                    "predictor": "auto",
                    "random_state": 0,
                    "reg_alpha": 8.531151528439326e-06,
                    "reg_lambda": 0.006678010524298995,
                    "scale_pos_weight": 1,
                    "subsample": 0.7761340636250333,
                    "tree_method": "exact",
                    "validate_parameters": 1,
                    "verbosity": None,
                }
                clf = XGBClassifier(**params)
                clf.fit(self.x_train, self.y_train)
                return clf

        except Exception as e:
            logging.error("Error in training XGBoost model")
            logging.error(e)
            return None

    def catboost(self, fine_tuning=True, best_trial=None):
        logging.info("Entered for training CatBoost model")
        if fine_tuning:
            hy_opt = MLHyperparameterOpt(
                self.x_train, self.y_train, self.x_test, self.y_test
            )
            study = optuna.create_study(direction="maximize")
            study.optimize(hy_opt.optimize_catboost, n_trials=10)
            trial = study.best_trial

            max_depth = trial.params["max_depth"]
            l2_leaf_reg = trial.params["l2_leaf_reg"]
            max_bin = trial.params["max_bin"]
            bagging_fraction = trial.params["bagging_fraction"]
            learning_rate = trial.params["learning_rate"]
            loss_function = trial.params["loss_function"]
            n_estimators = trial.params["n_estimators"]
            random_state = trial.params["random_state"]
            min_data_in_leaf = trial.params["min_data_in_leaf"]

            clf = CatBoostClassifier(
                max_depth=max_depth,
                l2_leaf_reg=l2_leaf_reg,
                max_bin=max_bin,
                bagging_fraction=bagging_fraction,
                learning_rate=learning_rate,
                loss_function=loss_function,
                n_estimators=n_estimators,
                random_state=random_state,
                min_data_in_leaf=min_data_in_leaf,
            )

            clf.fit(self.x_train, self.y_train)
            return clf
        else:
            clf = CatBoostClassifier(
                loss_function="Logloss",
                l2_leaf_reg=0.005603859124543057,
                max_bin=332,
                learning_rate=0.0075037108414941255,
                n_estimators=1297,
                max_depth=16,
                random_state=24,
                min_data_in_leaf=94,
            )
            clf.fit(self.x_train, self.y_train)
            return clf

