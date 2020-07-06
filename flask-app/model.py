"""
Main model file.
"""

from itertools import chain, combinations
import pandas as pd 
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import joblib
import os

# disable warnings from pandas
import warnings
warnings.filterwarnings("ignore")

class Model:
    
    # number of repeats when evaluating the k-means algo
    NUM_REPEATS = 5

    def __init__(self):
        
        self.best_feats = None
        self.kmeans = None    
        self.scaler = StandardScaler()
    
    def train_and_evaluate(self, path, verbose=False):
        """
        Full Pipeline to train and evaluate on a csv file (like the one provided).
        Performs model selection by finding the best features to apply to a k-means
        algorithm with k=2, used for classifing whether an intrusion has occured.

        Args:
            path: File path to data directory.
            verbose: Print accuracy values on evaluation test during training
        """

        self.verbose = verbose

        # import the data 
        df = pd.read_csv(path) 
        # divide in val and test data  
        df_val, df_test = train_test_split(df, test_size=0.2)
        
        # do feature selection
        self._feature_selection(df_val) 

        # fit on best data
        df_val = self._feature_eng(df_val)
        self.fit_clusters(df_val, self.best_feats) 
        
        # evalaute on test data
        df_test = self._feature_eng(df_test)
        y_test = df_test["flag"]
        self.evaluate_predictions(df_test, y_test) 
        
        # re-run on whole dataset
        df = self._feature_eng(df)
        self.fit_clusters(df, self.best_feats)
        # save model
        self.save_model()


    def _feature_eng(self, df):
        """
        Add new features to the dataset given previous data-analysis conducted.

        Args:
            df: Dataframe on which to perform the feature engineering.
        """
        
        # as the RxKBTot and the RxKBTot are the most important features, play around with them
        df["R_per_T"] = df["RxKBTot"]/df["TxKBTot"]
        df["T_per_R"] = df["TxKBTot"]/df["RxKBTot"]

        df["R_per_CPU"] = df["RxKBTot"]/df["CPU"]
        df["T_per_CPU"] = df["TxKBTot"]/df["CPU"]
        
        # replace inf values created
        df["R_per_T"].replace(np.inf, 1e2, inplace=True)
        df["T_per_R"].replace(np.inf, 1e2, inplace=True)
        df["R_per_CPU"].replace(np.inf, 1e3, inplace=True)
        df["T_per_CPU"].replace(np.inf, 1e3, inplace=True)
        
        
        # replace nans with high values
        df = df.dropna()
    
        return df

    def _feature_selection(self, df):
        """
        Run an evaluation to find the most important features in the given dataset.
        
        Args:
            df: Dataframe on which to perform selection.
        """

        # do feature enng on the dataset 
        df = self._feature_eng(df)

        # get a list of features
        all_features = list(df.columns)
        # drop the target label 
        y = df["flag"]
        
        # drop the time (t) as it should not matter and the flag 
        del all_features[all_features.index("flag")]
        del all_features[all_features.index("t")]
        
        # delete these extra features because they don't seem very useful
        del all_features[all_features.index("Watts")]
        del all_features[all_features.index("Amps")]
        del all_features[all_features.index("RMS")]
        del all_features[all_features.index("diff_encoder_l")]
        
        # check importance at the start
        self.best_feats, best_score = self._score_all_features(df, y, all_features)
        # convert best features to a list 
        self.best_feats = list(self.best_feats)

    def _score_all_features(self, df, y, features):
        """
        Perform an evaluation on the k-means algorithm by selecting the best features all the features selected.
        Returns a tuple of the best features and the evaluation score for the respective features.

        Args:
            df: Dataframe on which to train.
            y: Labels to perform evaluation of each iteration.
            features: All the features from which to combine and create different subsets.
        """
         
        # get the powerset to check all feature combinatations (there are only 2**8=256 possible combinations without feat eng)
        power_set = list(chain.from_iterable(combinations(features, r) for r in range(len(features)+1)))
        self.tot_trials = len(power_set)

        best_score = 0
        best_feats = None 
        for self.i, feat_sel in enumerate(power_set):
            
            # if empty set skip
            if len(feat_sel) == 0:
                continue

            # repeat k-mans 5 times to reduce randomness
            scores = []
            for _ in range(self.NUM_REPEATS):
                # fit k-means to this data 
                self.fit_clusters(df, list(feat_sel))
                # evaluate the clusters 
                score = self.evaluate_clusters(y, feat_sel, self.verbose)
                scores.append(score)

            # compute mean score
            scores = np.array(scores)
            score = scores.mean()
            std_score = scores.std()

            # keep track of best score and feat only if std is smaller than 0.2, else dont count it. NOTE: this number is pretty aribitrary-can be improved upon
            if score > best_score and std_score < 0.2:
                best_score = score
                best_feats = feat_sel

        # print the best scores and feat 
        print("--------------------- Best Feat Selection -------------------------")
        print(f"Feature Selection: {best_feats}, Accuracy: {best_score}")
        
        return best_feats, best_score

    def fit_clusters(self, df, features):
        """
        Fit clustering to the selected dataset.

        Args:
            df: Dataset on which to perform the dataset.
            features: Features selected from the dataset on which to fit k-means.
        """
        # keep only the selected features
        df = df[features]

        # pre-process the data
        df = self.scaler.fit_transform(df)

        # k_means with k=2 to cluster the data 
        self.kmeans = KMeans(n_clusters=2).fit(df)
   
    def evaluate_clusters(self, y, features, verbose=False):
        """
        Evaluate the clusters using the target dataset flag.
        """
        # get the predicted labes 
        pred_clust = self.kmeans.labels_

        # print the accuracy
        score = (pred_clust  == y).sum()/len(y)
        
        if verbose:
            print(f"Trial: {self.i}/{self.tot_trials}, Feature Selection: {features}, Accuracy: {score}")
        
        return score

    def evaluate_predictions(self, df, y):
        """
        Make predictions on new data and evaluate using the corresponding target label
        the results.

        Args:
            df: Dataframe used to predict.
            y: Labels for the corresponding columns.
        """
       
        # predict on new incoming data
        pred_clust = self.predict(df)

        # print the accuracy
        score = (pred_clust  == y).sum()/len(y)
        
        print(f"Test Accuracy: {score}")


    def save_model(self, path="./SavedModel/"):
        """
        Save the trained clustering and pre-processing model.

        Args:
            path: Path directory to which the model will be saved.
        """
        
        # check if path exits, if not create it 
        directory = os.path.dirname(path)
        if not os.path.exists(directory):
            os.makedirs(directory)
        
        joblib.dump(self.best_feats, path + "best_feats.pkl") # save also the best features 
        joblib.dump(self.scaler, path + "scaler.pkl") # save also the preprocess
        joblib.dump(self.kmeans, path + "kmeans.pkl")

    def load_model(self, path="./SavedModel/"):
        """
        Load a pre-selected model.

        Args:
            path: Directory from which to save the model. 
        """
        
        self.best_feats = joblib.load(path + "best_feats.pkl") # load the best features
        self.scaler = joblib.load(path + "scaler.pkl") # load the preprocess
        self.kmeans = joblib.load(path + "kmeans.pkl")

    def _preprocess(self, df):
        """
        Preprocess the data before feeeding it to the model. 
        Feature engineering and selection followed bynormalisation
        is performed on a given dataset.

        Args:
            df: Dataframe consisting only of the feature to preprocess.
        """
        
        # perform feat eng 
        df = self._feature_eng(df)
        # select best features from feat selection
        df = df[self.best_feats]
        # normalise data 
        df = self.scaler.transform(df)  
        
        return df

    def predict(self, df):
        """
        Predict the cluster label on new data.

        Args:
            df: Dataframe on which to predict.
        """
    
        # check if the model exists 
        if not self.kmeans:
            self.load_model()
        
        df = self._preprocess(df)

        return self.kmeans.predict(df)


if __name__ == "__main__":

    model = Model()
    
    model.train_and_evaluate("../data/vulnerable_robot_challenge.csv", verbose=True)

    # ---- try load path ---------- 
    
    # prepare the data 
    #df = pd.read_csv("../data/vulnerable_robot_challenge.csv") 
    # load the data
    #model.load_model()
    # predict 
    #new_data = model.predict(df)
    #print(new_data)
    #print((new_data == 0).sum())
    #print((new_data == 1).sum())

