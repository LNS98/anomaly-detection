"""
Main model file.
"""

from itertools import chain, combinations
import pandas as pd 
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
import joblib
import os

class Model:

    def __init__(self):
        
        self.best_feats = None
        self.kmeans = None    
        self.scaler = StandardScaler()
    
    def train_and_evaluate(self, path, verbose=False):
        """
        Read in data from a csv file and return a pandas dataframe.

        Args:
            path: File path to data directory.
        """

        self.verbose = verbose
        # import the data 
        df = pd.read_csv(path) 
        # do feature eng 
        df = self._feature_eng(df)
        # do feature selection
        self._feature_selection(df) 
        # fit to best feats
        self.fit_clusters(df, self.best_feats) 
        # save model
        self.save_model()

    def _preprocess(self, df):
        """
        Preprocess the data before feeeding it to the model. 
        Feature selection and normalisation is performed on a given dataset.

        Args:
            df: Dataframe consisting only of the feature to preprocess.
        """
        
        # perform feat eng 
        df = self._feature_eng(df)
        # select best features from feat selection
        df = df[self.best_feats]
        # normalise data 
        df = self.scaler.transform(df_X)  
        
        return df

    def _feature_eng(self, df):

        # as the RxKBTot and the RxKBTot are the most important features, play around with them
        df["R_per_T"] = df["RxKBTot"]/df["TxKBTot"] 
        df["T_per_R"] = df["TxKBTot"]/df["RxKBTot"]

        df["R_per_CPU"] = df["RxKBTot"]/df["CPU"]
        df["T_per_CPU"] = df["TxKBTot"]/df["CPU"]
        
        # replace nans with high values
        df = df.replace(np.inf, 1e4)
        df = df.dropna()
    
        return df

    def _feature_selection(self, df):

        # get a list of features
        all_features = list(df.columns)
        # drop the target label 
        y = df["flag"]
        
        # drop the time (t) as it should not matter and the flag 
        del all_features[all_features.index("flag")]
        del all_features[all_features.index("t")]
        
        # delete these extra features because they don't seem very useful
        del all_features[all_features.index("Watts")]
        del all_features[all_features.index("RMS")]
        del all_features[all_features.index("diff_encoder_l")]
        
        # check importance at the start
        self.best_feats, best_score = self._score_all_features(df, y, all_features)
        
    def _score_all_features(self, df, y, features):
         
        # get the powerset to check all feature combinatations (there are only 2**8=256 possible combinations without feat eng)
        power_set = list(chain.from_iterable(combinations(features, r) for r in range(len(features)+1)))
        
        best_score = 0
        best_feats = None 
        for feat_sel in power_set:
            
            # if empty set skip
            if len(feat_sel) == 0:
                continue

            # fit k-means to this data 
            self.fit_clusters(df, list(feat_sel))
            # evaluate the clusters 
            score = self.evaluate_clusters(y, feat_sel, self.verbose)
            
            # keep track of best score and feat
            if score > best_score:
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
            target: Name of the target feature in the dataset (which will be dropped).
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
            print(f"Feature Selection: {features}, Accuracy: {score}")
        
        return score

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


    def predict(self, df_X):
        """
        Predict the cluster label on new data.

        Args:
            df_X: Dataframe on which to predict.
        """
    
        # check if the model exists 
        if not self.kmeans:
            self.load_model()
        
        df_X = self._preprocess(df_X)

        return self.kmeans.predict(df_X)


if __name__ == "__main__":

    model = Model()
    
    model.train_and_evaluate("../data/vulnerable_robot_challenge.csv", verbose=True)

    # ---- try load path ---------- 
    
    # prepare the data 
    #df_X = model.df.drop(columns=["flag"])
    # load the data
    #model.load_model()
    # predict 
    #new_data = model.predict(df_X)
    #print(new_data)


