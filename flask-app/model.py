"""
Main model file.
"""

import pandas as pd 
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
import joblib
import os

class Model:


    def __init__(self):
        
        self.kmeans = None    
        self.scaler = StandardScaler()
    
    def read_in_data(self, path):
        
        # import the data 
        df = pd.read_csv(path)
        
        return df
         

    def _preprocess(self, df_X):
        
        # normalise data 
        return self.scaler.transform(df_X)  
        

    def fit_cluster(self, df, target="flag"):
        
        try:
            y = df[target]
            df = df.drop(columns=[target])
        except:
            print("Target not present")
            return None 
        
        # pre-process the data
        self.scaler.fit(df)
        self._preprocess(df)

        # k_means with k=2 to cluster the data 
        self.kmeans = KMeans(n_clusters=2, random_state=0).fit(df)
   

    def evaluate_clusters(self):
        
        # get the predicted labes 
        pred_clust = self.kmeans.labels_

        # print the accuracy 
        print(f"Accuracy on whole data set: {(pred_clust  == self.y).sum()/len(self.y)}")

    
    def save_model(self, path="./SavedModel/"):
        
        # check if path exits, if not create it 
        directory = os.path.dirname(path)
        if not os.path.exists(directory):
            os.makedirs(directory)

        joblib.dump(self.scaler, path + "scaler.pkl") # save also the preprocess
        joblib.dump(self.kmeans, path + "kmeans.pkl")
        

    def load_model(self, path="./SavedModel/"):
        
        self.scaler = joblib.load(path + "scaler.pkl") # load the preprocess
        self.kmeans = joblib.load(path + "kmeans.pkl")


    def predict(self, df_X):
    
        # check if the model exists 
        if not self.kmeans:
            self.load_model()
        
        df_X = self._preprocess(df_X)

        return self.kmeans.predict(df_X)


if __name__ == "__main__":

    model = Model()
    #model.fit_cluster()
    #model.evaluate_clusters()

    # save model
    #model.save_model()

    # ---- try load path ---------- 
    
    # prepare the data 
    df_X = model.df.drop(columns=["flag"])
    # load the data
#    model.load_model()
    # predict 
    new_data = model.predict(df_X)
    print(new_data)


