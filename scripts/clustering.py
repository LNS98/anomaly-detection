

import pandas as pd 
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

def main():
    
    # import the data 
    df = pd.read_csv("./../vulnerable_robot_challenge.csv")
    
    # run clustering 
    clustering(df)


    return 0

def k_means(df):

    # drop the target label 
    y = df["flag"]
    df.drop(columns=["flag"], inplace=True)
    # normalise data 
    scaler = StandardScaler()
    df = scaler.fit_transform(df)
        
    # run k-means 
    kmeans = KMeans(n_clusters=2, random_state=0).fit(df)
    predicted_labels = kmeans.labels_
    print(predicted_labels)
    # compare the outputs to class (to a confusion matrix) 
    print((predicted_labels == y).sum()/len(y))

    return None 

if __name__ == "__main__":


    main()

