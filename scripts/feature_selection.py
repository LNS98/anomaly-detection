"""
Perform different kinds of feature selection and evaluate them on the kmeans pipeline.
"""

from itertools import chain, combinations
import pandas as pd 
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

def main():
    
    # Run feat importance 
    important_features()

    return 0

def k_means(df, y, features, verbose=False):
    
    # keep only the selected features
    df = df[features]

    # normalise data 
    scaler = StandardScaler()
    df = scaler.fit_transform(df)
        
    # run k-means 
    kmeans = KMeans(n_clusters=2, random_state=0).fit(df)
    predicted_labels = kmeans.labels_
    # compare the outputs to class (to a confusion matrix)
    score = (predicted_labels == y).sum()/len(y)
    
    if verbose:
        print(f"Feature Selection: {features}, Accuracy: {score}")

    return score 

def important_features():
    
    # read in data 
    df = pd.read_csv("./../data/vulnerable_robot_challenge.csv")
    
    # add features from feat eng
    df = feature_eng(df)

    # get a list of features
    all_features = list(df.columns)
    # drop the target label 
    y = df["flag"]
    
    # drop the time (t) as it should not matter and the flag 
    del all_features[all_features.index("flag")]
    del all_features[all_features.index("t")]
    
    # score the pca 
    pca_score = score_pca(df, y)

    # delete these extra features because they don't seem very useful
    del all_features[all_features.index("Watts")]
    del all_features[all_features.index("Amps")]
    del all_features[all_features.index("RMS")]
    del all_features[all_features.index("diff_encoder_l")]
    
    # check importance at the start
    best_feats, best_score = score_all_features(df, y, all_features)


def feature_eng(df):
    
    # as the RxKBTot and the RxKBTot are the most important features, play around with them
    df["R_per_T"] = df["RxKBTot"]/df["TxKBTot"] 
    df["T_per_R"] = df["TxKBTot"]/df["RxKBTot"]

    df["R_per_CPU"] = df["RxKBTot"]/df["CPU"]
    df["T_per_CPU"] = df["TxKBTot"]/df["CPU"]
    
    # replace nans with high values
    df = df.replace(np.inf, 1e4)
    df = df.dropna()
    
    return df


def score_all_features(df, y, all_features): 

    # get the powerset to check all feature combinatations (there are only 2**8=256 possible combinations without feat eng)
    power_set = list(chain.from_iterable(combinations(all_features, r) for r in range(len(all_features)+1)))
    
    best_score = 0
    best_feats = None 
    for feat_sel in power_set:
        
        # if empty set skip
        if len(feat_sel) == 0:
            continue

        # compute k-means accuracy using the target column
        score = k_means(df, y, list(feat_sel), verbose=False)
        
        # keep track of best score and feat
        if score > best_score:
            best_score = score
            best_feats = feat_sel

    
    # print the best scores and feat 
    print("--------------------- Best Feat Selection -------------------------")
    print(f"Feature Selection: {best_feats}, Accuracy: {best_score}")
    
    return best_feats, best_score

def score_pca(df, y):
    
    pca = PCA(n_components=2)
    
    # apply pca to the df
    X = pca.fit_transform(df)
    df_pca = pd.DataFrame.from_records(X)
    
    print(f"Explained Variance: {pca.explained_variance_ratio_}")
    
    # evaluate k-means on new dataset 
    features = list(df_pca.columns)
    score = k_means(df_pca, y, features)
    
    print(f"PCA Score: {score}")

    return score


if __name__ == "__main__":

    main()


