"""
Perform different kinds of feature selection and evaluate them on the kmeans pipeline.
"""

from itertools import chain, combinations
import pandas as pd 
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

# number of repeats when evaluating the k-means algo
NUM_REPEATS = 5


def main():
    
    # Run feat importance 
    important_features()

    return 0

def k_means(df, y, features, verbose=False):
    """
    Run a k-means, with k=2, on the a dataset and evaluate the performance on the
    labels for the dataset.

    Args:
        df: Dataframe containing data on which to run algorithm.
        y: Target labels for the dataset.
        features: Features selected from the original dataframe used in the algorithm.
        verbose: Display the accuracy values.

    """
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
    """
    Run an evaluation to find the most important features in the given dataset.
    """

    # read in data 
    df = pd.read_csv("./../data/vulnerable_robot_challenge.csv")
    
    # add features from feat eng
    df = feature_eng(df)
    
    # divide in validation and test 
    df_val, df_test = train_test_split(df, test_size=0.2)

    # get a list of features
    all_features = list(df.columns)
    # drop the target label 
    y_val, y_test = df_val["flag"], df_test["flag"]
    
    # drop the time (t) as it should not matter and the flag 
    del all_features[all_features.index("flag")]
    del all_features[all_features.index("t")]
    
    # score the pca 
    pca_score = score_pca(df_val, y_val)

    # delete these extra features because they don't seem very useful
    del all_features[all_features.index("Watts")]
    del all_features[all_features.index("Amps")]
    del all_features[all_features.index("RMS")]
    del all_features[all_features.index("diff_encoder_l")]
    
    # check importance at the start
    best_feats, best_score = score_all_features(df_val, y_val, all_features)
    

def feature_eng(df):
    """
    Add new features to the dataset given previous data-analysis conducted.

    Args:
        df: Dataframe on which to perform the feature engineering.
    """
    
    # as the RxKBTot and the RxKBTot are the most important features, play around with them
    df["R_per_T"] = df["RxKBTot"]/df["TxKBTot"]
    df["T_per_R"] = df["TxKBTot"]/df["RxKBTot"].replace(np.inf, 1e2)

    df["R_per_CPU"] = df["RxKBTot"]/df["CPU"].replace(np.inf, 1e3)
    df["T_per_CPU"] = df["TxKBTot"]/df["CPU"].replace(np.inf, 1e3)
    
    # replace inf values created
    df["R_per_T"].replace(np.inf, 1e2, inplace=True)
    df["T_per_R"].replace(np.inf, 1e2, inplace=True)
    df["R_per_CPU"].replace(np.inf, 1e3, inplace=True)
    df["T_per_CPU"].replace(np.inf, 1e3, inplace=True)
    
    
    # replace nans with high values
    df = df.dropna()
    
    
    return df


def score_all_features(df, y, all_features):
    """
    Perform an evaluation on the k-means algorithm by selecting the best features all the features selected.
    Returns a tuple of the best features and the evaluation score for the respective features.

    Args:
        df: Dataframe on which to train.
        y: Labels to perform evaluation of each iteration.
        all_features: All the features from which to combine and create different subsets.
    """

    # get the powerset to check all feature combinatations (there are only 2**8=256 possible combinations without feat eng)
    power_set = list(chain.from_iterable(combinations(all_features, r) for r in range(len(all_features)+1)))
    
    best_score = 0
    best_feats = None 
    for feat_sel in power_set:
        
        # if empty set skip
        if len(feat_sel) == 0:
            continue
        
        # repeat k-mans 5 times to reduce randomness
        scores = []
        for _ in range(NUM_REPEATS):
            # compute k-means accuracy using the target column
            score = k_means(df, y, list(feat_sel), verbose=True)
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

def score_pca(df, y):
    """
    PCA analysis to see whether reduced space performs better when fed into k-means algorithm.
    
    Args:
        df: Dataframe on which to apply PCA.
        y: Labels used to evaluate performace on k-means.
    """

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


