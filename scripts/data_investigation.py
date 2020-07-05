"""
Quick investigation of the data.
"""

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns 

def main():
    
    # load data 
    df = pd.read_csv("./../data/vulnerable_robot_challenge.csv")

    # print info, describe head
    print(df.head())
    print(df.info())
    print(df.describe())
    
    # check for missing values 
    print(f"Total number of nan values is:\n {df.isnull().sum()}\n")

    # check class distribution for target class 
    num_unique = df["flag"].value_counts()
    plt.bar(num_unique.index, num_unique, 1)
    plt.title('Class Frequency')
    plt.xlabel('Class')
    plt.ylabel('Frequency')
    plt.show()

    # pairplot to see for any obvious patterns
    #sns.pairplot(df, hue="flag")
    #plt.show()
    
    # check data distribution for +1 flag 
    df_one = df[df["flag"]==1]
    print(df_one.info())
    print(df_one.describe())    
    
    # check data distribution for 0 flag 
    df_zero = df[df["flag"]==0]
    print(df_zero.info())
    print(df_zero.describe())    
    
    return 0




if __name__ == "__main__":
    main()

