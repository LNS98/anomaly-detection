"""
Transform the data into JSON formato to test app.
"""


import pandas as pd 


if __name__ == "__main__":

    data_path_in = "vulnerable_robot_challenge.csv"
    data_path_out = "test_json_data.json"
    data_path_out_one_datapoint = "one_data_point.json"

    df = pd.read_csv(data_path_in)
    df.drop(columns="flag", inplace=True)
    df.to_json(data_path_out)
    
    df_one = df.sample(1)
    df_one.to_json(data_path_out_one_datapoint)



