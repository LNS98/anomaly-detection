"""
Api used for prediction on json data to the defined model.
"""

from flask import abort, Flask, jsonify, make_response, request
import pandas as pd
from model import Model

MODEL = Model()

app = Flask(__name__)


@app.route(f'/predict', methods=['POST'])
def predict():
    """
    Predict on json data using the clustering model.
    """
    
    try:        
        df_X = pd.DataFrame(request.json)
        prediction = MODEL.predict(df_X).tolist()
        
        return make_response(jsonify({'prediction': prediction}))
    
    except ValueError:
        raise RuntimeError('Data is not in the correct format.')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

