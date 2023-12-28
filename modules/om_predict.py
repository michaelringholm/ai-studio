import os
import numpy as np
import pandas as pd
from keras.models import load_model
from tensorflow import keras
from keras.models import model_from_json
import csv

def load_model(model_weights, model_architecture):
    # Load model architecture 
    json_file = open(model_architecture, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    model.load_weights(model_weights)
    # Verify model loaded
    model.summary()
    return model

def create_validation_data():
    np.array([[100.0,105.0,95.0,102.5,1000],  
                     [107.5, 97.5, 105.0, 100,1000], 
                     [110.0, 100.0, 107.5, 100,1000]])
    return np

def load_validation_data(file_name):
    df = pd.read_csv(file_name)
    data = df[['open','high','low','close','volume']].head(5).to_numpy()
    return data

os.system('cls')
#validation_data =create_validation_data()
validation_data=load_validation_data('stock_prices.csv')

# Load the saved model   
model = load_model("model-weights.h5","model-architecture.json")

# Make predictions
predictions = model.predict(validation_data)

# Print predictions
print("Predictions:")
print(predictions)