import pickle 
import pandas as pd

model = pickle.load("app_engine/xg_boost_model.pkl")

def make_prediction(inputs): 
    """
    Make a prediction using the trained model 
    """
    inputs_df = pd.DataFrame(
        inputs, 
        columns=['time', 'speed', 'torque', 'f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7',
       'f8']
        )
    predictions = model.predict(inputs_df)
    
    return predictions