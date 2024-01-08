import numpy as np 
from flask import Flask, request
from predict import make_prediction

app = Flask(__name__)

@app.route("/", methods=["GET"])
def index():
    """Basic HTML response."""
    body = (
        "<html>"
        "<body style='padding: 10px;'>"
        "<h1>Welcome to my Machine Learning API</h1>"
        "</body>"
        "</html>"
    )
    return body

@app.route("/predict", methods=["POST"])

def predict():
    data_json = request.get_json()
    
    time_taken= data_json["time"]
    speed = data_json["speed"]
    torque = data_json["torque"]
    f1 = data_json["f1"]
    f2 = data_json["f2"]
    f3 = data_json["f3"]
    f4 = data_json["f4"]
    f5 = data_json["f5"]
    f6 = data_json["f6"]
    f7 = data_json["f7"]
    f8 = data_json["f8"]
    

    data = np.array([[time_taken, speed, torque, f1, f2, f3, f4, f5, f6, f7,
       f8]])
    predictions = make_prediction(data)
    
    return str(predictions)

if __name__ == "__main__":
    app.run()
