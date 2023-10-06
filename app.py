import sys
import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import xgboost as xgb
import traceback
import pandas as pd

app = Flask(__name__)
CORS(app)
curr_loc = os.path.dirname(os.path.realpath(__file__))

fertilizer = pd.read_csv(os.path.realpath(os.path.join(
    curr_loc, "dataset/fertilizer.csv")))
fertilizer.drop("Unnamed: 0", axis=1, inplace=True)
crop_recomm = pd.read_csv(os.path.realpath(os.path.join(
    curr_loc, "dataset/crop_recommendation.csv")))

# model = joblib.load(os.path.join(curr_loc, "model.pkl"))
model = xgb.XGBClassifier()
model.load_model(os.path.join(curr_loc, "crop-recommendation-model.json"))
print("Model loaded")


@app.route("/")
def index():
    return "<h1>Crop Recommendation API</h1>"


@app.route("/recommend", methods=["POST"])
def recommend():
    if model:
        try:
            reqData = request.json

            prediction = model.predict(pd.DataFrame([reqData]))

            return jsonify({"recommended": list(prediction)})
        except:
            return jsonify({"error": traceback.format_exc()})
    else:
        print("ML model not loaded.")


@app.route("/crops", methods=["GET"])
def getAllCrops():
    try:
        if request.args.get("name") is None:
            return jsonify({"crops": list(fertilizer.T.to_dict().values())})

        indexes = fertilizer.index[fertilizer["crop"]
                                   == request.args.get("name")].tolist()

        if len(indexes) < 1:
            return jsonify({"error": "Crop Not Found"})

        return jsonify({"crop": fertilizer.loc[indexes[0]].to_dict()})
    except:
        return jsonify({"error": traceback.format_exc()})


if __name__ == "__main__":
    app.run(debug=True)
