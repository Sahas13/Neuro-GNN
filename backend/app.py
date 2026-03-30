from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import numpy as np

from model import SeizureGNN
from eeg_graph_utils import eeg_to_graph

app = Flask(__name__)
CORS(app)

model = SeizureGNN()
model.load_state_dict(torch.load("seizure_gnn.pth"))
model.eval()

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    eeg_values = np.array(data["eeg"])

    graph = eeg_to_graph(eeg_values)

    with torch.no_grad():
        output = model(graph)
        prediction = torch.argmax(output).item()

    return jsonify({
        "prediction": "Seizure Detected" if prediction == 1 else "Normal EEG",
        "class": prediction
    })

if __name__ == "__main__":
    app.run(debug=True)
