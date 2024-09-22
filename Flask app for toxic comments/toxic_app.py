from flask import Flask, render_template, url_for, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import numpy as np

app = Flask(__name__)

# Load TF-IDF Vectorizers
with open(r"toxic_vect.pkl", "rb") as f:
    tox = pickle.load(f)

with open(r"severe_toxic_vect.pkl", "rb") as f:
    sev = pickle.load(f)

with open(r"obscene_vect.pkl", "rb") as f:
    obs = pickle.load(f)

with open(r"insult_vect.pkl", "rb") as f:
    ins = pickle.load(f)

with open(r"threat_vect.pkl", "rb") as f:
    thr = pickle.load(f)

with open(r"identity_hate_vect.pkl", "rb") as f:
    ide = pickle.load(f)

# Load Pre-trained Models
with open(r"toxic_model.pkl", "rb") as f:
    tox_model = pickle.load(f)

with open(r"severe_toxic_model.pkl", "rb") as f:
    sev_model = pickle.load(f)

with open(r"obscene_model.pkl", "rb") as f:
    obs_model = pickle.load(f)

with open(r"insult_model.pkl", "rb") as f:
    ins_model = pickle.load(f)

with open(r"threat_model.pkl", "rb") as f:
    thr_model = pickle.load(f)

with open(r"identity_hate_model.pkl", "rb") as f:
    ide_model = pickle.load(f)

@app.route("/")
def home():
    return render_template('index_toxic.html')

def is_toxic(predictions, threshold=0.5):
    return any(pred > threshold for pred in predictions)

@app.route("/predict", methods=['POST'])
def predict():
    user_input = request.form['text']
    data = [user_input]

    vect = tox.transform(data)
    pred_tox = tox_model.predict_proba(vect)[:, 1]

    vect = sev.transform(data)
    pred_sev = sev_model.predict_proba(vect)[:, 1]

    vect = obs.transform(data)
    pred_obs = obs_model.predict_proba(vect)[:, 1]

    vect = thr.transform(data)
    pred_thr = thr_model.predict_proba(vect)[:, 1]

    vect = ins.transform(data)
    pred_ins = ins_model.predict_proba(vect)[:, 1]

    vect = ide.transform(data)
    pred_ide = ide_model.predict_proba(vect)[:, 1]

    predictions = [pred_tox[0], pred_sev[0], pred_obs[0], pred_thr[0], pred_ins[0], pred_ide[0]]
    toxic_label = "Toxic" if is_toxic(predictions) else "Not Toxic"

    return render_template('index_toxic.html',
                           pred_tox='Prob (Toxic): {:.2f}'.format(pred_tox[0]),
                           pred_sev='Prob (Severe Toxic): {:.2f}'.format(pred_sev[0]),
                           pred_obs='Prob (Obscene): {:.2f}'.format(pred_obs[0]),
                           pred_ins='Prob (Insult): {:.2f}'.format(pred_ins[0]),
                           pred_thr='Prob (Threat): {:.2f}'.format(pred_thr[0]),
                           pred_ide='Prob (Identity Hate): {:.2f}'.format(pred_ide[0]),
                           toxic_label=toxic_label
                           )

if __name__ == "__main__":
    app.run(debug=True)
