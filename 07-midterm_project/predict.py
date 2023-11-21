import pickle
from flask import Flask, request, jsonify
import numpy as np

model_file = 'xgboost_model.bin'

with open(model_file, 'rb') as f_in:
    (ohe, model) = pickle.load(f_in)

app = Flask('subscribed')

categorical_features_list = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome', 'age_group', 'contacted']
numerical_features_list = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']

@app.route('/predict', methods=['POST'])
def predict():
    customer = request.get_json()

    # Separate features into categorical and numerical based on lists
    categorical_features = [customer[feature] for feature in categorical_features_list]
    numerical_features = [customer[feature] for feature in numerical_features_list]

    # Transform categorical features using one-hot encoder
    X_categorical = ohe.transform([categorical_features])

    # Concatenate transformed categorical features with numerical features
    X = np.concatenate([X_categorical, np.array([numerical_features])], axis=1)

    # Make prediction
    y_pred = model.predict_proba(X)[0, 1]

    results = {
        'subscribed_probability': float(y_pred),
    }

    return jsonify(results)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=1313)