import pickle
from flask import Flask
from flask import request
from flask import jsonify

model_file = 'model1.bin'
dv_file = 'dv.bin'

with open(dv_file, 'rb') as dv_in:
    dv = pickle.load(dv_in)

with open(model_file, 'rb') as m_in:
    model = pickle.load(m_in)

app = Flask('credit')

@app.route('/predict', methods=['POST'])
def predict():
    customer = request.get_json()

    X = dv.transform(customer)
    y_pred = model.predict_proba(X)[0, 1]

    results = {
        'credit_probability': float(y_pred)
    }

    return jsonify(results)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=1313)



