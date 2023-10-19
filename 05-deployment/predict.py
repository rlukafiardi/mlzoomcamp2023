import pickle

model_file = 'model1.bin'
dv_file = 'dv.bin'

with open(dv_file, 'rb') as dv_in:
    dv = pickle.load(dv_in)

with open(model_file, 'rb') as m_in:
    model = pickle.load(m_in)

customer = {
    "job": "retired",
    "duration": 445,
    "poutcome": "success"
}

X = dv.transform(customer)
y_pred = model.predict_proba(X)[0, 1]

print("input:", X)
print("credit_probability:", y_pred)