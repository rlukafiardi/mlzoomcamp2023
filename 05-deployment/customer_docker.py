import requests

url = 'http://localhost:1313/predict'

customer = {
    "job": "retired",
    "duration": 445,
    "poutcome": "success"
}

response = requests.post(url, json=customer).json()
print("credit probability:", response)
      



