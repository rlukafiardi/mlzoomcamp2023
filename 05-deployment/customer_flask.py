import requests

url = 'http://localhost:1313/predict'

customer = {
    "job": "unknown",
    "duration": 270,
    "poutcome": "failure"
}

response = requests.post(url, json=customer).json()
print("credit probability:", response)
      



