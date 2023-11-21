import requests

url = 'http://localhost:1313/predict'

customer = {
    'age': 32,
    'job': 'management',
    'marital': 'single',
    'education': 'tertiary',
    'default': 'no',
    'balance': 710,
    'housing': 'no',
    'loan': 'no',
    'contact': 'cellular',
    'day': 5,
    'month': 'aug',
    'duration': 683,
    'campaign': 2,
    'pdays': -1,
    'previous': 0,
    'poutcome': 'success',
    'age_group': '31-45',
    'contacted': 'yes'
}
  
response = requests.post(url, json=customer).json()
print(response)