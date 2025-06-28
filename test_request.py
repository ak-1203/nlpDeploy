import requests

url = 'http://127.0.0.1:5000/predict'
data = {'review': 'BAd'}

response = requests.post(url, json=data)
print(response.json())
