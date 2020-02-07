import requests

url = 'http://localhost:5000/predict_api'
r = requests.post(url,json={'images':"https://images.unsplash.com/photo-1560782202-154b39d57ef2?ixlib=rb-1.2.1&ixid=eyJhcHBfaWQiOjEyMDd9&auto=format&fit=crop&w=500&q=60"})

print(r.json())