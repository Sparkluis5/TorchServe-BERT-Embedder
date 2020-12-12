import requests

text = 'Ola teste um dois acento com çédilha'

for i in range(0,1):
    r = requests.post('http://0.0.0.0:8443/predictions/bert', data = text.encode(encoding='utf-8'), headers={'Content-Type': 'application/json"'})
    response = r.json()
    print(response)
