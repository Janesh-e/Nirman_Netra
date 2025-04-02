

import requests
r = requests.post(
    "https://api.deepai.org/api/torch-srgan",
    files={
        'image': open(r"C:\Users\shobh\Downloads\sam\data\test_image3.png", 'rb'),
    },
    headers={'api-key': 'e6ee2a29-7dc9-4204-bc1c-ffe3c8500bb6'}
)
print(r.json())