import requests 

def agequery(image,API_TOKEN): 
    API_URL = "https://api-inference.huggingface.co/models/nateraw/vit-age-classifier" 
    headers = {"Authorization": f"Bearer {API_TOKEN}"} 
    
    response = requests.post(API_URL, headers=headers, data=image)
    return response.json() 

