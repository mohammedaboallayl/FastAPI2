import requests 

def ganderquery(img,API_TOKEN):  
    API_URL = "https://api-inference.huggingface.co/models/rizvandwiki/gender-classification" 
    headers = {"Authorization": f"Bearer {API_TOKEN}"} 
    response = requests.post(API_URL, headers=headers, data=img) 
    return  response.json() 