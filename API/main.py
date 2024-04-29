from fastapi import FastAPI,UploadFile
from PIL import Image
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from io import BytesIO
import onnxruntime as ort
from Logs.detectnameanddistance import render
from Logs.detectage import agequery
from Logs.detectgander import ganderquery


FacesEmbedding=pd.read_csv("./Models/FacesMeanEmbeddings.csv",index_col=0)
persons=list(FacesEmbedding.columns)
model_path="./Models/FaceModelV5.onnx"
EP_list = [ 'CPUExecutionProvider']
Session = ort.InferenceSession(model_path,providers=EP_list)
input_name = Session.get_inputs()[0].name
output_name=Session.get_outputs()[0].name
MediapipeModelPath="./Models/face_landmarker.task"
BaseOptions=mp.tasks.BaseOptions
FaceLandMarker=mp.tasks.vision.FaceLandmarker
FaceLandMarkerOptions=mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode=mp.tasks.vision.RunningMode
FaceLandMarkerResult=mp.tasks.vision.FaceLandmarkerResult
options=FaceLandMarkerOptions(base_options=BaseOptions(model_asset_path=MediapipeModelPath),running_mode=VisionRunningMode.IMAGE)
landmarker= FaceLandMarker.create_from_options(options)



HuggingFaceTocken="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"
App=FastAPI()

@App.post("/upload")
async def detect(img:UploadFile):
    
    image=np.array(Image.open(BytesIO(img.file.read())))
    mp_img=mp.Image(image_format=mp.ImageFormat.SRGB,data=image)
    result=landmarker.detect(mp_img)
    
    if len(result.face_landmarks)==0:
        return {"state":False,"message":"No Face Found","distance":0,"name":"null","age":"0","gander":"null"}
    faceimage,name,distance=render(Session,input_name,output_name,FacesEmbedding,result,mp_img.numpy_view(),persons)
    
    textimg=BytesIO()
    Image.fromarray(image.astype("uint8")).save(textimg,"PNG")
    textimg.seek(0)
    im=Image.fromarray(image.astype("uint8"))
    rawbytes=BytesIO()
    im.save(rawbytes,"PNG")
    rawbytes.seek(0)
    
    ganders=ganderquery(rawbytes.read(),HuggingFaceTocken)
    gander=getHighScore(ganders)
    
    
    im=Image.fromarray(faceimage.astype("uint8"))
    rawbytes=BytesIO()
    im.save(rawbytes,"PNG")
    rawbytes.seek(0)
    rawbytes.seek(0)
    
    age=agequery(rawbytes.read(),HuggingFaceTocken)
    myage=getHighScore(age)
    print("sllksls")
    # "age":myage,"gander":gander
    if gander==False or myage==False:
        return {"state":False,"message":"Model Is Loading On hugging Face ","distance":0,"name":"name","age":"0","gander":"null"}
    return {"state":True,"message":"null","distance":distance,"name":name,"age":myage,"gander":gander}

def getHighScore(Scores):
    
    if type(Scores)==dict:
        return False
    scorep=0.00
    label="0-10"
    for Score_ in Scores:
        if Score_["score"] > scorep:
            scorep=float(Score_["score"])
            label=Score_["label"]
    return label