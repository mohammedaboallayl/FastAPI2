import cv2
import numpy as np
def render(Session,input_name,output_name,FacesEmbedding,results,FaceImage,persons):
    for res in results.face_landmarks:
        
        x_=int(res[145].x*FaceImage.shape[1])
        y_=int(res[145].y*FaceImage.shape[0])
        x2_=int(res[374].x*FaceImage.shape[1])
        y2_=int(res[374].y*FaceImage.shape[0])
        w=np.sqrt((x_-x2_)**2+(y_-y2_)**2)
        W=6.3
        f = 840
        d = (W * f) / w
        x=int(res[356].x*FaceImage.shape[1])
        y=int(res[152].y*FaceImage.shape[0])
        x2=int(res[162].x*FaceImage.shape[1])
        y2=int(res[338].y*FaceImage.shape[0])
        if x<FaceImage.shape[1]-10:
            x+=10
        if y>FaceImage.shape[0]-10:
            y+=10
        if x2>10:
            x2-=10
        if y2>10:
            y2-=10
        
        
        modelimg=FaceImage[y2:y,x2:x]
        
        
        if modelimg.size<9:
            continue
        modelimg=cv2.resize(modelimg,(224,224)).astype(np.float32)
        modelimg=modelimg/255
        
        distances=[]
        if d>0:
            for index,name in enumerate(persons):
                output=np.squeeze(Session.run([output_name],{f"{input_name}":np.expand_dims(modelimg,axis=0).astype(np.float16)})[0])
                personimpeding=FacesEmbedding[name].values
                distance=np.sum(np.power(output-personimpeding,2))
                distances.append(distance)
            name=persons[np.argmin(distances)]
            distance=distances[np.argmin(distances)]
            if distance <0.3:
                return modelimg,name,d
                
            else:
                return modelimg,"UnKnow",d
                

