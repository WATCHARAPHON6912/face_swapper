import numpy as np
import os
import glob
import cv2
import matplotlib.pyplot as plt
import insightface
from insightface.app import FaceAnalysis
import pyvirtualcam
import keyboard as kb
from insightface.data import get_image as ins_get_inmage
app = FaceAnalysis(name='buffalo_l')
app.prepare(ctx_id=0,det_size=(640,640))
swapper = insightface.model_zoo.get_model("inswapper_128.onnx",download=False,download_zip=False,)
def face_swapper(org_img,img):
    faces = app.get(img)
    org_img = app.get(org_img)
    for face in faces:
        img=swapper.get(img,face,org_img[0],paste_back=True)
    return img

if __name__ =="__main__":
    cap = cv2.VideoCapture(0) 
    with pyvirtualcam.Camera(width=1280, height=720, fps=60) as cam:
        print(f'Using virtual camera: {cam.device}')
        frame = np.zeros((cam.height, cam.width, 3), np.uint8)  # RGB
        while True:
            success, img = cap.read()
            if not kb.is_pressed("s"):
                img =face_swapper(cv2.imread("t.jpg"),img)
            img=cv2.resize(img, (1280,720))
        
            cam.send( img[:,:,::-1])
            cam.sleep_until_next_frame()
    cap.release()
    cv2.destroyAllWindows()