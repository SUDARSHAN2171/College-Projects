import cv2 as cv
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def load_networks():
    faceProto = os.path.join(BASE_DIR, "models", "opencv_face_detector.pbtxt")
    faceModel = os.path.join(BASE_DIR, "models", "opencv_face_detector_uint8.pb")
    ageProto = os.path.join(BASE_DIR, "models", "age_deploy.prototxt")
    ageModel = os.path.join(BASE_DIR, "models", "age_net.caffemodel")
    genderProto = os.path.join(BASE_DIR, "models", "gender_deploy.prototxt")
    genderModel = os.path.join(BASE_DIR, "models", "gender_net.caffemodel")

    faceNet = cv.dnn.readNet(faceModel, faceProto)
    ageNet = cv.dnn.readNet(ageModel, ageProto)
    genderNet = cv.dnn.readNet(genderModel, genderProto)

    return faceNet, ageNet, genderNet

def getFaceBox(net, frame, conf_threshold=0.7):
    frameOpencvDnn = frame.copy()
    frameHeight = frameOpencvDnn.shape[0]
    frameWidth = frameOpencvDnn.shape[1]
    blob = cv.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)

    net.setInput(blob)
    detections = net.forward()
    bboxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            bboxes.append([x1, y1, x2, y2])
            cv.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0), int(round(frameHeight/150)), 8)
    return frameOpencvDnn, bboxes

def age_gender_detector(frame, faceNet, ageNet, genderNet):
    frameFace, bboxes = getFaceBox(faceNet, frame)
    
    ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
    genderList = ['Male', 'Female']

    results = []
    
    for bbox in bboxes:
        face = frame[max(0,bbox[1]):min(bbox[3],frame.shape[0]-1),max(0,bbox[0]):min(bbox[2], frame.shape[1]-1)]
        
        blob = cv.dnn.blobFromImage(face, 1.0, (227, 227), (78.4263377603, 87.7689143744, 114.895847746), swapRB=False)
        
        genderNet.setInput(blob)
        genderPreds = genderNet.forward()
        gender = genderList[genderPreds[0].argmax()]
        
        ageNet.setInput(blob)
        agePreds = ageNet.forward()
        age = ageList[agePreds[0].argmax()]
        
        label = f"{gender}, {age}"
        cv.putText(frameFace, label, (bbox[0], bbox[1]-10), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv.LINE_AA)
        
        results.append((gender, age))
    
    return frameFace, results