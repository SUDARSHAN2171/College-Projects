import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_DIR = os.path.join(BASE_DIR, "models")

faceProto = os.path.join(MODEL_DIR, "opencv_face_detector.pbtxt")
faceModel = os.path.join(MODEL_DIR, "opencv_face_detector_uint8.pb")
ageProto = os.path.join(MODEL_DIR, "age_deploy.prototxt")
ageModel = os.path.join(MODEL_DIR, "age_net.caffemodel")
genderProto = os.path.join(MODEL_DIR, "gender_deploy.prototxt")
genderModel = os.path.join(MODEL_DIR, "gender_net.caffemodel")

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList = ['Male', 'Female']

padding = 20