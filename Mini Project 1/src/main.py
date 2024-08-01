import cv2 as cv
import time
from utils import load_networks, age_gender_detector

def process_camera_input():
    faceNet, ageNet, genderNet = load_networks()

    cap = cv.VideoCapture(0)
    
    print("Press 'q' to quit")
    
    previous_results = []
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("Failed to grab frame")
            break
        
        output, results = age_gender_detector(frame, faceNet, ageNet, genderNet)
        
        if output is not None:
            cv.imshow('Age and Gender Detection', output)
            
            # Check for new faces or changes
            if results != previous_results:
                for i, (gender, age) in enumerate(results):
                    print(f"Person {i+1}: Gender - {gender}, Age Range - {age}")
                previous_results = results
        else:
            print("Error: Output is empty")
        
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    process_camera_input()