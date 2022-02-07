import cv2
import mediapipe as mp
import numpy as np
import datetime
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose


#Function to calculate angle between a joint.
def calculate_angle(a,b,c):
    a = np.array(a) # First
    b = np.array(b) # Mid
    c = np.array(c) # End
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)  
    return angle 


cap = cv2.VideoCapture("KneeBend.mp4")
angle_at_knee_thresh = 150
time_thresh = 8
time_diff = 0
knee_bent = False
rep_counted = False
rep_count = 0
now = 0
## Setup mediapipe instance
with mp_pose.Pose(min_detection_confidence=0.4, min_tracking_confidence=0.4) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if np.shape(frame) == ():
            break
        
        # Recolor image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
      
        # Make detection
        results = pose.process(image)
    
        # Recolor back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Extract landmarks
        try:
            landmarks = results.pose_landmarks.landmark
            
            (hip, knee, ankle) = (23, 25, 27) if (landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].z < landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].z) else (24, 26, 28)
            hip_coord = [landmarks[hip].x, landmarks[hip].y]
            knee_coord = [landmarks[knee].x, landmarks[knee].y]
            ankle_coord = [landmarks[ankle].x, landmarks[ankle].y]
            angle_at_knee = calculate_angle(hip_coord, knee_coord, ankle_coord)
            image = cv2.putText(image,"Knee Angle:" + str(angle_at_knee), (int(landmarks[knee].x * 640), int(landmarks[knee].y * 480)),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)
            
            if angle_at_knee > angle_at_knee_thresh:
                if time_diff < 8:
                    image = cv2.putText(image, "Keep Your Knee bent", (20, 70),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)
                knee_bent = False
            
            if angle_at_knee < angle_at_knee_thresh and knee_bent==False:
                knee_bent = True
                rep_counted = False
                now = datetime.datetime.now()
                
            if knee_bent:
                time_diff = (datetime.datetime.now() - now).total_seconds()
                image = cv2.putText(image,"Time held:" + str(time_diff), (20, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)
                
            if time_diff > time_thresh and rep_counted==False:
                rep_count += 1
                rep_counted = True
        except:
            pass
        
        image = cv2.putText(image,"Rep count:" + str(rep_count), (20, 460),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)
        
        # Render detections
        #mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
        #                        mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
        #                        mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
        #                         )               
        
        cv2.imshow('Mediapipe Feed', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

