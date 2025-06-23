import cv2
import face_recognition
import pickle
import numpy as np
import csv
from datetime import datetime

def mark_attendance():
    with open("trained_model/model.pkl", "rb") as f:
        data = pickle.load(f)
    
    known_encodings = data["encodings"]
    known_names = data["names"]
    
    cap = cv2.VideoCapture(0)
    marked = set()
    process_this_frame = True
    
    print("Starting attendance system. Press 'q' to exit...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
            
        if process_this_frame:
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
            
            for face_encoding in face_encodings:
                matches = face_recognition.compare_faces(known_encodings, face_encoding)
                name = "Unknown"
                
                face_distances = face_recognition.face_distance(known_encodings, face_encoding)
                best_match_idx = np.argmin(face_distances)
                
                if matches[best_match_idx]:
                    name = known_names[best_match_idx]
                    
                    if name not in marked:
                        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        with open("attendance.csv", "a", newline='') as f:
                            writer = csv.writer(f)
                            writer.writerow([name, timestamp])
                        marked.add(name)
                        print(f"✅ Marked attendance for {name}")
        
        process_this_frame = not process_this_frame
        
        for (top, right, bottom, left) in face_locations:
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4
            
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
            cv2.putText(frame, name, (left + 6, bottom - 6), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1)
        
        cv2.putText(frame, f"Marked: {len(marked)}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        cv2.imshow('Attendance System', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    mark_attendance()
