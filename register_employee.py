import cv2
import os

def register_employee(emp_id, name):
    # Create employee directory
    emp_dir = f"dataset/{emp_id}_{name}"
    os.makedirs(emp_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(0)
    count = 0
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    print(f"Starting registration for {name}. Please face the camera directly...")
    
    while count < 30:  # Reduced to 30 quality images
        ret, frame = cap.read()
        if not ret:
            continue
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        for (x, y, w, h) in faces:
            # Save cropped face only
            face_img = frame[y:y+h, x:x+w]
            # Resize for consistency
            face_img = cv2.resize(face_img, (200, 200))
            cv2.imwrite(f"{emp_dir}/{count}.jpg", face_img)
            count += 1
            
            # Visual feedback
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, f"Captured: {count}/30", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow('Registration', frame)
        if cv2.waitKey(1) & 0xFF == ord('q') or count >= 30:
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print(f"✅ Successfully registered {name} with {count} images")

if __name__ == "__main__":
    emp_id = input("Enter Employee ID: ")
    name = input("Enter Full Name: ")
    register_employee(emp_id, name)
