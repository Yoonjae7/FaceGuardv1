import cv2

def test_camera():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        return
        
    ret, frame = cap.read()
    if not ret:
        print("Cannot read frame")
    else:
        print("Camera is working!")
        print(f"Frame shape: {frame.shape}")
    
    cap.release()

test_camera()