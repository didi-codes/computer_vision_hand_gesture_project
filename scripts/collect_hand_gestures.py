import cv2
import os

# Debugged issue with path identification, the images should now save in the correct folders.
gesture_name = "ok"
save_path = f"/Users/damarisgarcia/Desktop/computer_vision_hand_gesture_project/dataset/{gesture_name}"
os.makedirs(save_path, exist_ok=True)

cap = cv2.VideoCapture(0)
count = 0

# Debugging to see if frame is empty as the photos were not savings!
ret,frame = cap.read()
print("ret =", ret)
print("frame is None =", frame is None)

while True: 
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    cv2.imshow("Capture Images", frame)

    key = cv2.waitKey(1)
    if key == ord(' '):
        image_name = f"{save_path}/{count}.jpg"
        cv2.imwrite(image_name, frame)
        print("Saved Image:", image_name)
        count += 1
    
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()