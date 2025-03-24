import cv2
import os

# Ensure correct path to cascade file
cascade_file = os.path.join(os.getcwd(), "z number plate", "haarcascade_russian_plate_number.xml")

# Check if the file exists
if not os.path.exists(cascade_file):
    print(f"⚠️ Error: Cascade file not found at {cascade_file}")
    exit()

# Load the cascade classifier
plate_cascade = cv2.CascadeClassifier(cascade_file)

# Start video capture
cap = cv2.VideoCapture(0)
cap.set(3, 640)  # Width
cap.set(4, 480)  # Height

min_area = 500
count = 0

while True:
    success, img = cap.read()
    if not success:
        print("⚠️ Could not read frame from camera.")
        break

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    plates = plate_cascade.detectMultiScale(img_gray, 1.1, 4)

    for (x, y, w, h) in plates:
        area = w * h
        if area > min_area:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(img, "Number Plate", (x, y - 5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 255), 2)

            img_roi = img[y: y + h, x:x + w]
            cv2.imshow("ROI", img_roi)

    cv2.imshow("Result", img)

    key = cv2.waitKey(1) & 0xFF  # Read key input

    if key == ord('s'):
        if not os.path.exists("plates"):
            os.makedirs("plates")
        cv2.imwrite(f"plates/scanned_img_{count}.jpg", img_roi)
        cv2.rectangle(img, (0, 200), (640, 300), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, "Plate Saved", (150, 265), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 0, 255), 2)
        cv2.imshow("Results", img)
        cv2.waitKey(500)
        count += 1

    elif key == ord('q'):  # Exit on pressing 'q'
        print("🚀 Exiting program...")
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
cv2.waitKey(1)  # Ensures all OpenCV windows close properly
