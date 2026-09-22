import cv2
import os
import re

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_DIR = os.path.join(BASE_DIR, "dataset")
CASCADE_PATH = os.path.join(
    BASE_DIR,
    "haarcascade",
    "haarcascade_frontalface_default.xml"
)

MAX_IMAGES = 100

os.makedirs(DATASET_DIR, exist_ok=True)

face_detector = cv2.CascadeClassifier(CASCADE_PATH)

if face_detector.empty():
    print("Error: Haar Cascade file not found.")
    print("Expected location:")
    print(CASCADE_PATH)
    exit()

name = input("Enter person's name: ").strip()

if not name:
    print("Error: Name cannot be empty.")
    exit()

name = re.sub(r'[<>:"/\\|?*]', '_', name)

person_dir = os.path.join(
    DATASET_DIR,
    name
)

os.makedirs(person_dir, exist_ok=True)

existing_images = [
    file
    for file in os.listdir(person_dir)
    if file.lower().endswith(
        (".jpg", ".jpeg", ".png")
    )
]

count = len(existing_images)

if count >= MAX_IMAGES:
    print(f"{name} already has {count} images.")
    exit()

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Error: Could not open webcam.")
    exit()

print()
print(f"Collecting face images for: {name}")
print(f"Images already available: {count}")
print(f"Target images: {MAX_IMAGES}")
print()
print("Look at the camera and move your face slightly.")
print("Press Q to stop.")
print()

while True:

    ret, frame = camera.read()

    if not ret:
        print("Error: Could not read frame.")
        break

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(100, 100)
    )

    for (x, y, w, h) in faces:

        if count >= MAX_IMAGES:
            break

        face = gray[
            y:y + h,
            x:x + w
        ]

        count += 1

        image_path = os.path.join(
            person_dir,
            f"{count}.jpg"
        )

        cv2.imwrite(
            image_path,
            face
        )

        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (255, 0, 0),
            2
        )

        cv2.putText(
            frame,
            f"Images: {count}/{MAX_IMAGES}",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 0, 0),
            2
        )

        break

    cv2.putText(
        frame,
        f"Collecting: {name}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        "Press Q to stop",
        (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 255),
        2
    )

    cv2.imshow(
        "Face Dataset Collection",
        frame
    )

    key = cv2.waitKey(100) & 0xFF

    if key == ord("q"):
        break

    if count >= MAX_IMAGES:
        break

camera.release()

cv2.destroyAllWindows()

print()
print(f"Dataset collection completed for {name}.")
print(f"Total images: {count}")
print(f"Saved in: {person_dir}")