import cv2
import os
import json
import time

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

CASCADE_PATH = os.path.join(
    BASE_DIR,
    "haarcascade",
    "haarcascade_frontalface_default.xml"
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "trainer",
    "trained_model.yml"
)

LABELS_PATH = os.path.join(
    BASE_DIR,
    "trainer",
    "labels.json"
)

RECOGNITION_THRESHOLD = 65

WINDOW_NAME = "Face Detection and Recognition"


def check_file(path, description):

    if not os.path.exists(path):

        print()
        print(
            f"ERROR: {description} not found."
        )

        print()
        print(
            "Expected location:"
        )

        print(path)

        print()

        return False

    return True


def calculate_similarity(distance):

    score = 100 - distance

    score = max(
        0,
        min(
            100,
            score
        )
    )

    return score


def main():

    print()
    print("=" * 65)
    print("          FACE DETECTION & RECOGNITION SYSTEM")
    print("=" * 65)
    print()

    print("Project directory:")
    print(BASE_DIR)
    print()

    if not check_file(
        CASCADE_PATH,
        "Haar Cascade file"
    ):
        return

    if not check_file(
        MODEL_PATH,
        "Trained model"
    ):
        print()
        print(
            "Run train_model.py first."
        )
        return

    if not check_file(
        LABELS_PATH,
        "Labels file"
    ):
        print()
        print(
            "Run train_model.py first."
        )
        return

    if not hasattr(cv2, "face"):

        print()
        print(
            "ERROR: OpenCV Face module "
            "is not available."
        )

        print()
        print(
            "Run:"
        )

        print(
            "pip uninstall "
            "opencv-python "
            "opencv-contrib-python -y"
        )

        print(
            "pip install opencv-contrib-python"
        )

        return

    face_detector = cv2.CascadeClassifier(
        CASCADE_PATH
    )

    if face_detector.empty():

        print()
        print(
            "ERROR: Haar Cascade could "
            "not be loaded."
        )

        return

    recognizer = cv2.face.LBPHFaceRecognizer_create()

    try:

        recognizer.read(
            MODEL_PATH
        )

    except cv2.error as error:

        print()
        print(
            "ERROR: Could not load trained model."
        )

        print()
        print(error)

        return

    try:

        with open(
            LABELS_PATH,
            "r",
            encoding="utf-8"
        ) as file:

            labels = json.load(file)

    except Exception as error:

        print()
        print(
            "ERROR: Could not load labels."
        )

        print()
        print(error)

        return

    labels = {
        int(key): value
        for key, value in labels.items()
    }

    print(
        f"Registered people: {len(labels)}"
    )

    for person_id, name in labels.items():

        print(
            f"  ID {person_id}: {name}"
        )

    print()
    print(
        f"Recognition threshold: "
        f"{RECOGNITION_THRESHOLD}"
    )

    print()
    print("Starting webcam...")
    print()
    print("Controls:")
    print("Q - Quit")
    print("S - Save screenshot")
    print()

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():

        print()
        print(
            "ERROR: Could not open webcam."
        )

        print()
        print(
            "Try changing:"
        )

        print(
            "cv2.VideoCapture(0)"
        )

        print(
            "to:"
        )

        print(
            "cv2.VideoCapture(1)"
        )

        return

    camera.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        1280
    )

    camera.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        720
    )

    previous_time = time.time()

    fps = 0

    screenshot_dir = os.path.join(
        BASE_DIR,
        "screenshots"
    )

    os.makedirs(
        screenshot_dir,
        exist_ok=True
    )

    print(
        "Face Recognition System started."
    )

    print()

    while True:

        ret, frame = camera.read()

        if not ret:

            print(
                "ERROR: Could not read "
                "webcam frame."
            )

            break

        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )

        faces = face_detector.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(80, 80)
        )

        recognized_count = 0
        unknown_count = 0

        for (x, y, w, h) in faces:

            face_region = gray[
                y:y + h,
                x:x + w
            ]

            if face_region.size == 0:
                continue

            try:

                person_id, distance = recognizer.predict(
                    face_region
                )

                similarity = calculate_similarity(
                    distance
                )

            except cv2.error:

                person_id = -1
                distance = 999
                similarity = 0

            if (
                person_id in labels
                and distance < RECOGNITION_THRESHOLD
            ):

                name = labels[
                    person_id
                ]

                recognized_count += 1

                label = (
                    f"{name} | "
                    f"Score: {similarity:.1f}%"
                )

                box_color = (
                    0,
                    255,
                    0
                )

            else:

                name = "Unknown"

                unknown_count += 1

                label = (
                    f"Unknown | "
                    f"Score: {similarity:.1f}%"
                )

                box_color = (
                    0,
                    0,
                    255
                )

            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                box_color,
                2
            )

            label_width = min(
                w + 120,
                frame.shape[1] - x
            )

            label_height = 30

            label_y = y - label_height

            if label_y < 100:

                label_y = y + h

            cv2.rectangle(
                frame,
                (
                    x,
                    label_y
                ),
                (
                    x + label_width,
                    label_y + label_height
                ),
                box_color,
                -1
            )

            cv2.putText(
                frame,
                label,
                (
                    x + 5,
                    label_y + 21
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (
                    255,
                    255,
                    255
                ),
                1,
                cv2.LINE_AA
            )

        current_time = time.time()

        elapsed = (
            current_time
            - previous_time
        )

        if elapsed > 0:

            instant_fps = 1 / elapsed

            fps = (
                0.9 * fps
                + 0.1 * instant_fps
            )

        previous_time = current_time

        header_height = 105

        cv2.rectangle(
            frame,
            (0, 0),
            (
                frame.shape[1],
                header_height
            ),
            (0, 0, 0),
            -1
        )

        cv2.putText(
            frame,
            "FACE DETECTION & RECOGNITION",
            (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (
                255,
                255,
                255
            ),
            2,
            cv2.LINE_AA
        )

        cv2.putText(
            frame,
            (
                f"Faces: {len(faces)}   "
                f"Recognized: {recognized_count}   "
                f"Unknown: {unknown_count}"
            ),
            (15, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (
                0,
                255,
                255
            ),
            2,
            cv2.LINE_AA
        )

        cv2.putText(
            frame,
            (
                f"FPS: {fps:.1f}   "
                f"Threshold: {RECOGNITION_THRESHOLD}"
            ),
            (15, 88),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (
                255,
                255,
                255
            ),
            1,
            cv2.LINE_AA
        )

        cv2.imshow(
            WINDOW_NAME,
            frame
        )

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):

            break

        elif key == ord("s"):

            timestamp = time.strftime(
                "%Y%m%d_%H%M%S"
            )

            screenshot_path = os.path.join(
                screenshot_dir,
                f"screenshot_{timestamp}.jpg"
            )

            cv2.imwrite(
                screenshot_path,
                frame
            )

            print()
            print(
                "Screenshot saved:"
            )
            print(
                screenshot_path
            )
            print()

    camera.release()

    cv2.destroyAllWindows()

    print()
    print("=" * 65)
    print("       FACE RECOGNITION SYSTEM STOPPED")
    print("=" * 65)
    print()


if __name__ == "__main__":
    main()