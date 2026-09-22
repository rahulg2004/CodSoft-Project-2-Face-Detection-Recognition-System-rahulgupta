import cv2
import os
import json
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_DIR = os.path.join(
    BASE_DIR,
    "dataset"
)

TRAINER_DIR = os.path.join(
    BASE_DIR,
    "trainer"
)

MODEL_PATH = os.path.join(
    TRAINER_DIR,
    "trained_model.yml"
)

LABELS_PATH = os.path.join(
    TRAINER_DIR,
    "labels.json"
)

os.makedirs(
    TRAINER_DIR,
    exist_ok=True
)


def main():

    print()
    print("=" * 60)
    print("          FACE RECOGNITION MODEL TRAINING")
    print("=" * 60)
    print()

    print("Project directory:")
    print(BASE_DIR)
    print()

    print("Dataset directory:")
    print(DATASET_DIR)
    print()

    if not os.path.exists(DATASET_DIR):

        print("ERROR: Dataset folder not found.")
        print()
        print("Run collect_faces.py first.")
        return

    if not hasattr(cv2, "face"):

        print("ERROR: OpenCV Face module is not available.")
        print()
        print("Install the correct package using:")
        print()
        print("pip uninstall opencv-python opencv-contrib-python -y")
        print("pip install opencv-contrib-python")
        return

    people = sorted(
        [
            folder
            for folder in os.listdir(DATASET_DIR)
            if os.path.isdir(
                os.path.join(
                    DATASET_DIR,
                    folder
                )
            )
        ]
    )

    if not people:

        print("ERROR: No person folders found.")
        print()
        print("Example:")
        print("dataset/Rahul/")
        print()
        print("Run collect_faces.py first.")
        return

    print("People found:")
    print()

    for person in people:

        person_path = os.path.join(
            DATASET_DIR,
            person
        )

        image_count = len(
            [
                file
                for file in os.listdir(person_path)
                if file.lower().endswith(
                    (".jpg", ".jpeg", ".png")
                )
            ]
        )

        print(
            f"  {person}: {image_count} images"
        )

    print()

    recognizer = cv2.face.LBPHFaceRecognizer_create(
        radius=1,
        neighbors=8,
        grid_x=8,
        grid_y=8
    )

    faces = []
    ids = []

    label_names = {}

    current_id = 0

    print("Loading training images...")
    print()

    for person_name in people:

        person_path = os.path.join(
            DATASET_DIR,
            person_name
        )

        image_files = sorted(
            [
                file
                for file in os.listdir(person_path)
                if file.lower().endswith(
                    (".jpg", ".jpeg", ".png")
                )
            ]
        )

        if not image_files:

            print(
                f"WARNING: No images found for "
                f"{person_name}"
            )

            continue

        current_id += 1

        label_names[current_id] = person_name

        person_images = 0

        print(
            f"Processing {person_name}..."
        )

        for image_file in image_files:

            image_path = os.path.join(
                person_path,
                image_file
            )

            image = cv2.imread(
                image_path,
                cv2.IMREAD_GRAYSCALE
            )

            if image is None:

                print(
                    f"  Could not read: "
                    f"{image_file}"
                )

                continue

            if image.size == 0:

                print(
                    f"  Empty image: "
                    f"{image_file}"
                )

                continue

            faces.append(image)
            ids.append(current_id)

            person_images += 1

        print(
            f"  Valid images: "
            f"{person_images}"
        )

    print()

    if not faces:

        print(
            "ERROR: No valid training images found."
        )

        return

    print(
        f"Total training images: {len(faces)}"
    )

    print(
        f"Total registered people: "
        f"{len(label_names)}"
    )

    print()
    print("Training LBPH model...")
    print()

    try:

        recognizer.train(
            faces,
            np.array(ids)
        )

    except cv2.error as error:

        print(
            "ERROR: Model training failed."
        )

        print()
        print(error)

        return

    recognizer.write(
        MODEL_PATH
    )

    with open(
        LABELS_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            label_names,
            file,
            indent=4,
            ensure_ascii=False
        )

    print()
    print("=" * 60)
    print("          TRAINING COMPLETED SUCCESSFULLY")
    print("=" * 60)
    print()

    print("Model:")
    print(MODEL_PATH)
    print()

    print("Labels:")
    print(LABELS_PATH)
    print()

    print("Registered people:")

    for person_id, person_name in label_names.items():

        print(
            f"  ID {person_id}: {person_name}"
        )

    print()
    print("Next step:")
    print("python face_recognition.py")
    print()


if __name__ == "__main__":
    main()