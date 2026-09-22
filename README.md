# Face Detection and Recognition System

A real-time computer vision application developed using Python and OpenCV that detects and recognizes human faces through a webcam.

This project was developed as **Task 5** of the **CodSoft Artificial Intelligence Internship**. The application combines face detection and face recognition to identify registered individuals and classify unregistered faces as **Unknown**.

---

## 📌 Project Overview

The **Face Detection and Recognition System** is an AI-based computer vision application designed to detect human faces in real time and recognize registered individuals.

The application uses the **Haar Cascade Classifier** for face detection and **LBPH (Local Binary Patterns Histograms)** for face recognition.

The system captures face images through a webcam, creates a local dataset, trains a recognition model, and then uses the trained model to identify people in real time.

The project demonstrates the complete workflow of a basic face recognition system:

```text
Face Dataset
     ↓
Face Detection
     ↓
Dataset Preparation
     ↓
Model Training
     ↓
Face Recognition
     ↓
Recognized Person / Unknown
```

---

## 🎯 Objective

The main objectives of this project are:

* Detect human faces using a webcam.
* Create a face dataset automatically.
* Train a face recognition model.
* Recognize registered individuals in real time.
* Detect multiple faces simultaneously.
* Identify unregistered faces as `Unknown`.
* Display face detection and recognition information.
* Understand practical computer vision workflows using Python and OpenCV.

---

## 🏢 Internship Information

**Organization:** CodSoft
**Internship:** Artificial Intelligence Internship
**Task:** Task 5 - Face Detection and Recognition
**Developer:** Rahul Gupta

According to the CodSoft internship task description, Task 5 requires developing an AI application capable of detecting and recognizing faces in images or videos. The suggested approaches include Haar cascades or deep-learning-based face detectors, with optional recognition techniques such as Siamese networks or ArcFace.

---

## ✨ Features

### 1. Real-Time Face Detection

The application detects faces directly from the webcam feed.

Each detected face is highlighted using a rectangular bounding box.

### 2. Face Dataset Collection

The project includes a dataset collection program that automatically captures face images from the webcam.

The collected images are stored according to the person's name.

Example:

```text
dataset/
├── Rahul/
├── Person2/
└── Person3/
```

### 3. Face Recognition

The trained LBPH model compares detected faces with the registered dataset.

When a match is found, the person's name is displayed.

Example:

```text
Rahul | 92.4%
```

### 4. Unknown Face Detection

If a detected face does not match the registered dataset within the configured recognition threshold, the application displays:

```text
Unknown
```

### 5. Multiple Face Detection

The system can detect multiple faces within the same webcam frame.

For example:

```text
Rahul
Person2
Unknown
```

can be displayed simultaneously when three faces are visible.

### 6. Face Count

The application displays the total number of detected faces.

Example:

```text
Faces: 3
```

### 7. Recognition Statistics

The application displays:

```text
Recognized: 2
Unknown: 1
```

This provides a quick overview of the current frame.

### 8. FPS Display

The application calculates and displays an approximate frames-per-second value.

Example:

```text
FPS: 28.6
```

### 9. Simple Real-Time Interface

The webcam window contains:

* Project title
* Face count
* Recognized count
* Unknown count
* FPS
* Recognition labels
* Bounding boxes

---

# 🛠️ Technologies Used

| Technology          | Purpose                               |
| ------------------- | ------------------------------------- |
| Python              | Main programming language             |
| OpenCV              | Computer vision and webcam processing |
| OpenCV Haar Cascade | Face detection                        |
| LBPH                | Face recognition                      |
| NumPy               | Numerical operations                  |
| JSON                | Storing person labels                 |
| Webcam              | Real-time image acquisition           |

---

# 🧠 Concepts Used

## Computer Vision

Computer vision enables the application to process images and video frames and identify visual information such as human faces.

## Haar Cascade Classifier

The project uses the Haar Cascade classifier:

```text
haarcascade_frontalface_default.xml
```

It is used to locate potential face regions in webcam frames.

## Grayscale Conversion

Captured frames are converted from BGR color images into grayscale images before face detection and recognition.

This reduces the amount of image information that needs to be processed.

## LBPH Face Recognition

The project uses OpenCV's **Local Binary Patterns Histograms (LBPH) Face Recognizer**.

LBPH extracts local texture patterns from face images and uses them for recognition.

The trained model produces a recognition distance when comparing a detected face with the trained dataset.

A lower distance generally indicates a closer match.

---

# 📂 Project Structure

```text
CODSOFT_TASK5_FACE_DETECTION_RECOGNITION/
│
├── dataset/
│   └── README.md
│
├── trainer/
│   ├── trained_model.yml
│   └── labels.json
│
├── haarcascade/
│   └── haarcascade_frontalface_default.xml
│
├── collect_faces.py
├── train_model.py
├── face_recognition.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

# 📄 File Description

## `collect_faces.py`

Responsible for collecting face images using the webcam.

The program:

1. Opens the webcam.
2. Detects a face.
3. Converts the frame to grayscale.
4. Extracts the face region.
5. Saves the face image.
6. Continues until the target number of images is reached.

---

## `train_model.py`

Responsible for training the face recognition model.

The program:

1. Reads the dataset.
2. Identifies person folders.
3. Loads the face images.
4. Detects face regions.
5. Assigns numeric labels.
6. Trains the LBPH recognizer.
7. Saves the trained model.
8. Saves the person-label mapping.

The generated files are:

```text
trainer/trained_model.yml
trainer/labels.json
```

---

## `face_recognition.py`

This is the main real-time application.

It:

1. Opens the webcam.
2. Captures video frames.
3. Converts frames to grayscale.
4. Detects faces.
5. Sends detected face regions to the LBPH recognizer.
6. Compares recognition results with the configured threshold.
7. Displays the person's name or `Unknown`.
8. Displays statistics and FPS.

---

## `requirements.txt`

Contains the Python dependencies required by the project.

```text
opencv-contrib-python
numpy
```

---

## `.gitignore`

Prevents unnecessary and private files from being uploaded to GitHub.

The face dataset is excluded because facial images can contain biometric information.

---

# 💻 System Requirements

Recommended environment:

* Windows 10/11
* Python 3.10 or newer
* Working webcam
* At least 4 GB RAM
* Internet connection for initial package installation

A dedicated GPU is not required because this implementation uses OpenCV's Haar Cascade and LBPH approach.

---

# ⚙️ Installation

## Step 1: Clone the Repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

Then move into the project folder:

```bash
cd CODSOFT_TASK5_FACE_DETECTION_RECOGNITION
```

Replace `YOUR_GITHUB_REPOSITORY_URL` with the actual GitHub repository URL after uploading the project.

---

## Step 2: Create a Virtual Environment

Windows:

```bash
python -m venv venv
```

---

## Step 3: Activate the Virtual Environment

Windows PowerShell:

```bash
venv\Scripts\activate
```

You should see:

```text
(venv)
```

before the terminal path.

---

## Step 4: Install Dependencies

Run:

```bash
pip install -r requirements.txt
```

Alternatively:

```bash
pip install opencv-contrib-python numpy
```

---

# 📥 Haar Cascade Setup

The project requires:

```text
haarcascade_frontalface_default.xml
```

Place the file inside:

```text
haarcascade/
```

The final path should be:

```text
haarcascade/haarcascade_frontalface_default.xml
```

The Haar Cascade file can be obtained from the official OpenCV repository.

---

# 📸 Step 1: Create the Face Dataset

Run:

```bash
python collect_faces.py
```

The program will ask:

```text
Enter person's name:
```

Enter a name.

Example:

```text
Rahul
```

The webcam will open and start capturing face images.

The images will be saved inside:

```text
dataset/Rahul/
```

The default target is approximately **100 images per person**.

---

# 👤 Dataset Example

After collecting data for three people:

```text
dataset/
│
├── Rahul/
│   ├── 1.jpg
│   ├── 2.jpg
│   ├── 3.jpg
│   └── ...
│
├── Person2/
│   ├── 1.jpg
│   ├── 2.jpg
│   └── ...
│
└── Person3/
    ├── 1.jpg
    ├── 2.jpg
    └── ...
```

For better recognition results, collect images with slight changes in:

* Face position
* Head angle
* Facial expression
* Distance from camera
* Lighting

---

# 🧠 Step 2: Train the Model

After collecting the dataset, run:

```bash
python train_model.py
```

The program reads all available person folders and trains the LBPH face recognition model.

After successful training, the following files are created:

```text
trainer/
├── trained_model.yml
└── labels.json
```

Example `labels.json`:

```json
{
    "1": "Rahul",
    "2": "Person2",
    "3": "Person3"
}
```

The numeric IDs are generated automatically during training.

---

# 🎥 Step 3: Run Face Recognition

Start the application using:

```bash
python face_recognition.py
```

The webcam will open.

The application will detect and recognize registered faces in real time.

Press:

```text
Q
```

to close the application.

---

# 🔄 Application Workflow

The complete workflow is:

```text
             START
               │
               ▼
       Open Webcam
               │
               ▼
        Capture Frame
               │
               ▼
      Convert to Grayscale
               │
               ▼
       Detect Face(s)
               │
               ▼
     Extract Face Region
               │
               ▼
      LBPH Recognition
               │
               ▼
       Compare Distance
               │
          ┌────┴────┐
          │         │
       Match     No Match
          │         │
          ▼         ▼
        Name      Unknown
          │         │
          └────┬────┘
               │
               ▼
        Display Result
               │
               ▼
        Process Next Frame
```

---

# 🔍 Recognition Logic

The application uses:

```python
RECOGNITION_THRESHOLD = 65
```

The LBPH recognizer returns:

* A numeric person ID
* A recognition distance

The application checks whether the returned ID exists and whether the distance is below the configured threshold.

Conceptually:

```text
If person exists AND distance < threshold
        ↓
Recognized
```

Otherwise:

```text
Unknown
```

The threshold can be adjusted depending on the quality of the dataset and the environment.

---

# 📊 Recognition Score

The interface displays a percentage-style similarity value for easier visual interpretation.

For example:

```text
Rahul | 91.5%
```

This value is derived from the LBPH distance and should be treated as an approximate similarity indicator rather than a calibrated probability.

---

# 🧪 Testing

The application should be tested under different conditions.

## Test 1: Registered Person

Stand in front of the webcam using a face that exists in the dataset.

Expected result:

```text
Rahul
```

---

## Test 2: Unknown Person

Ask someone who is not included in the dataset to stand in front of the webcam.

Expected result:

```text
Unknown
```

---

## Test 3: Multiple Faces

Place multiple people in front of the camera.

Expected result:

```text
Rahul
Person2
Unknown
```

depending on the registered people present.

---

## Test 4: Different Lighting

Test the system in:

* Bright lighting
* Normal indoor lighting
* Slightly darker lighting

Recognition performance may vary depending on lighting.

---

## Test 5: Different Face Angles

Test with:

* Straight face
* Slight left turn
* Slight right turn
* Different distances

The dataset should ideally include some variation to improve performance.

---

# 🖥️ Example Output

The application interface displays information similar to:

```text
FACE DETECTION & RECOGNITION

Faces: 2  Recognized: 1  Unknown: 1
FPS: 28.4 | Press Q to Quit
```

Detected faces receive bounding boxes.

Registered faces are labelled with their corresponding names.

Unregistered faces are labelled:

```text
Unknown
```

---

# 🔐 Privacy Considerations

This project uses facial images for recognition.

The collected dataset should be treated as sensitive biometric data.

For this reason:

* Do not upload personal face datasets to a public repository.
* Do not upload another person's facial images without permission.
* Keep locally collected face images private.
* Use the project for educational and authorized purposes.

The `.gitignore` file excludes the contents of the `dataset` folder from Git tracking.

---

# ⚠️ Limitations

This is an educational face recognition system and should not be considered a production-grade biometric identification system.

Possible limitations include:

* Recognition can be affected by lighting.
* Large changes in face angle can reduce accuracy.
* Low-quality webcam images can affect recognition.
* LBPH may perform poorly in challenging environments.
* The displayed similarity percentage is not a calibrated probability.
* The system does not include advanced anti-spoofing.
* The system does not verify whether a detected face belongs to a live person.
* Recognition performance depends heavily on the quality and diversity of the training dataset.

---

# 🚀 Future Improvements

The project can be extended with more advanced techniques.

## Deep Learning Face Detection

Replace Haar Cascade with a deep learning-based face detector.

## ArcFace

Implement ArcFace-based face embeddings for stronger face recognition performance.

## Siamese Network

Use a Siamese neural network to learn face similarity.

## Face Embeddings

Generate numerical representations of faces and compare them using distance metrics.

## Anti-Spoofing

Add protection against photographs or videos being presented to the camera.

## Attendance System

The recognition system could be connected to an attendance database.

Example:

```text
Name       Date          Time
--------------------------------
Rahul      11-09-2026    11:30
Person2    11-09-2026    11:32
```

## Database Integration

Store registered users and recognition records in:

* SQLite
* MySQL
* PostgreSQL

## Web Interface

The application could be converted into a web-based system using:

* Flask
* Streamlit
* FastAPI

---

# 🧩 Troubleshooting

## Error: `No module named cv2`

Install OpenCV:

```bash
pip install opencv-contrib-python
```

---

## Error: `module 'cv2' has no attribute 'face'`

Uninstall conflicting OpenCV packages:

```bash
pip uninstall opencv-python opencv-contrib-python -y
```

Then install:

```bash
pip install opencv-contrib-python
```

Check:

```bash
python -c "import cv2; print(hasattr(cv2, 'face'))"
```

Expected:

```text
True
```

---

## Error: Haar Cascade Not Found

Check that this file exists:

```text
haarcascade/haarcascade_frontalface_default.xml
```

The project expects exactly:

```text
haarcascade/haarcascade_frontalface_default.xml
```

---

## Error: Trained Model Not Found

Run:

```bash
python train_model.py
```

before:

```bash
python face_recognition.py
```

---

## Webcam Does Not Open

Check:

* Webcam connection
* Windows camera permissions
* Whether another application is using the webcam

You can also try changing:

```python
cv2.VideoCapture(0)
```

to:

```python
cv2.VideoCapture(1)
```

if your system has multiple cameras.

---

# 📦 Complete Execution Commands

After installation, the normal workflow is:

### Create dataset

```bash
python collect_faces.py
```

### Train model

```bash
python train_model.py
```

### Start recognition

```bash
python face_recognition.py
```

---

# 📈 Skills Gained

Through this project, I gained practical experience in:

* Computer Vision
* Face Detection
* Face Recognition
* Python
* OpenCV
* Haar Cascade
* LBPH
* Dataset Preparation
* Model Training
* Real-Time Video Processing
* Webcam-Based AI Applications

---

# 🎓 Learning Outcomes

This project helped me understand how a basic face recognition pipeline works from data collection to real-time prediction.

I gained practical experience in collecting and preparing image data, detecting faces, training a recognition model, processing webcam frames, handling unknown faces, and building a complete computer vision application using Python.

---

# 📹 Project Demonstration

A video demonstration of the project can be shared on LinkedIn as part of the CodSoft internship requirements.

The demonstration can include:

1. Project introduction
2. Dataset collection
3. Model training
4. Real-time face detection
5. Registered face recognition
6. Unknown face detection
7. Multiple face detection
8. GitHub repository

---

# 🏆 Internship Task

**CodSoft Artificial Intelligence Internship**

**Task 5: Face Detection and Recognition**

The project demonstrates the practical implementation of face detection and recognition using computer vision techniques.

---

# 👨‍💻 Author

**Rahul Gupta**

B.Sc. (Hons) Computer Science
Delhi University

---

# 📜 License

This project is created for educational and internship purposes.

You may modify and extend the project for learning and development purposes.

---

# ⭐ Acknowledgement

Thanks to **CodSoft** for providing the Artificial Intelligence internship opportunity and project-based learning experience.

The internship task encouraged practical exploration of AI concepts and real-world implementation.
