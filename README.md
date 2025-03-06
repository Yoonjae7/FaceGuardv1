# FaceGuard

A real-time face recognition system using your laptop's camera to detect and recognize individuals. The system allows you to add new persons by providing a set of images and alerts for unrecognized personnel.

## Features

- Real-time face detection and recognition
- Add new persons to the system with multiple photos
- Delete persons from the database
- List all registered individuals
- Reset the entire system
- Visual feedback with bounding boxes and confidence scores

## Installation

### Prerequisites

- Python 3.7+
- A working webcam

### Setup

1. Clone this repository:
```bash
git clone https://github.com/Yoonjae7/FaceGuard.git
cd FaceGuard
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

Note: Installing `face_recognition` may require additional system dependencies:
- On macOS: `brew install cmake`
- On Ubuntu/Debian: `sudo apt-get install -y cmake`
- On Windows: Visual C++ Build Tools might be required

## Usage

Run the main script:
```bash
python main.py
```

The program offers the following options:

1. **Start Face Detection**: Launch the webcam for real-time face recognition
2. **Add New Person**: Register a new person with their face images
3. **Delete Person**: Remove a person from the database
4. **List Registered People**: Show all people in the database
5. **Reset System**: Delete all face data
6. **Exit**: Close the program

### Detailed Instructions for Adding a Person

1. First, prepare clear face images of the person:
   - Create a folder on your computer with these images
   - Use JPG, JPEG, or PNG formats
   - Each image should contain only one face (the person you're registering)
   - 5-10 images with different angles and expressions work best

2. In the main menu, select option 2 (Add New Person)

3. Enter the person's name when prompted

4. When asked for "path to images directory", provide the full file path to the folder containing the images:
   - **Windows example**: `C:\Users\YourName\Pictures\PersonName`
   - **Mac example**: `/Users/YourName/Pictures/PersonName`
   - **Linux example**: `/home/YourName/Pictures/PersonName`

   You can copy-paste the full path from your file explorer/finder to avoid typing errors.

5. The system will process each image, extract face data, and add the person to the database

6. You'll see confirmation messages for each image processed

### Using Face Detection

After adding people to the system:

1. Select option 1 (Start Face Detection) from the main menu
2. Your webcam will activate and display the video feed
3. Known faces will be highlighted with green boxes and their names
4. Unknown faces will be highlighted with red boxes
5. Press 'q' to quit the detection
6. Press 'r' to reload the face database (if you added new people in another terminal)

### Keyboard Shortcuts (during detection)

- Press `q` to quit detection
- Press `r` to reload face database

## Project Structure

```
project_folder/
│
├── known_faces/              # Directory for storing face images (not tracked by git)
│   ├── person1/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   └── ...
│
├── known_encodings.pkl       # Face encodings database (not tracked by git)
├── main.py                   # Main program
├── requirements.txt          # Dependencies
└── README.md                 # This file
```

## Privacy and Ethical Considerations

This system processes biometric data (facial features). Please use responsibly:

- Get consent before adding someone's face to the database
- Don't use this for surveillance without proper authorization
- Be aware of local laws regarding facial recognition technology
- Secure any stored face data appropriately

