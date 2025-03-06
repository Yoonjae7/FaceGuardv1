import cv2
import face_recognition
import numpy as np
import pickle
import os
from pathlib import Path
from typing import List, Tuple, Optional

class FaceRecognitionSystem:
    def __init__(self):
        self.known_encodings = []
        self.known_names = []
        self.encodings_file = "known_encodings.pkl"
        self.face_tolerance = 0.45  # Stricter matching threshold
        self.load_known_faces()

    def load_known_faces(self) -> None:
        """Load known face encodings from pickle file if it exists."""
        if os.path.exists(self.encodings_file):
            try:
                with open(self.encodings_file, "rb") as f:
                    self.known_encodings, self.known_names = pickle.load(f)
                    print(f"Loaded {len(self.known_names)} known faces")
            except Exception as e:
                print(f"Error loading known faces: {e}")
                self.known_encodings = []
                self.known_names = []

    def save_known_faces(self) -> None:
        """Save current face encodings to pickle file."""
        try:
            with open(self.encodings_file, "wb") as f:
                pickle.dump((self.known_encodings, self.known_names), f)
                print("Saved face encodings to file")
        except Exception as e:
            print(f"Error saving known faces: {e}")

    def add_person(self, name: str, image_paths: List[str]) -> bool:
        """Add a new person to the recognition system."""
        try:
            successful_encodings = 0
            for img_path in image_paths:
                # Load and resize image for faster processing
                image = cv2.imread(img_path)
                if image is None:
                    print(f"Warning: Could not load image {img_path}")
                    continue
                    
                # Convert BGR to RGB
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # Detect faces with more accurate CNN model
                face_locations = face_recognition.face_locations(rgb_image, model="hog")
                
                if len(face_locations) != 1:
                    print(f"Warning: Skipping {img_path} - found {len(face_locations)} faces")
                    continue
                    
                # Compute face encoding with increased accuracy
                encoding = face_recognition.face_encodings(rgb_image, 
                                                         face_locations,
                                                         num_jitters=3)[0]
                
                # Check if this face is too similar to existing faces
                if self.known_encodings:
                    face_distances = face_recognition.face_distance(self.known_encodings, encoding)
                    if np.min(face_distances) < self.face_tolerance:
                        print(f"Warning: Skipping {img_path} - too similar to existing face")
                        continue

                self.known_encodings.append(encoding)
                self.known_names.append(name)
                successful_encodings += 1
                print(f"Added face from {img_path}")
            
            if successful_encodings > 0:
                self.save_known_faces()
                print(f"Successfully added {successful_encodings} faces for {name}")
                return True
            else:
                print("No valid faces were added")
                return False
            
        except Exception as e:
            print(f"Error adding person: {str(e)}")
            return False

    def delete_person(self, name: str) -> bool:
        """Delete all face encodings for a specific person."""
        try:
            if not self.known_names:
                print("No faces in database")
                return False

            if name not in self.known_names:
                print(f"No person named '{name}' found in database")
                return False

            # Find all indices where this name appears
            indices_to_remove = [i for i, n in enumerate(self.known_names) if n == name]
            
            # Remove the encodings and names in reverse order
            for index in sorted(indices_to_remove, reverse=True):
                self.known_encodings.pop(index)
                self.known_names.pop(index)

            self.save_known_faces()
            print(f"Successfully deleted {len(indices_to_remove)} faces for {name}")
            return True

        except Exception as e:
            print(f"Error deleting person: {str(e)}")
            return False

    def list_known_people(self) -> None:
        """Display all unique names in the database."""
        if not self.known_names:
            print("No faces in database")
            return

        unique_names = sorted(set(self.known_names))
        print("\nRegistered people:")
        for i, name in enumerate(unique_names, 1):
            count = self.known_names.count(name)
            print(f"{i}. {name} ({count} face{'s' if count > 1 else ''})")

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, List[str]]:
        """Process a video frame and recognize faces."""
        if frame is None:
            return None, []

        if not self.known_encodings:
            cv2.putText(frame, "No faces in database!", (10, 30), 
                        cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 1)
            return frame, []

        # Resize frame for faster processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        # Show debug info
        cv2.putText(frame, f"Known faces: {len(self.known_names)}", (10, 30), 
                    cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0), 1)

        # Detect faces
        face_locations = face_recognition.face_locations(rgb_small_frame, model="hog")
        detected_names = []

        if not face_locations:
            cv2.putText(frame, "No face detected", (10, 60), 
                        cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 1)
            return frame, []

        try:
            # Get face encodings with more jitters for accuracy
            face_encodings = face_recognition.face_encodings(rgb_small_frame, 
                                                           face_locations,
                                                           num_jitters=3)

            # Scale back up face locations
            face_locations = [(top * 4, right * 4, bottom * 4, left * 4) 
                            for (top, right, bottom, left) in face_locations]

            # Process each face
            for face_encoding, face_location in zip(face_encodings, face_locations):
                # Calculate face distances to all known faces
                face_distances = face_recognition.face_distance(self.known_encodings, face_encoding)
                
                if len(face_distances) > 0:
                    best_match_index = np.argmin(face_distances)
                    min_distance = face_distances[best_match_index]
                    
                    # Only consider it a match if the distance is below our strict threshold
                    if min_distance < self.face_tolerance:
                        name = self.known_names[best_match_index]
                        confidence = (1 - min_distance) * 100
                    else:
                        name = "Unknown"
                        confidence = 0
                else:
                    name = "Unknown"
                    confidence = 0
                
                detected_names.append(name)
                
                # Draw face box and label
                top, right, bottom, left = face_location
                color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                
                cv2.rectangle(frame, (left, top), (right, bottom), color, 3)
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
                
                # Display name with confidence score
                if name != "Unknown":
                    label = f"{name} ({confidence:.1f}%)"
                else:
                    label = name
                cv2.putText(frame, label, (left + 6, bottom - 6), 
                           cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 2)

        except Exception as e:
            print(f"Error processing frame: {e}")
            return frame, []

        return frame, detected_names

    def run_detection(self) -> None:
        """Run the real-time face detection system using webcam."""
        # Try different camera indices
        camera = None
        for index in range(3):  # Try indices 0, 1, 2
            print(f"Trying camera index {index}...")
            camera = cv2.VideoCapture(index)
            if camera is not None and camera.isOpened():
                print(f"Successfully opened camera {index}")
                break
        
        if camera is None or not camera.isOpened():
            print("Error: Could not open any camera")
            return

        # Test frame reading
        ret, frame = camera.read()
        if not ret or frame is None:
            print("Error: Camera opened but cannot read frames")
            camera.release()
            return

        print(f"Camera resolution: {frame.shape}")
        print("Starting face detection... Press 'q' to quit")
        print(f"Known faces loaded: {len(self.known_names)}")

        try:
            while True:
                ret, frame = camera.read()
                if not ret or frame is None:
                    print("Error: Lost camera connection")
                    break

                processed_frame, detected_names = self.process_frame(frame)
                if processed_frame is not None:
                    cv2.imshow('Face Recognition', processed_frame)

                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("Quitting...")
                    break
                elif key == ord('r'):
                    print("Reloading known faces...")
                    self.load_known_faces()

        except Exception as e:
            print(f"Error during detection: {e}")

        finally:
            print("Cleaning up...")
            camera.release()
            cv2.destroyAllWindows()
            # Ensure windows are closed
            for i in range(4):
                cv2.waitKey(1)

    def reset_system(self) -> bool:
        """Reset the entire face recognition system."""
        try:
            self.known_encodings = []
            self.known_names = []
            if os.path.exists(self.encodings_file):
                os.remove(self.encodings_file)
            print("System reset - all faces have been deleted")
            return True
        except Exception as e:
            print(f"Error resetting system: {e}")
            return False

def setup_directories() -> None:
    """Create necessary directories if they don't exist."""
    Path("known_faces").mkdir(exist_ok=True)

def main():
    # Setup
    setup_directories()
    system = FaceRecognitionSystem()
    
    while True:
        print("\nFace Recognition System")
        print("1. Start Face Detection")
        print("2. Add New Person")
        print("3. Delete Person")
        print("4. List Registered People")
        print("5. Reset System")
        print("6. Exit")
        
        try:
            choice = input("Enter your choice (1-6): ").strip()
            
            if choice == "1":
                system.run_detection()
            
            elif choice == "2":
                name = input("Enter person's name: ").strip()
                if not name:
                    print("Error: Name cannot be empty")
                    continue
                    
                image_dir = input("Enter path to images directory: ").strip()
                if not os.path.exists(image_dir):
                    print("Error: Directory does not exist")
                    continue
                    
                image_paths = [
                    os.path.join(image_dir, f) 
                    for f in os.listdir(image_dir) 
                    if f.lower().endswith(('.png', '.jpg', '.jpeg'))
                ]
                
                if not image_paths:
                    print("Error: No images found in directory")
                    continue
                    
                if system.add_person(name, image_paths):
                    print(f"Successfully added {name} to the system")
                else:
                    print("Failed to add person")

            elif choice == "3":
                system.list_known_people()
                if system.known_names:
                    name = input("\nEnter name to delete: ").strip()
                    if name:
                        if system.delete_person(name):
                            print(f"Successfully deleted {name} from the system")
                    else:
                        print("Error: Name cannot be empty")

            elif choice == "4":
                system.list_known_people()
            
            elif choice == "5":
                confirm = input("Are you sure you want to reset the system? This will delete all faces. (y/n): ").strip().lower()
                if confirm == 'y':
                    if system.reset_system():
                        print("System successfully reset")
                    else:
                        print("Failed to reset system")
                else:
                    print("Reset cancelled")
            
            elif choice == "6":
                print("Exiting...")
                break
            
            else:
                print("Invalid choice. Please enter 1-6.")
                
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"An error occurred: {e}")
            continue

if __name__ == "__main__":
    main()