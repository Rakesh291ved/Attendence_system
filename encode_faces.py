import face_recognition
import os
import pickle

DATASET_PATH = "dataset"
ENCODINGS_FILE = "encodings.pickle"

# Delete old encodings file if it exists
if os.path.exists(ENCODINGS_FILE):
    os.remove(ENCODINGS_FILE)

known_encodings = []
known_names = []
known_roll_numbers = []

for filename in os.listdir(DATASET_PATH):
    if filename.endswith((".jpg", ".png", ".jpeg")):
        try:
            roll_no, name = filename.split("_", 1)
            name = os.path.splitext(name)[0]  # Remove file extension

            image_path = os.path.join(DATASET_PATH, filename)
            image = face_recognition.load_image_file(image_path)
            encodings = face_recognition.face_encodings(image)

            if encodings:
                known_encodings.append(encodings[0])  
                known_names.append(name)  
                known_roll_numbers.append(roll_no)
            else:
                print(f"Warning: No face found in {filename}")

        except ValueError:
            print(f"Skipping {filename}: Incorrect filename format. Use 'RollNo_Name.jpg'")

# Save updated encodings
with open(ENCODINGS_FILE, "wb") as f:
    pickle.dump({"encodings": known_encodings, "names": known_names, "roll_numbers": known_roll_numbers}, f)

print("Updated face encodings saved successfully.")
