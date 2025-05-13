# Attendence_system
Attendence_system
🎓 Face Recognition Attendance System
This is a Python-based real-time face recognition attendance system that automatically marks student attendance using webcam input. It identifies faces using pre-encoded facial data and logs present and absent students into a CSV file.

Built using:

OpenCV

face_recognition

pandas

pickle

📸 Features
Real-time face detection via webcam

Automatically marks present students in attendance.csv

Logs timestamped entries to avoid duplicate daily entries

Marks absent students at the end of the session

Visualizes recognized faces with bounding boxes and labels

🗂️ Project Structure
File	Description
encodings.pickle	Stores known face encodings with names and roll numbers
attendance.csv	Stores timestamped attendance records
Attendance_system.py	Main attendance script

📦 Requirements
Install required packages with:

bash
Copy
Edit
pip install opencv-python face_recognition numpy pandas
You will also need dlib installed (required by face_recognition). On some systems, installing dlib might require CMake and Visual Studio Build Tools (on Windows).

📁 Face Encoding Format
The encodings.pickle file should be a dictionary with keys:

"encodings": List of face encodings

"names": List of names corresponding to encodings

"roll_numbers": List of roll numbers corresponding to encodings

You must create this file in advance using a script to encode known student faces.

Example structure:
{
"encodings": [ ... ],
"names": ["Alice", "Bob", ...],
"roll_numbers": ["101", "102", ...]
}

🚀 How to Run
Place encodings.pickle in the project folder.

Run the attendance system:

bash
Copy
Edit
python Attendance_system.py
The webcam will open. Recognized students will be marked present.

Press q to quit or wait for the timeout (10 seconds default).

Absent students will be marked after session ends.

📝 Attendance Format
The attendance.csv will store entries as:

Roll No	Name	Time	Status
101	Alice	2025-05-13 09:15:01	
102	Bob	2025-05-13 09:16:42	
103	Carol	2025-05-13 00:00:00	Absent

Present entries have timestamps; absentees are marked at the end of the session with "Absent".

✅ Notes
Tolerance for matching faces can be adjusted via compare_faces tolerance parameter.

Ensure good lighting and camera quality for accurate detection.

Can be extended to web apps using Flask or Django.
