# idphoto
📸 Smart ID Photo Capture Softwar

An offline, AI-powered desktop application that automates the process of capturing, segmenting, and cropping standardized biometric ID photos right at home. Developed for the 2025–2026 Provincial Science and Engineering Fair for High School Students (Category: Systems Software) by students of Tu Nghia No. 1 High School, Quang Ngai .

💡 MotivationWith the rapid expansion of digital public services (such as chip-based Citizen IDs, e-passports, VNeID, and university entrance dossiers under Project 06), the demand for standardized digital ID photos is growing. However, traditional photo studios pose several barriers:  

+Time & Distance: Inconvenient for residents in rural/remote areas, students, and office workers. 
+ Accessibility: Difficult for the elderly and people with disabilities to travel and pose at traditional studios.  +
+ Cost Waste: Studios often require paid physical prints even when online portals only need digital files.
+ Privacy Risks: Uploading portraits to untrusted online editing tools risks facial data leaks (e.g., deepfakes or scams).
+ Smart ID Photo Capture Software solves these issues by turning any PC with a webcam into a private, offline "at-home photo studio".
+ 
✨ Key Features

+Real-Time AI Alignment: Uses YOLOv8n to detect the subject's position in real time and displays a bounding box with visual on-screen guidance so users can align their head and shoulders independently.

+AI Background Removal: Integrates the U²-Net neural network (via rembg) to separate the subject from complex backgrounds without needing a physical backdrop.

+Edge Smoothing: Applies morphological erosion (cv2.erode) and Gaussian blur (cv2.GaussianBlur) to the alpha channel to minimize jagged edges (aliasing) around hair and shoulders.

+Standardized Biometric Cropping: Automatically crops photos to standard 4x6 or 3x4 ratios (with the face centered and occupying 70%–80% of the frame) and replaces the background with standard Blue or White, adhering to ICAO standards and Ministry of Public Security regulations.

+Fast & One-Click UI: Built with Tkinter for simplicity; completes capture and processing in under 5 seconds.

+100% Offline & Standalone: Runs completely locally to guarantee absolute personal data privacy, packaged as a standalone Windows executable (.exe) requiring no Python installation.

⚙️ System WorkflowVideo Capture: Stream real-time frames from the webcam via OpenCV.
+Subject Detection: YOLOv8n scans and detects the user's face/body.

+Position Check: If misaligned, display on-screen guidance; if aligned, trigger capture (Blue or White background mode).

+Segmentation & Post-Processing: Remove the background using U²-Net (rembg) and smooth the edges (erode + GaussianBlur).

+Compositing & Cropping: Merge the subject onto the selected solid background and auto-crop to standard 4x6 proportions using YOLO coordinates.

+Output & Cleanup: Preview and save the final image (with error alerts if cropping fails), and automatically release camera resources upon exit.

🛠️ Tech Stack
+Language: Python

+Computer Vision: OpenCV (cv2), NumPy, Pillow

+AI / Deep Learning: YOLOv8n (ultralytics), U²-Net (rembg)

+GUI & Concurrency: Tkinter, threading   Distribution: Windows standalone executable (.exe)

📊 Experimental ResultsField testing was conducted with 50–100 volunteers at school, achieving over 85% compatibility with Vietnamese ID photo standards (suitable for student dossiers, student IDs, and standard administrative procedures).
<img width="977" height="509" alt="image" src="https://github.com/user-attachments/assets/a1cf7d4a-e8f4-4395-b55a-b20fa7e1099e" />

🚀 Future RoadmapGenerative AI Enhancement:
+Auto-adjust lighting, smooth skin, and automatically edit attire (e.g., adding collared shirts or suits).
+Cross-Platform Support: Develop mobile versions (Android/iOS) and a Web-app.
+Full Biometric Validation: Add automated checks for head tilt and eye openness before saving.
+Direct Printing: Integrate a printer management module to print directly on standard 10x15cm photo paper.
