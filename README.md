# Medicine-Drug-Name-Detection-Object-recognition-using-Deep-learning-based-OCR-Systems
This is used to read the name of the drug on the medicine label using EAST and Tesseract.

This code is a Flask application for performing text detection and recognition on images uploaded through a web interface. Here's a breakdown of its main components:

1. **Flask Setup:** The code sets up a Flask web application.

2.**Text Detection:** The textdetect function defines the process for detecting text regions within an image. It utilizes the EAST (Efficient and Accurate Scene Text Detector) text detection model, which is based on a deep learning architecture.

3.**Index Route**: This route handles both GET and POST requests. For GET requests, it renders the index.html template, which contains a form for uploading images. For POST requests, it processes the uploaded image using the textdetect function and performs text recognition on the detected regions.

4.**Text Recognition and Matching:** After detecting text regions, the code uses Tesseract OCR (Optical Character Recognition) to recognize text within those regions. It then matches the recognized text with a list of medicine names stored in an Excel file (medicine_data.xlsx) using the Levenshtein distance algorithm to find the closest match.

5.**Text-to-Speech**: Once the matching medicine is found, the code utilizes the pyttsx3 library to convert the matched medicine name and its use into speech.

6.**Results Rendering:** Finally, the code renders the results.html template, displaying the matched medicine name and its use.

To ensure this code works properly, make sure to have the following dependencies installed:
**Flask
OpenCV (cv2)
imutils
numpy
pytesseract
pyttsx3
pandas
Levenshtein
base64**

You'll also need the pre-trained EAST text detection model **(frozen_east_text_detection.pb)** and an Excel file **(medicine_data.xlsx)** containing medicine names and their uses.

To run this application locally, execute the script and access the web interface through a browser. Make sure to include the necessary HTML templates (index.html and results.html) in a folder named templates within the same directory as the script.

The following directory should be used while using index,html and results.html.

**flask_text_detection/
│
├── app.py
├── templates/
│   ├── index.html
│   └── results.html
├── frozen_east_text_detection.pb
├── medicine_data.xlsx
└── static/
    └── css/
        └── style.css**
Note: frozen_east_text_detection.pb is free to download.It plays a major role in the whole project.

The working is as follows:
**Step1:** Initially upload any of the image as follows
![OCR INOUT](https://github.com/Jas-04/Medicine-Drug-Name-Detection-Object-recognition-using-Deep-learning-based-OCR-Systems/assets/133671306/5377030c-bb8b-450b-9b14-e5cff3bfe1d5)
**Step2:** The processing and the output is as follows
![image](https://github.com/Jas-04/Medicine-Drug-Name-Detection-Object-recognition-using-Deep-learning-based-OCR-Systems/assets/133671306/809b7f55-e877-40c2-9b67-21fd225f7499)

The sample **medicine_data.xlsx** is in the main branch refer to create your own.

Note:Welcome contributions from others by inviting them to submit pull requests with proposed changes or fixes.


