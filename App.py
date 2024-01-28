from flask import Flask, render_template, request, redirect, url_for
import cv2
from imutils.object_detection import non_max_suppression
import numpy as np
import pytesseract
import pyttsx3
import pandas as pd
import Levenshtein as lev
import base64


app = Flask(__name__)

east_model_path = 'frozen_east_text_detection.pb'
net = cv2.dnn.readNet(east_model_path)

def textdetect(image):
    # ... (Include your existing text detection code here)
    orig = image.copy()  # Make a copy of the original image
    (H, W) = image.shape[:2]
    (newW, newH) = (320, 320)
    rW = W / float(newW)
    rH = H / float(newH)
    image = cv2.resize(image, (newW, newH))
    (H, W) = image.shape[:2]

    layerNames = ["feature_fusion/Conv_7/Sigmoid", "feature_fusion/concat_3"]
    blob = cv2.dnn.blobFromImage(image, 1.0, (W, H), (123.68, 116.78, 103.94), swapRB=True, crop=False)
    net.setInput(blob)
    (scores, geometry) = net.forward(layerNames)

    (numRows, numCols) = scores.shape[2:4]
    rects = []
    confidences = []

    for y in range(0, numRows):
        scoresData = scores[0, 0, y]
        xData0 = geometry[0, 0, y]
        xData1 = geometry[0, 1, y]
        xData2 = geometry[0, 2, y]
        xData3 = geometry[0, 3, y]
        anglesData = geometry[0, 4, y]

        for x in range(0, numCols):
            if scoresData[x] < 0.5:
                continue

            (offsetX, offsetY) = (x * 4.0, y * 4.0)
            angle = anglesData[x]
            cos = np.cos(angle)
            sin = np.sin(angle)
            h = xData0[x] + xData2[x]
            w = xData1[x] + xData3[x]
            endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
            endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
            startX = int(endX - w)
            startY = int(endY - h)

            rects.append((startX, startY, endX, endY))
            confidences.append(scoresData[x])

    boxes = non_max_suppression(np.array(rects), probs=confidences)

    recognized_text = []  # Store recognized text

    for (startX, startY, endX, endY) in boxes:
        startX = int(startX * rW)
        startY = int(startY * rH)
        endX = int(endX * rW)
        endY = int(endY * rH)
        boundary = 2
        # Ensure the text region is within bounds
        if startY - boundary >= 0 and endY + boundary <= orig.shape[0] and startX - boundary >= 0 and endX + boundary <= orig.shape[1]:
            text = orig[startY - boundary:endY + boundary, startX - boundary:endX + boundary]
            text = cv2.cvtColor(text.astype(np.uint8), cv2.COLOR_BGR2GRAY)
            textRecognized = pytesseract.image_to_string(text)
            recognized_text.append(textRecognized)  # Append recognized text to the list
            cv2.rectangle(orig, (startX, startY), (endX, endY), (0, 255, 0), 3)
            orig = cv2.putText(orig, textRecognized, (endX, endY + 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    return orig, recognized_text


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            return redirect(request.url)

        file = request.files['file']

        if file.filename == '':
            return redirect(request.url)

        if file:
            # Read the uploaded image
            image = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_UNCHANGED)
            image = cv2.resize(image, (640, 320), interpolation=cv2.INTER_AREA)

            # Process the image using textdetect function
            orig, recognized_text = textdetect(image)
            _, buffer = cv2.imencode('.png', orig)
            encoded_image = base64.b64encode(buffer).decode('utf-8')
            with open('recognized_text.txt', 'w') as file:
                for text in recognized_text:
                    file.write(text.replace('\n', ' '))
            with open('recognized_text.txt', 'r') as f:
                for line in f:
                    recognized_text.append(line.strip())
            medicine_data = pd.read_excel('medicine_data.xlsx')
            medicine_names = medicine_data['medicine_name'].tolist()
            matched_medicines = []
            for text in recognized_text:
                closest_match = None
                closest_match_distance = float('inf')

                for medicine_name in medicine_names:
                    distance = lev.distance(text, medicine_name)
                    if distance < closest_match_distance:
                        closest_match = medicine_name
                        closest_match_distance = distance

                if closest_match:
                    matched_medicine = {
                        'medicine_name': closest_match,
                        'use': medicine_data[medicine_data['medicine_name'] == closest_match]['use'].values[0]
                    }
                    matched_medicines.append(matched_medicine)


            # Store the matched medicines in a new file
            with open('matched_medicines.txt', 'w') as f:
                for medicine in matched_medicines:
                    f.write(f"{medicine['medicine_name']}: {medicine['use']}\n")

            text_speech = pyttsx3.init()

            for i in matched_medicines:
                answer=i

            text_speech.say(answer)
            text_speech.runAndWait()

            # Render the results template
            return render_template('results.html', medicine_name=answer['medicine_name'], use=answer['use'])




    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
