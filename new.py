import streamlit as st
import google.generativeai as genai
from PIL import Image, ImageDraw
import io
from gtts import gTTS
import pytesseract
import os
import cv2
import numpy as np

# Streamlit App Interface
st.title("AI For :blue[Visually Impaired Individuals]")
st.write("Upload an image to understand its content.")

# Loading API key from key.txt
with open("key.txt", "r") as f:
    GOOGLE_API_KEY = f.read().strip()

# Configuring Generative AI
genai.configure(api_key=GOOGLE_API_KEY)

pytesseract.pytesseract.tesseract_cmd = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

# Sidebar for navigation
st.sidebar.title(":blue[FEATURES]")
st.sidebar.write("Real-time scene understanding.")
st.sidebar.write("Text-to-speech conversion for reading visual content.")
st.sidebar.write("Object and obstacle detection for safe navigation.")
st.sidebar.write("Personalized assistance for daily tasks.")
feature = st.sidebar.selectbox(
    ":blue[Choose a Feature:]",
    ["Real-Time Scene Understanding", "Text-to-Speech Conversion", "Object and Obstacle Detection", "Personalized Assistance for Daily Tasks"]
)

# Defining a function to generate image descriptions for scene understanding
def generate_image_description(image_data):
    
    prompt = "Describe the content of this image in detail. Assume the image represents a real-world scene. Dont give me in different elements."
    try:
        # Converting the image bytes back into a PIL Image object
        image = Image.open(io.BytesIO(image_data))

        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content([prompt, image])
        
        return response.text
    except Exception as e:
        return f"Error generating description: {e}"

# Defining a function for text-to-speech conversion from extracted text
def extract_text_and_convert_to_speech(image_data):

    try:
        image = Image.open(io.BytesIO(image_data))
        extracted_text = pytesseract.image_to_string(image)

        if not extracted_text.strip():
            return "No text found in the image."

        tts = gTTS(extracted_text, lang='en')
        tts.save("extracted_text.mp3")
        return "extracted_text.mp3"
    except Exception as e:
        return f"Error during OCR or text-to-speech conversion: {e}"

# Defining a function for object and obstacle detection using OpenCV
def detect_objects_and_obstacles(image_data):
    """
    Detects objects and obstacles in the uploaded image using OpenCV and YOLO model.
    """
    try:
        net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
        layer_names = net.getLayerNames()
        output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

        image = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            return "Error: Unable to load image."

        blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)

        height, width, channels = image.shape
        class_ids = []
        confidences = []
        boxes = []

        for out in outs:
            for detection in out:
                if len(detection) >= 5:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]

                    if confidence > 0.5:
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)

                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)

                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

        for i in range(len(boxes)):
            if i in indexes.flatten():
                x, y, w, h = boxes[i]
                label = f"Object {class_ids[i]}"
                color = (0, 255, 0)
                cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
                cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        img_with_boxes_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        return img_with_boxes_pil
    except Exception as e:
        return f"Error detecting objects: {e}"

# Defining a function for personalized assistance based on image content
def provide_personalized_assistance(image_data):
    
    try:
        image = Image.open(io.BytesIO(image_data))
        extracted_text = pytesseract.image_to_string(image)
        
        prompt = f"Based on the image, provide guidance on how to use or understand the objects, labels, or content."
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content([prompt, image])
        
        return response.text or f"Extracted text: {extracted_text}"
    except Exception as e:
        return f"Error providing personalized assistance: {e}"


# File uploader for image upload
uploaded_file = st.file_uploader("Choose an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

    if feature == "Real-Time Scene Understanding":
        if st.button("Generate Description"):
            with st.spinner("Analyzing image..."):
                image_data = uploaded_file.read()
                description = generate_image_description(image_data)
                st.subheader("Image Description:")
                st.write(description)

    elif feature == "Text-to-Speech Conversion":
        # Convert image to speech if the feature is "Text-to-Speech Conversion"
        if st.button("Generate Speech from Image"):
            with st.spinner("Extracting text and converting to speech..."):
                try:
                    image_data = uploaded_file.read()
                    description = generate_image_description(image_data)
                    st.subheader("Image Description:")
                    st.write(description)
                    tts = gTTS(description, lang='en')
                
                    tts.save("description.mp3")
                
                    # Playing the audio
                    st.audio("description.mp3", format="audio/mp3")
                    os.remove("description.mp3")
                
                except Exception as e:
                    st.error(f"An error occurred: {e}")

    elif feature == "Object and Obstacle Detection":
        if st.button("Detect Objects and Obstacles"):
            with st.spinner("Detecting objects..."):
                image_data = uploaded_file.read()
                result = detect_objects_and_obstacles(image_data)
                if isinstance(result, str):
                    st.error(result)
                else:
                    st.image(result, caption="Objects and Obstacles Detected")

    elif feature == "Personalized Assistance for Daily Tasks":
        if st.button("Provide Personalized Assistance"):
            with st.spinner("Processing..."):
                image_data = uploaded_file.read()
                assistance = provide_personalized_assistance(image_data)
                st.subheader("Personalized Assistance:")
                st.write(assistance)

