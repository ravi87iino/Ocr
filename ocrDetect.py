import cv2 
import easyocr
import streamlit as st
import numpy as np
from PIL import Image

def main():
    st.title("EasyOCR Text Detection")
    img = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

    if img is not None:
        image = Image.open(img)
        st.image(image, caption='Uploaded Image.', use_column_width=True)
        st.write("")

        reader = easyocr.Reader(['en'])
        result = reader.readtext(np.array(image))

        final_img = np.array(image)
        for detection in result:
            top_left = tuple(map(int, detection[0][0]))
            bottom_right = tuple(map(int, detection[0][2]))
            text = detection[1]
            final_img = cv2.rectangle(final_img, top_left, bottom_right, (0, 255, 0), 3)
            final_img = cv2.putText(final_img, text, top_left, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

        st.image(final_img, caption='Result Image.', use_column_width=True)

        df = [detection[1] for detection in result]
        st.write("Detected Text:")
        st.write(df)

        # Download Button
        download_button = st.download_button(
            label="Download Detected Text",
            data='\n'.join(df),
            file_name='detected_text.txt',
            mime='text/plain'
        )

if _name_ == "_main_":
    main()
