import cv2
import pytesseract
import streamlit as st
import numpy as np
from PIL import Image

def main():
    st.title("Pytesseract Text Detection")
    img = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

    if img is not None:
        image = Image.open(img)
        st.image(image, caption='Uploaded Image.', use_column_width=True)
        st.write("")

        # Convert the image to grayscale
        gray_img = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2GRAY)

        # Draw green boxes around detected text
        final_img = np.array(image)
        h, w = gray_img.shape
        boxes = pytesseract.image_to_boxes(gray_img)
        for b in boxes.splitlines():
            b = b.split()
            top_left = (int(b[1]), h - int(b[2]))
            bottom_right = (int(b[3]), h - int(b[4]))
            text = b[0]
            final_img = cv2.rectangle(final_img, top_left, bottom_right, (0, 255, 0), 3)
            final_img = cv2.putText(final_img, text, (top_left[0], top_left[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

        # Display the image with green boxes
        st.image(final_img, caption='Result Image.', use_column_width=True)

        # Use pytesseract to extract words
        result = pytesseract.image_to_string(gray_img)
        res = [result]
        st.write("Detected Text:")
        st.write(res)
        # Download Button for Text
        download_button = st.download_button(
            label="Download Detected Text",
            data=result,
            file_name='detected_text.txt',
            mime='text/plain'
        )

if __name__ == "__main__":
    main()
