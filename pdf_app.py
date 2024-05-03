import streamlit as st
import fitz
import cv2
import pytesseract
from transformers import pipeline

def extract_text_from_pdf(pdf_path):
    with fitz.open(pdf_path) as doc:
        text = ""
        for page in doc:
            images = page.get_images()
            if images:
                for image in images:
                    xref = image[0]
                    image_data = doc.extract_image(xref)
                    img = cv2.imdecode(image_data["image"], cv2.IMREAD_COLOR)
                    text += pytesseract.image_to_string(img)
            else:
                text += page.get_text()
    return text

def answer_question(question, context):
    model_checkpoint = "akshit-g/distilbert-base-cased"
    question_answerer = pipeline("question-answering", model=model_checkpoint)
    result = question_answerer(question=question, context=context)
    return result['answer'], result['score'] * 100  # Convert score to percentage

def main():
    if 'enter_pressed' not in st.session_state:
        st.session_state.enter_pressed = False

    st.title("PDF Question Answering")
    st.write("This is a simple tool that extracts text from uploaded PDF files and answers questions based on the content using a pretrained DistilBERT model.")

    pdf_file = st.file_uploader("Upload a PDF file", type=["pdf"])

    if pdf_file is not None:
        context = extract_text_from_pdf(pdf_file)

        question = st.text_input("Enter your question", key="question_input")
        ask_question = st.button("Ask Question")

        if st.session_state.enter_pressed or ask_question or question:  # Check if Enter key is pressed or Ask Question button is clicked
            if question:
                answer, confidence = answer_question(question, context)
                st.write(f"Answer: {answer}")
                st.write(f"Confidence: {confidence:.2f}%")
            else:
                st.warning("Please enter a question.")

        if st.session_state.enter_pressed:  # Reset Enter key status
            st.session_state.enter_pressed = False

        st.text("")  # Add space between components
        st.write("*Press 'Ask Question' to get the answer.*")

if __name__ == "__main__":
    main()
