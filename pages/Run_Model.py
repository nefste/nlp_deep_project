import streamlit as st
import docx2txt
import PyPDF2
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from streamlit_lottie import st_lottie

st.set_page_config(
    page_title="NLP Fingerprint Detection",
    page_icon="🐝",
    layout="wide",
)

st.title("Run Model to detect Author Change")
st.write("Please upload a Document (PDF or Word) in the left Sidebar")
st_lottie("https://assets2.lottiefiles.com/packages/lf20_qdmak08f.json", height=300)

with st.spinner('Loading model...'):
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    model = AutoModelForSequenceClassification.from_pretrained("bamertl/nlp_deep_project")
    st.success('Model loaded successfully!')
st.spinner()


def predict_author_changes(paragraphs):
    pairs = []
    for i in range(len(paragraphs) - 1):
        pairs.append({"text1": paragraphs[i], "text2": paragraphs[i + 1]})

    author_changes = []
    for pair in pairs:
        inputs = tokenizer(pair["text1"], pair["text2"], is_split_into_words=True,
                           truncation=True, return_tensors="pt")
        with torch.no_grad():
            logits = model(**inputs).logits
            predicted_class_id = logits.argmax().item()
        author_changes.append(predicted_class_id)
    return author_changes

def main():

    uploaded_file = st.sidebar.file_uploader("Upload Document", type=["docx", "pdf"])

    if uploaded_file is not None:
        if uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":  # Check if file is Word document
            file_contents = docx2txt.process(uploaded_file)
        elif uploaded_file.type == "application/pdf":  # Check if file is PDF
            file_reader = PyPDF2.PdfReader(uploaded_file)
            file_contents = ""
            for page in range(len(file_reader.pages)):
                file_contents += file_reader.pages[page].extract_text()
        else:
            st.error("Invalid file format. Please upload a Word document (docx) or a PDF.")
            return


        arr = file_contents.split("\n")
        valid_paragraphs = []
        for element in arr:
            if len(element) > 100:
                valid_paragraphs.append(element)

        st.success(f"File uploaded, {len(valid_paragraphs)} Paragraphs detected")
        st.warning("For PDF Docs: sometimes the paragraph detection does not work steady, to debug copy past the text into a doxc file and upload again in the sidebar left.")
            
        if st.button("Run NLP Model"):
            # Your NLP model code goes here
            st.spinner("NLP model is running...")
            # Run your NLP model on the `file_contents` variable
            with st.spinner('Predicting with model...'):
                changes = predict_author_changes(valid_paragraphs)
                changes.append(0)
            st.spinner()
            for i,paragraph in enumerate(valid_paragraphs):

                with st.expander(f"Paragraph {i+1} | {paragraph[:80]}..."):
                    st.write(paragraph)
                author_changed = bool(changes[i])
                if author_changed:
                    st.success(f"Probability of Author Change to next Paragraph {i+1} = {author_changed}")
                else:
                    st.info(f"Probability of Author Change to next Paragraph {i+1} = {author_changed}")


if __name__ == '__main__':
    main()
