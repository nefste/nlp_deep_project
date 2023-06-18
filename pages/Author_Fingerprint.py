import os
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from docx import Document
from PyPDF2 import PdfReader
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize
from nltk.util import ngrams
from collections import Counter
import numpy as np


st.set_page_config(
     page_title="NLP Fingerprint Detection",
     page_icon="🐝",
     layout="wide",
)

def analyze_text(text, ngram_length):
    # text into sentences
    sentences = sent_tokenize(text)
    # word counts per sentence
    word_counts = [len(sentence.split()) for sentence in sentences]

    # calculate ratio
    sentences_with_comma = [sentence for sentence in sentences if ',' in sentence]
    sentences_without_comma = [sentence for sentence in sentences if ',' not in sentence]
    ratio_with_comma = len(sentences_with_comma) / len(sentences)
    ratio_without_comma = len(sentences_without_comma) / len(sentences)

   # most common N-grams
    ngrams_counter = Counter(ngrams(text.split(), ngram_length))
    most_common_ngrams = ngrams_counter.most_common(10)

    df_word_counts = pd.DataFrame({'Sentence': sentences, 'Word Count': word_counts})

    return df_word_counts, ratio_with_comma, ratio_without_comma, most_common_ngrams


def analyze_docx(docx_file, ngram_length):
    document = Document(docx_file)
    text = " ".join([paragraph.text for paragraph in document.paragraphs])
    return analyze_text(text, ngram_length)


def analyze_pdf(pdf_file, ngram_length):
    with open(pdf_file, 'rb') as file:
        reader = PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
    return analyze_text(text, ngram_length)


def analyze_documents(author_folder, ngram_length):
    files = os.listdir(author_folder)
    df_word_counts_list = []
    ratios_with_comma = []
    ratios_without_comma = []
    most_common_ngrams_list = []

    for file in files:
        file_path = os.path.join(author_folder, file)
        if file.endswith('.docx'):
            df_word_counts, ratio_with_comma, ratio_without_comma, most_common_ngrams = analyze_docx(file_path, ngram_length)
        elif file.endswith('.pdf'):
            df_word_counts, ratio_with_comma, ratio_without_comma, most_common_ngrams = analyze_pdf(file_path, ngram_length)
        else:
            continue  # Skip files with unsupported extensions

        df_word_counts_list.append(df_word_counts)
        ratios_with_comma.append(ratio_with_comma)
        ratios_without_comma.append(ratio_without_comma)
        most_common_ngrams_list.append(most_common_ngrams)

    return df_word_counts_list, ratios_with_comma, ratios_without_comma, most_common_ngrams_list


def show_word_counts(df_word_counts_list):
    for i, df_word_counts in enumerate(df_word_counts_list):
        word_counts_sorted = df_word_counts.sort_values(by='Word Count', ascending=False)
        fig = go.Figure(data=[go.Bar(x=word_counts_sorted['Sentence'], y=word_counts_sorted['Word Count'])])
        fig.update_layout(title=f"Word Counts per Sentence (Document {i+1})", xaxis_title="Sentence", yaxis_title="Word Count")
        st.plotly_chart(fig)

def plot_word_counts_boxplot(df_word_counts_list, selected_docs):
    fig = go.Figure()
    mean_values = []
    confidence_intervals = []

    for i, df_word_counts in enumerate(df_word_counts_list):
        doc_label = f"Document {i+1}"
        if doc_label in selected_docs or "All Documents" in selected_docs:
            fig.add_trace(go.Box(y=df_word_counts['Word Count'], name=doc_label))
            mean = df_word_counts['Word Count'].mean()
            std = df_word_counts['Word Count'].std()
            n = len(df_word_counts)
            z = 1.96  # 95% confidence interval, see z-values
            confidence_interval = z * (std / np.sqrt(n))
            mean_values.append(mean)
            confidence_intervals.append(confidence_interval)

    fig.update_layout(
        title=f"Word Counts per Sentence (Mean: {np.mean(mean_values):.2f} ± {np.mean(confidence_intervals):.2f} [95% CI])",
        xaxis_title="Documents",
        yaxis_title="Sum Word Count per Sentence"
    )

    # Add horizontal line with the mean
    fig.add_shape(
        type='line',
        x0=-0.5,
        x1=len(mean_values) - 0.5,
        y0=np.mean(mean_values),
        y1=np.mean(mean_values),
        line=dict(color='red', width=2)
    )

    st.plotly_chart(fig)



import scipy.stats as stats

def plot_sentence_ratio_bar_chart(ratios_with_comma, ratios_without_comma, selected_docs):
    fig_bar = go.Figure()
    mean_ratio_with_comma = sum(ratios_with_comma) / len(ratios_with_comma)

    fig_bar.add_trace(go.Bar(x=[f'Document {i+1}' for i in range(len(ratios_with_comma))], y=ratios_with_comma,
                             name='With Comma'))
    fig_bar.add_trace(go.Bar(x=[f'Document {i+1}' for i in range(len(ratios_without_comma))], y=ratios_without_comma,
                             name='Without Comma'))

    fig_bar.add_shape(type='line', x0=-0.5, x1=len(ratios_with_comma)-0.5, y0=mean_ratio_with_comma, y1=mean_ratio_with_comma,
                      line=dict(color='red', width=1))

    fig_bar.update_layout(title=f"Ratio of Sentences with Comma vs. without Comma (Mean Ratio: {mean_ratio_with_comma:.2f})",
                          xaxis_title="Documents",
                          yaxis_title="Ratio",
                          barmode='stack')

    fig_box = px.box(y=ratios_with_comma + ratios_without_comma,
                     x=['With Comma'] * len(ratios_with_comma) + ['Without Comma'] * len(ratios_without_comma),
                     labels={'x': 'Sentence Type', 'y': 'Ratio'},
                     title='Overall Sentence Ratio',
                     width=200)

    st.subheader('Overall Sentence Ratio Boxplot')
    col1, col2 = st.columns([3, 1])
    col1.plotly_chart(fig_bar)
    col2.plotly_chart(fig_box)


    mean_ratio_with_comma = round(sum(ratios_with_comma) / len(ratios_with_comma),2)
    mean_ratio_without_comma = round(sum(ratios_without_comma) / len(ratios_without_comma),2)

    confidence_interval_with_comma = round(stats.t.interval(0.95, len(ratios_with_comma)-1, loc=mean_ratio_with_comma, scale=stats.sem(ratios_with_comma))[1],2)
    confidence_interval_without_comma = round(stats.t.interval(0.95, len(ratios_without_comma)-1, loc=mean_ratio_without_comma, scale=stats.sem(ratios_without_comma))[0],2)

    col1.metric(label="With Comma", value=mean_ratio_with_comma, delta=round(confidence_interval_with_comma-mean_ratio_with_comma,2))
    col2.metric(label="Without Comma", value=mean_ratio_without_comma, delta=round(confidence_interval_without_comma-mean_ratio_without_comma,2))




def display_ngrams(most_common_ngrams_list, selected_docs, ngram_length):
    ngrams_data = {}
    colors = px.colors.qualitative.Plotly

    for i, most_common_ngrams in enumerate(most_common_ngrams_list):
        doc_label = f"Document {i+1}"
        if doc_label in selected_docs or "All Documents" in selected_docs:
            for ngram, frequency in most_common_ngrams:
                if len(ngram) == ngram_length:
                    if ngram in ngrams_data:
                        if "All Documents" in selected_docs:
                            ngrams_data[ngram]["All Documents"] += frequency
                        else:
                            if doc_label in ngrams_data[ngram]:
                                ngrams_data[ngram][doc_label] += frequency
                            else:
                                ngrams_data[ngram][doc_label] = frequency
                    else:
                        if "All Documents" in selected_docs:
                            ngrams_data[ngram] = {"All Documents": frequency}
                        else:
                            ngrams_data[ngram] = {doc_label: frequency}

    df_ngrams = pd.DataFrame(ngrams_data).transpose()
    df_ngrams = df_ngrams.fillna(0)


    st.subheader("Top N-grams")

    selected_ngrams = st.multiselect("Select N-grams", list(df_ngrams.index), default=list(df_ngrams.index[:10]))

    fig = go.Figure()

    for doc_label in selected_docs:
        if doc_label in df_ngrams.columns:
            fig.add_trace(go.Bar(x=selected_ngrams, y=df_ngrams.loc[selected_ngrams, doc_label], name=doc_label))

    fig.update_layout(
        title="Word Frequency of N-grams",
        xaxis_title="N-gram",
        yaxis_title="Frequency",
        barmode='stack',
        colorway=colors
    )

    st.plotly_chart(fig)
    with st.expander("Dataset N-Grams"):
        df_ngrams




# Main Streamlit App
def main():
    st.title("Authors Fingerprint")
    st.sidebar.title("Select Author Folder")
    author_folder = st.sidebar.selectbox("Select an author folder", os.listdir('./docs'))

    if author_folder:
        author_folder_path = os.path.join('./docs', author_folder)
        ngram_length = st.sidebar.number_input(
            'n-gram length', min_value=1, max_value=5, value=2, step=1, key='ngram_length'
        )
        df_word_counts_list, ratios_with_comma, ratios_without_comma, most_common_ngrams_list = analyze_documents(
            author_folder_path, ngram_length
        )
        selected_docs = st.multiselect(
            "Select Documents", [f"Document {i+1}" for i in range(len(df_word_counts_list))] ,
            default=[f"Document {i+1}" for i in range(len(df_word_counts_list))]
        )
        if selected_docs:
            plot_word_counts_boxplot(df_word_counts_list, selected_docs)
            with st.expander("See word counts per document:"):
                show_word_counts(df_word_counts_list)
            plot_sentence_ratio_bar_chart(ratios_with_comma, ratios_without_comma, selected_docs)
            display_ngrams(most_common_ngrams_list, selected_docs, ngram_length)


if __name__ == '__main__':
    main()
