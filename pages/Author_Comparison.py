import os
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from docx import Document
from PyPDF2 import PdfReader
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.util import ngrams
from collections import Counter
import scipy.stats as stats
import numpy as np
from textstat import flesch_kincaid_grade



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

    # calculate type-token ratio (TTR)
    words = word_tokenize(text.lower())
    unique_words = set(words)
    ttr = len(unique_words) / len(words)

    # calculate Flesch-Kincaid Grade Level
    fk_grade_level = flesch_kincaid_grade(text)

    df_word_counts = pd.DataFrame({'Sentence': sentences, 'Word Count': word_counts})

    return df_word_counts, ratio_with_comma, ratio_without_comma, most_common_ngrams, ttr, fk_grade_level


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
    ttr_list = []
    fk_grade_levels = []

    for file in files:
        file_path = os.path.join(author_folder, file)
        if file.endswith('.docx'):
            df_word_counts, ratio_with_comma, ratio_without_comma, most_common_ngrams, ttr, fk_grade_level = analyze_docx(
                file_path, ngram_length
            )
        elif file.endswith('.pdf'):
            df_word_counts, ratio_with_comma, ratio_without_comma, most_common_ngrams, ttr, fk_grade_level = analyze_pdf(
                file_path, ngram_length
            )
        else:
            continue

        df_word_counts_list.append(df_word_counts)
        ratios_with_comma.append(ratio_with_comma)
        ratios_without_comma.append(ratio_without_comma)
        most_common_ngrams_list.append(most_common_ngrams)
        ttr_list.append(ttr)
        fk_grade_levels.append(fk_grade_level)

    return df_word_counts_list, ratios_with_comma, ratios_without_comma, most_common_ngrams_list, ttr_list, fk_grade_levels


def main():
    st.title("Author Comparison")
    st.info("All Documents for each author are considered within this analysis.")
    st.sidebar.title("Select Authors")
    author_folder = st.sidebar.multiselect("Select an author folder", os.listdir('./docs'),
                                           default=os.listdir('./docs'))

    ngram_length = st.sidebar.number_input(
        'n-gram length', min_value=1, max_value=5, value=2, step=1, key='ngram_length'
    )

    fig = go.Figure()
    ratio_fig = go.Figure()
    words_fig = go.Figure()
    ttr_fig = go.Figure()
    fk_grade_fig = go.Figure()

    for author in author_folder:
        author_folder_path = os.path.join('./docs', author)
        df_word_counts_list, ratios_with_comma, ratios_without_comma, most_common_ngrams_list, ttr_list, fk_grade_levels = analyze_documents(
            author_folder_path, ngram_length
        )

        if len(df_word_counts_list) > 0:
            df_word_counts_all = pd.concat(df_word_counts_list)
            fig.add_trace(go.Box(y=df_word_counts_all['Word Count'], name=author))
            ratio_fig.add_trace(go.Bar(x=[author], y=[ratios_with_comma[0]], name='With Comma'))
            ratio_fig.add_trace(go.Bar(x=[author], y=[ratios_without_comma[0]], name='Without Comma'))

        if len(most_common_ngrams_list) > 0:
            words_fig.add_trace(go.Bar(x=[word[0] for word in most_common_ngrams_list[0]],
                                       y=[word[1] for word in most_common_ngrams_list[0]], name=author))

        if len(ttr_list) > 0:
            ttr_fig.add_trace(go.Indicator(
                mode="gauge+number",
                value=ttr_list[0],
                title={'text': '{}'.format(author), 'font': {'color': 'white'}},
                number={'font': {'color': 'white'}},
                gauge={'axis': {'range': [0, 1]},
                       'bar': {'color': '#ffc900'},
                       'steps': [
                           {'range': [0, 0.3], 'color': '#6581BF'},
                           {'range': [0.3, 0.7], 'color': '#2F57AB'},
                           {'range': [0.7, 1], 'color': '#0B389D'}],
                       },
                domain={'x': [(author_folder.index(author) / len(author_folder))+0.03,
                              ((author_folder.index(author) + 1) / len(author_folder))-0.03],
                        'y': [0.05, 0.05]}
            ))

        if len(fk_grade_levels) > 0:
            fk_grade_fig.add_trace(go.Indicator(
                mode="gauge+number",
                value=np.average(fk_grade_levels),
                title={'text': '{}'.format(author), 'font': {'color': 'white'}},
                number={'font': {'color': 'white'}},
                gauge={'axis': {'range': [0, 100]},
                       'bar': {'color': '#ffc900'},
                       'steps': [
                           {'range': [0, 33], 'color': '#6581BF'},
                           {'range': [34, 66], 'color': '#2F57AB'},
                           {'range': [67, 100], 'color': '#0B389D'}],
                       },
                domain={'x': [(author_folder.index(author) / len(author_folder))+ 0.03,
                              ((author_folder.index(author) + 1) / len(author_folder))- 0.03],
                        'y': [0.05, 0.05]}
            ))


    fig.update_layout(title='Comparison - Word Counts per Sentence and Author over all Docs',
                      xaxis_title='Author',
                      yaxis_title='Word Count per Sentence')
    ratio_fig.update_layout(title='Comparison - Ratio of Sentences with and without Commas',
                            xaxis_title='Author',
                            yaxis_title='Ratio')
    words_fig.update_layout(title='Most Common Words',
                            xaxis_title='Word',
                            yaxis_title='Frequency')
    ttr_fig.update_layout(title='Vocabulary Richness - Type-Token Ratio (TTR)',
                          font={'color': "black"},
                          paper_bgcolor='rgba(0,0,0,0)',
                          plot_bgcolor='rgba(0,0,0,0)',
                          showlegend=False)
    ttr_fig.update_xaxes(showticklabels=False)
    ttr_fig.update_yaxes(showticklabels=False)
    fk_grade_fig.update_layout(title='Flesch-Kincaid Grade Level',
                               font={'color': "black"},
                               paper_bgcolor='rgba(0,0,0,0)',
                               plot_bgcolor='rgba(0,0,0,0)',
                               showlegend=False)
    fk_grade_fig.update_xaxes(showticklabels=False)
    fk_grade_fig.update_yaxes(showticklabels=False)

    st.subheader("Word Count per Sentence")
    st.plotly_chart(fig)

    st.subheader("Comma Ratio")
    st.plotly_chart(ratio_fig)

    st.subheader("Vocabulary Richness")
    with st.expander("ℹ️  What is TTR?"):
        st.info(
            "The Type-Token Ratio (TTR) is a measure of vocabulary richness that calculates the ratio of unique words (types) to the total number of words (tokens) in a text. A higher TTR indicates a larger variety of words used, suggesting a more diverse and potentially sophisticated vocabulary."
        )
    st.plotly_chart(ttr_fig)

    st.subheader("Flesch-Kincaid Grade Level")
    with st.expander("ℹ️  What is the Flesch-Kincaid Grade Level?"):
        st.info(
            "The Flesch-Kincaid Grade Level is a readability test that indicates the approximate grade level required to understand a text. A lower grade level indicates easier readability, while a higher grade level suggests more advanced vocabulary and sentence complexity."
        )
    st.plotly_chart(fk_grade_fig)

    st.subheader("N-Grams")
    st.plotly_chart(words_fig)


if __name__ == '__main__':
    main()
