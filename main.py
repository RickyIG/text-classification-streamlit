import streamlit as st

# Library untuk data manipulation.
import numpy as np
import scipy as sp
from scipy.spatial import distance
import pandas as pd
import spacy

# Library untuk data visualization.
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from prettytable import PrettyTable

# Library untuk data modeling.
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.metrics import f1_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
# from tensorflow.keras import utils, Input, callbacks
# from tensorflow.keras.layers import Conv1D, Dense, Embedding, Flatten, MaxPool1D, Dropout

import os
import re
import math
import time
import shutil
import pickle
from spacy.cli import download

download("en_core_web_sm")
# Versi Library.
print("NumPy version:", np.__version__)
print("Pandas version:", pd.__version__)
print("Matplotlib version:", matplotlib.__version__)
print("Seaborn version:", sns.__version__)
print("Scikit-learn version:", sklearn.__version__)
print("TensorFlow version:", tf.__version__)

# Konfigurasi NumPy.
np.set_printoptions(precision=4, linewidth=130)

# Konfigurasi  Seaborn.
sns.set_style("whitegrid") 
sns.set_palette("deep")  
sns.set_context("paper", font_scale=1.25) 

# Konfigurasi Tensorflow.
tf.random.set_seed(100)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"


def is_empty(element):
    """
    Fungsi untuk mengecek apakah input element sedang kosong.
    """
    if (isinstance(element, int) or isinstance(element, float)) and element == 0:
        return False
    elif isinstance(element, str) and len(element.strip()) == 0:
        return True
    elif isinstance(element, bool):
        return False
    else:
        return False if element else True

def get_dup_doc_ids(df):
    """
    Fungsi untuk mengelist semua document id yang duplikat.
    """
    return [idx for idx, val in df["DocId"].value_counts().items() if val > 1]

def get_corr_mat(df, target):
    """
    Fungsi yang mengembalikan:
        1. Matriks korelasi diurutkan berdasarkan nilai target secara menurun.
        2. Varians variabel acak direset ke nol.
    """
    corr_mat = df.corr(numeric_only=True)
    for col_name in corr_mat.columns.tolist():
        # Varians direset ke nol.
        corr_mat.loc[col_name, col_name] = 0

    return corr_mat.drop(target).sort_values(by=target, ascending=False)

# Ubah daftar item unik menjadi dict dengan item sebagai kunci dan indeks sebagai nilai.
to_dict = lambda frm_list: {item: frm_list.index(item) for item in frm_list}

def show_n_vals(row, n=5):
    """
    Berfungsi untuk mengecilkan `baris` dengan menampilkan elipsis di antaranya
    nilai `n` pertama dan nilai `n` terakhir dipisahkan dengan koma.
    """
    if row is None:
        return "[]"

    lhs = []
    rhs = []
    for idx, val in enumerate(row):
        if idx < n:
            lhs.append(str(val))
        elif idx >= (len(row) - n):
            rhs.append(str(val))

    lhs = ", ".join(lhs)
    delim = ", ..." if len(row) > 2 * n else ""
    rhs = f", {', '.join(rhs)}" if len(rhs) else ""

    return f"[{lhs + delim + rhs}]"


# Lambda function to reduce input `value` by `percent` percentage.
reduce_by = lambda value, percent: value - (percent * value / 100)


def cosine_sim(gvec_index, word1, word2):
    """
    Berfungsi untuk menghitung kesamaan kosinus antara
    dua kata masukan berdasarkan vektor GloVe-nya.
    """
    vec1 = gvec_index.get(word1.lower())
    print(f"GloVe vector for `{word1}`: {show_n_vals(vec1, 3)}")

    vec2 = gvec_index.get(word2.lower())
    print(f"GloVe vector for `{word2}`: {show_n_vals(vec2, 3)}")

    c_dist = distance.cosine(vec1, vec2)
    print(f"Cosine similarity between '{word1}' and '{word2}': {round(c_dist, 4)}")

cls_map = {"Business": 0, "Entertainment": 1, "Politics": 2, "Sport": 3, "Tech": 4}
CLASS_COUNT = 5  # Jumlah label class

nlp = spacy.load("en_core_web_sm")

patterns = [
    {"patterns": [[{"TEXT": "'ve"}]], "attrs": {"LEMMA": "have"}},
    {"patterns": [[{"TEXT": "'m"}]], "attrs": {"LEMMA": "am"}},  # No use its length < 3
    {"patterns": [[{"TEXT": "'re"}]], "attrs": {"LEMMA": "are"}},
]


def process_language(Id, text, rm_stp=True, rm_pun=True):
    """
    Function to process language by
        1. Removing stop-words and punctuation marks.
        3. Lemmatizing tokens.
        3. Chunking entities.

    rm_stp :- Flag to remove stop-words.
    rm_pun :- Flag to remove punctuation marks.
    """
    doc = nlp(str(text))

    # Special cases of decontraction.
    atrib_rlr = nlp.get_pipe("attribute_ruler")
    atrib_rlr.add_patterns(patterns)

    # Preprocessing based on chunking.
    with doc.retokenize() as retokenizer:
        for ent in doc.ents:
            if ent.label_ in ["PERSON", "GPE", "ORG", "PRODUCT"]:
                new_val = "_".join(ent.text.split())
                retokenizer.merge(doc[ent.start : ent.end], attrs={"LEMMA": new_val})

    # Lemmatization, stop-words and punctuation marks removal.
    lemmas = []
    for tkn in doc:
        if (rm_stp and tkn.is_stop) or (rm_pun and tkn.is_punct):
            continue

        lemmas.append(tkn.lemma_.strip().lower())

    lmtzd_txt = " ".join(lemmas)

    # Capture POS count.
    pos_count = {"DocId": Id} if Id else {}
    for item in sorted(doc.count_by(spacy.attrs.POS).items(), key=lambda i: i[0]):
        pos_count[doc.vocab[item[0]].text] = item[1]

    return lmtzd_txt, pos_count


st.set_page_config(layout="wide")


st.title("*Text Classification*: Model *Deep Learning* yang Ditingkatkan untuk Klasifikasi Text dari BBC News Dataset")
st.write("")
st.write("")

col1, mid, col2 = st.columns([1,1,38])
with col1:
    st.image('./assets/img/circle.png', width=60)
with col2:
    st.markdown("**Ricky Indra Gunawan**<br>[linkedin.com/in/rickyindrag/](https://www.linkedin.com/in/rickyindrag/)", unsafe_allow_html=True)
st.markdown("---")

st.write("""Dalam domain analisis teks, klasifikasi dokumen merupakan tantangan yang sering dihadapi, khususnya dalam mengelola volume data yang besar. Proyek ini menangani klasifikasi teks menggunakan dataset BBC yang terdiri dari ribuan file teks yang telah tersegresi ke dalam kelas-kelas yang berbeda. Masing-masing kelas mewakili kategori berita tertentu. Klasifikasi yang akurat dan efisien dari dokumen-dokumen ini menjadi penting untuk pengelolaan informasi yang lebih baik dan akses yang cepat terhadap konten yang relevan. Klasifikasi ini harus mengatasi berbagai tantangan seperti pengelolaan teks dalam jumlah besar, dan pengambilan fitur yang efektif untuk meningkatkan performa prediksi.""")

st.subheader("Klasifikasi Text")
st.write("Masukkan teks (disarankan untuk memasukkan teks minimal 4 kalimat dalam bahasa Inggris)")
st.write("Contoh (Tech) :")
st.write("""Microsoft seeking spyware trojan
Microsoft is investigating a trojan program that attempts to switch off the firm's anti-spyware software.
The spyware tool was only released by Microsoft in the last few weeks and has been downloaded by six million people. Stephen Toulouse, a security manager at Microsoft, said the malicious program was called Bankash-A Trojan and was being sent as an e-mail attachment. Microsoft said it did not believe the program was widespread and recommended users to use an anti-virus program. The program attempts to disable or delete Microsoft's anti-spyware tool and suppress warning messages given to users.
It may also try to steal online banking passwords or other personal information by tracking users' keystrokes.
Microsoft said in a statement it is investigating what it called a criminal attack on its software. Earlier this week, Microsoft said it would buy anti-virus software maker Sybari Software to improve its security in its Windows and e-mail software. Microsoft has said it plans to offer its own paid-for anti-virus software but it has not yet set a date for its release. The anti-spyware program being targeted is currently only in beta form and aims to help users find and remove spyware - programs which monitor internet use, causes advert pop-ups and slow a PC's performance.
""")
st.write("Contoh (Sport) :")
st.write("""Claxton hunting first major medal
British hurdler Sarah Claxton is confident she can win her first major medal at next month's European Indoor Championships in Madrid.
The 25-year-old has already smashed the British record over 60m hurdles twice this season, setting a new mark of 7.96 seconds to win the AAAs title. "I am quite confident," said Claxton. "But I take each race as it comes. "As long as I keep up my training but not do too much I think there is a chance of a medal." Claxton has won the national 60m hurdles title for the past three years but has struggled to translate her domestic success to the international stage. Now, the Scotland-born athlete owns the equal fifth-fastest time in the world this year. And at last week's Birmingham Grand Prix, Claxton left European medal favourite Russian Irina Shevchenko trailing in sixth spot.
For the first time, Claxton has only been preparing for a campaign over the hurdles - which could explain her leap in form. In previous seasons, the 25-year-old also contested the long jump but since moving from Colchester to London she has re-focused her attentions. Claxton will see if her new training regime pays dividends at the European Indoors which take place on 5-6 March.
""")


teks_baru = st.text_input('Masukkan teks yang ingin diklasifikasikan')


def predicting(doc_text):
    try:
        lmtzd_txts = []
        pos_counts = []
        # Melakukan decontraction, lemmatization dan NER pada setiap barisnya.
        lmtzd_txt, pos_count = process_language(None, doc_text)
        lmtzd_txts.append(lmtzd_txt)
        pos_counts.append(pos_count)

        lemmatized_doc_text = lmtzd_txts

        # List of tags
        tags = ["ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "NOUN", "NUM", "PART", "PRON", "PROPN", "PUNCT", "SCONJ", "SYM", "VERB"]

        # New dictionary to follow the list of tags
        pos_features = {tag: pos_count.get(tag, 0) for tag in tags}

        # print(pos_features)
        
        with open('./pickle/tokenizer.pickle', 'rb') as handle:
            loaded_tokenizer = pickle.load(handle)
        word_index = loaded_tokenizer.word_index
        vocab_size = len(word_index)

        en_docs = loaded_tokenizer.texts_to_sequences(lemmatized_doc_text)

        seq_len = 450

        pad_docs = pad_sequences(en_docs, maxlen=seq_len, padding="post", truncating="post")

        # Dictionary
        dict_values = list(pos_features.values())
        print(dict_values)
        # Ubah dictionary menjadi array
        dict_arr = np.array(list(dict_values))
        print(dict_arr)
        # Gabungkan array dan dictionary menggunakan np.column_stack
        uji = np.append(pad_docs[0], dict_arr)
        # eX_uji = np.expand_dims(uji, axis=2)
        eX_uji = np.expand_dims(uji, axis=1)
        eX_uji = np.reshape(eX_uji, (1, 465, 1))

        loaded_model = tf.keras.models.load_model('./model/text_classification_model.keras')
        uji_prob = loaded_model.predict(eX_uji)
        # y_pred_uji = np.argmax(uji_prob, axis=1)
        labels = ["Business", "Entertainment", "Politics", "Sport", "Tech"]
        result = f"{labels[np.argmax(uji_prob)]}, {np.max(uji_prob)}"
    except BrokenPipeError as e:
        st.error(f"BrokenPipeError: {e}")
        raise
    return result




# file_uploaded = st.file_uploader("Pilih File", type=["png","jpg","jpeg"])
class_btn = st.button("Klasifikasikan!")
# if file_uploaded is not None:    
#     image = Image.open(file_uploaded)
#     st.write(image)
#     st.image(image, caption='Image yang diupload')
        
if class_btn:
    if teks_baru is None:
        st.write("Perintah tidak valid, harap isi teks!")
    else:
        with st.spinner('Model sedang bekerja....'):
            predictions = predicting(teks_baru)
            time.sleep(1)
            st.success('Telah diklasifikasi!')
            st.write(predictions)

# Plot!
st.subheader("Apakah Kamu Tahu?")
st.write("Di antara berikut ini, manakah yang menggunakan AI? (Pilih 3)")
q0, q1, q2, q3, q4, q5 = st.columns(6)
with q0:
    st.image('./assets/img/robot.jpg', width=100)
    a0 = st.checkbox("Robot")
with q1:
    st.image('./assets/img/selfDriving.png', width=100)
    a1 = st.checkbox("Self-Driving Car")
with q2:
    st.image('./assets/img/telepon.jpg', width=100)
    a2 = st.checkbox("Telepon")
with q3:
    st.image('./assets/img/kalkulator.jpg', width=100)
    a3 = st.checkbox("Kalkulator")
with q4:
    st.image('./assets/img/apple_vision_pro.jpg', width=100)
    a4 = st.checkbox("Apple Vision Pro")
with q5:
    st.image('./assets/img/pensil-warna.jpg', width=100)
    a5 = st.checkbox("Pensil Warna") 
if a0 and (a1 == True) and (a2 == False) and (a3 == False) and (a4 == True) and (a5 == False):
    # st.write("Anda sudah setuju")
    with st.expander("Selamat, Kamu Benar!"):
        st.write("""
            Fun Fact!
            \nRobot, Self-Driving Car, dan Apple Vision Pro memiliki teknologi AI.
        """)
else:
    st.write("Anda belum menjawab/jawaban anda kurang dari 3/jawaban anda salah.")


st.write("")
st.subheader("Informasi mengenai hasil proses *training* pada model teks klasifikasi")

left_co, cent_co, last_co = st.columns(3)
with cent_co:
    st.image('assets/img/acc_graph.png', caption='Training & Validation Accuracy Graph')

left_co2, cent_co2, last_co2 = st.columns(3)
with cent_co2:
    st.image('assets/img/loss_graph.png', caption='Training & Validation Loss Graph')

left_co3, cent_co3, last_co3 = st.columns(3)
with cent_co3:
    st.image('assets/img/class_report.png', caption='classification_report')

left_co4, cent_co4, last_co4 = st.columns(3)
with cent_co4:
    st.image('assets/img/confusion_matrix_text.png', caption='Confusion Matrix')
st.write("Keterangan: 'Business': 0, 'Entertainment': 1, 'Politics': 2, 'Sport': 3, 'Tech': 4")
