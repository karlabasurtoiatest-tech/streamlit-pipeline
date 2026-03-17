import streamlit as st
import os
import time

# -------------------------------
# NLTK: descargar recursos necesarios
# -------------------------------
import nltk
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
stop_words = set(stopwords.words('spanish'))

# -------------------------------
# Otros imports
# -------------------------------
import whisper
from transformers import pipeline
import yt_dlp

# -------------------------------
# Configuración de página
# -------------------------------
st.set_page_config(page_title=" Clasificador de Vídeos Deportivos", layout="wide")
st.title(" Clasificador de Vídeos Deportivos")
st.write("Introduce la URL de YouTube o sube un archivo de audio MP3 generado desde Colab.")

# Lista oficial de deportes
SPORTS_CATEGORIES = [
    "Fútbol", "Baloncesto", "Tenis", "Fórmula 1 / Automovilismo", "Ciclismo",
    "Natación", "Rugby", "Deportes de invierno", "Boxeo / Artes marciales", "Vela"
]

# Inputs
youtube_url = st.text_input("Introduce URL de YouTube (opcional)")
audio_file = st.file_uploader("O sube un audio MP3 desde Colab (opcional)", type=["mp3"])

# Nombre fijo para audio temporal
AUDIO_TEMP = "temp_audio.mp3"

# -------------------------------
# Función pipeline
# -------------------------------
def pipeline_func(audio_filename):
    start_total_time = time.time()
    output = {}

    #  Transcripción
    start_trans_time = time.time()
    model = whisper.load_model("base")
    result = model.transcribe(audio_filename, language="es")
    transcription = result["text"]
    trans_time = time.time() - start_trans_time
    output['transcription'] = transcription
    output['trans_time'] = trans_time

    #  Resumen (limpiando stopwords)
    words = [w for w in transcription.split() if w.lower() not in stop_words]
    clean_text = " ".join(words)
    if len(clean_text.split()) < 50:
        summary = clean_text
    else:
        sentences = sent_tokenize(clean_text)
        summary = " ".join(sentences[:3])
    output['summary'] = summary

    #  Clasificación Zero-Shot
    classifier = pipeline("zero-shot-classification", model="Recognai/bert-base-spanish-wwm-cased-xnli")
    result_cls = classifier(summary, candidate_labels=SPORTS_CATEGORIES)
    output['predicted_sport'] = result_cls['labels'][0]
    output['confidence'] = result_cls['scores'][0]*100
    output['top3'] = [(result_cls['labels'][i], result_cls['scores'][i]*100) for i in range(min(3,len(result_cls['labels'])))]

    #  Tiempos y entorno
    total_time = time.time() - start_total_time
    output['total_time'] = total_time
    output['environment'] = {
        'STT_model': "Whisper 'base'",
        'Classification_model': "Recognai/bert-base-spanish-wwm-cased-xnli"
    }

    # Limpiar audio temporal
    if os.path.exists(audio_filename):
        os.remove(audio_filename)

    return output

# ===============================
# Ejecutar pipeline al presionar botón
# ===============================
if st.button("Ejecutar pipeline"):
    audio_filename = None

    #  Intentar descargar desde YouTube si hay URL
    if youtube_url:
        try:
            ydl_opts = {
                'format': 'bestaudio/best',
                'postprocessors': [{
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': 'mp3',
                    'preferredquality': '192',
                }],
                'outtmpl': AUDIO_TEMP,
                'noplaylist': True,
                'quiet': True,
                'no_warnings': True
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([youtube_url])
            st.success(" Audio descargado de YouTube correctamente")
            audio_filename = AUDIO_TEMP
        except Exception as e:
            st.warning(f" No se pudo descargar el vídeo: {e}")
            audio_filename = None

    #  Si hay audio subido, usarlo
    if audio_file is not None:
        audio_filename = AUDIO_TEMP
        with open(audio_filename, "wb") as f:
            f.write(audio_file.read())
        st.success(f" Archivo {audio_file.name} subido correctamente")

    #  Validar que hay audio
    if audio_filename is None or not os.path.exists(audio_filename):
        st.error(" No hay audio para procesar. Introduce una URL válida o sube un archivo MP3.")
    else:
        with st.spinner("Procesando pipeline..."):
            results = pipeline_func(audio_filename)

        # Mostrar resultados
        st.subheader(" Transcripción")
        st.write(results['transcription'][:200] + "..." if len(results['transcription'])>200 else results['transcription'])

        st.subheader(" Resumen")
        st.write(results['summary'])

        st.subheader(" Clasificación")
        st.write(f"Deporte predicho: {results['predicted_sport']}")
        st.write(f"Confianza: {results['confidence']:.2f}%")
        st.write("Top 3:")
        for sport, score in results['top3']:
            st.write(f"- {sport}: {score:.2f}%")

        st.subheader(" Rendimiento y entorno")
        st.write(f"Tiempo transcripción: {results['trans_time']:.2f} s")
        st.write(f"Tiempo total pipeline: {results['total_time']:.2f} s")
        st.write("Modelos usados:", results['environment'])
