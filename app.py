import streamlit as st
import time
import os
import yt_dlp
import whisper
from nltk.tokenize import sent_tokenize
from transformers import pipeline

# Configuración inicial

st.title("Clasificador de vídeos deportivos")
st.write("Introduce la URL de YouTube y ejecuta el pipeline para obtener transcripción, resumen y clasificación de deportes.")

# Lista oficial de 10 deportes
SPORTS_CATEGORIES = [
    "Fútbol", "Baloncesto", "Tenis", "Fórmula 1 / Automovilismo", "Ciclismo",
    "Natación", "Rugby", "Deportes de invierno", "Boxeo / Artes marciales", "Vela"
]

# Input URL

youtube_url = st.text_input("Introduce la URL de YouTube")

# Pipeline por fases
def pipeline(url):
    start_total_time = time.time()
    output = {}

    # Flujo 0 – Inicialización
  
    output['url'] = url
    output['sports_categories'] = SPORTS_CATEGORIES

    # Obtener duración aproximada del vídeo usando yt-dlp
    ydl_opts_info = {'quiet': True, 'no_warnings': True}
    with yt_dlp.YoutubeDL(ydl_opts_info) as ydl:
        info_dict = ydl.extract_info(url, download=False)
        duration = info_dict.get('duration', 0) # en segundos
        output['duration'] = duration

    # Flujo 1 – Descarga de audio temporal

    audio_filename = "temp_audio.mp3"
    ydl_opts_audio = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'outtmpl': audio_filename,
        'noplaylist': True,
        'quiet': True,
        'no_warnings': True
    }
    with yt_dlp.YoutubeDL(ydl_opts_audio) as ydl:
        ydl.download([url])
    output['audio_filename'] = audio_filename

    # Flujo 2 – Transcripción

    start_transcription_time = time.time()
    model = whisper.load_model("base")  # Modelo base
    result = model.transcribe(audio_filename, language="es")
    transcription_text = result["text"]
    transcription_time = time.time() - start_transcription_time
    output['transcription'] = transcription_text
    output['transcription_time'] = transcription_time


    # Flujo 3 – Resumen
  
    # Para videos cortos usamos primeras 3 oraciones
    if len(transcription_text.split()) < 50:
        summary = transcription_text
    else:
        sentences = sent_tokenize(transcription_text)
        summary = ' '.join(sentences[:3])
    output['summary'] = summary

    # Flujo 4 – Clasificación Zero-Shot

    classifier = pipeline("zero-shot-classification", model="Recognai/bert-base-spanish-wwm-cased-xnli")
    result_cls = classifier(summary, candidate_labels=SPORTS_CATEGORIES)
    output['predicted_sport'] = result_cls['labels'][0]
    output['confidence'] = result_cls['scores'][0]*100
    output['top3'] = [(result_cls['labels'][i], result_cls['scores'][i]*100) for i in range(min(3,len(result_cls['labels'])))]


    # Tiempo total y entorno

    total_time = time.time() - start_total_time
    output['total_time'] = total_time
    output['environment'] = {
        'STT_model': "Whisper 'base'",
        'Classification_model': "Recognai/bert-base-spanish-wwm-cased-xnli"
    }

    # Eliminar archivo temporal de audio
    if os.path.exists(audio_filename):
        os.remove(audio_filename)

    return output

# Botón ejecutar

if st.button("Procesar vídeo"):
    if youtube_url == "":
        st.error(" Introduce una URL válida")
    else:
        with st.spinner("Ejecutando pipeline..."):
            results = pipeline(youtube_url)


        # Mostrar resultados

        st.subheader(" Flujo 0 – Inicialización")
        st.write(f"URL: {results['url']}")
        st.write(f"Duración aproximada: {results['duration']} segundos")
        st.write("Categorías de deportes oficiales:")
        for cat in results['sports_categories']:
            st.write(f"- {cat}")

        st.subheader(" Flujo 1 – Descarga de audio")
        st.write(f"Paso 1 completado: Archivo generado temporalmente (no se guarda)")

        st.subheader(" Flujo 2 – Transcripción")
        st.write(f"Paso 2 completado. Primeros 200 caracteres:")
        st.write(results['transcription'][:200])

        st.subheader(" Flujo 3 – Resumen")
        st.write("Paso 3 completado: Resumen generado")
        st.write(results['summary'])

        st.subheader(" Flujo 4 – Clasificación")
        st.write(f"Paso 4 completado")
        st.write(f"Deporte predicho: {results['predicted_sport']}")
        st.write(f"Confianza: {results['confidence']:.2f}%")
        st.write("Top 3:")
        for sport, score in results['top3']:
            st.write(f"- {sport}: {score:.2f}%")

        st.subheader(" Detalles de rendimiento y entorno")
        st.write(f"Tiempo de transcripción: {results['transcription_time']:.2f} segundos")
        st.write(f"Tiempo total del pipeline: {results['total_time']:.2f} segundos")
        st.write(f"Modelos usados: {results['environment']}")
