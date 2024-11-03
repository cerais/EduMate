import json
import concurrent.futures
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from vosk import Model, KaldiRecognizer
from pydub import AudioSegment
from mistralai import Mistral
import requests
from moviepy.editor import VideoFileClip
from gradio_client import Client, handle_file

app = FastAPI()

# Устанавливаем параметры
FRAME_RATE = 16000
CHANNELS = 1

# Модель данных для запросов
class YandexDiskRequest(BaseModel):
    public_key: str

# Функция для скачивания видео с Яндекс.Диска
def download_from_yandex_disk(public_key, output_path="downloaded_video.mp4"):
    download_url = f"https://cloud-api.yandex.net/v1/disk/public/resources/download?public_key={public_key}"
    response = requests.get(download_url)
    
    if response.status_code != 200:
        raise HTTPException(status_code=400, detail="Ошибка при получении ссылки для скачивания. Проверьте публичную ссылку.")
    
    download_link = response.json().get("href")
    if not download_link:
        raise HTTPException(status_code=400, detail="Не удалось получить ссылку для скачивания.")
    
    # Скачиваем файл
    video_response = requests.get(download_link, stream=True)
    if video_response.status_code == 200:
        with open(output_path, "wb") as f:
            f.write(video_response.content)
        print(f"Видео успешно скачано как {output_path}")
    else:
        raise HTTPException(status_code=500, detail="Ошибка при скачивании видео.")
    
    return output_path

# Функция для конвертации mp4 в wav
def convert_mp4_to_wav(mp4_file, wav_file):
    video = VideoFileClip(mp4_file)
    audio = video.audio
    audio.write_audiofile(wav_file, codec='pcm_s16le')
    video.close()  # Закрытие видео после обработки

# Функция для обработки аудио с помощью Vosk
def process_audio(wav_file, model_path="vosk-model-small-ru-0.22"):
    model = Model(model_path)
    rec = KaldiRecognizer(model, FRAME_RATE)
    mp3 = AudioSegment.from_wav(wav_file)
    mp3 = mp3.set_channels(CHANNELS).set_frame_rate(FRAME_RATE)

    rec.AcceptWaveform(mp3.raw_data)
    result = rec.Result()
    return json.loads(result)["text"]

# Функция для обработки видео с помощью Vision-CAIR/LongVU
def process_video(video_file):
    client = Client("Vision-CAIR/LongVU")
    result = client.predict(
        video={"video": handle_file(video_file)},
        chatbot=[],
        textbox_in="summarize",
        temperature=0.2,
        top_p=0.7,
        max_output_tokens=1024,
        api_name="/generate"
    )[1][0][1]
    return result

# Функция для создания контент-карточек
def generate_content_cards(full_summary):
    prompt = f"""
    You are a professional summarizer, and your task is to create concise and informative content cards in Russian. These content cards should highlight the key points of the summary provided. Each card should cover a single main point and present it in clear, simple language for easy understanding.

    # SUMMARY:
    {full_summary}

    # Instructions:
    1. Identify the most important points and details from the summary.
    2. Create content cards with each card containing one main point in a short, concise format.
    3. Each card should have a title summarizing the point, followed by a 1-2 sentence description.
    4. Present each card in Russian and use language that is accessible and informative.

    Please respond with each card in the following format:

    - **Card Title**: [Brief title summarizing the point]
    - **Description**: [1-2 sentence description of the point]
    """

    # Запрос к Mistral API для создания карточек
    api_key = "YOUR_MISTRAL_API_KEY"
    mistral_client = Mistral(api_key=api_key)
    model = "mistral-large-latest"

    chat_response = mistral_client.chat.complete(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )
    
    # Возвращаем карточки
    return chat_response.choices[0].message.content

# Основной процесс
def process_video_and_audio(public_key, mp4_file, wav_file):
    # Скачиваем видео
    download_from_yandex_disk(public_key, mp4_file)

    # Выполняем конвертацию видео и обработку параллельно
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Запуск задач параллельно после завершения скачивания
        convert_task = executor.submit(convert_mp4_to_wav, mp4_file, wav_file)
        audio_task = executor.submit(process_audio, wav_file)
        video_task = executor.submit(process_video, mp4_file)

        # Ожидаем выполнения задач
        convert_task.result()  # Ждем завершения конвертации
        audio_summary = audio_task.result()
        video_summary = video_task.result()

    # Используем Mistral для создания итогового отчета
    api_key = "O7lLJClDjHL3S5OGjEuUQzxU7crw22qZ"
    mistral_client = Mistral(api_key=api_key)
    model = "mistral-large-latest"

    prompt = f"""
        You are a commentator.
        # AUDIO SUMMARY:
        {audio_summary}
        
        # Instructions:
        ## Summarize:
        Clearly and concisely state the key points and main topics from both summaries combined. Present the unified summary in RUSSIAN.
        """

    # Запрос к Mistral API для итогового отчета
    chat_response = mistral_client.chat.complete(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )

    # Получаем итоговый отчет
    full_summary = chat_response.choices[0].message.content

    # Создаем карточки с основным содержанием
    content_cards = generate_content_cards(full_summary)

    # Возвращаем итоговый отчет и карточки
    return {
        "summary": full_summary,
        "content_cards": content_cards
    }

# Маршрут для обработки видео и аудио
@app.post("/process/")
async def process_request(request: YandexDiskRequest):
    public_key = request.public_key
    mp4_file = "downloaded_video.mp4"
    wav_file = "downloaded_video.wav"

    try:
        result = process_video_and_audio(public_key, mp4_file, wav_file)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
