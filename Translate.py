import sounddevice as sd
import queue
import threading
import whisper
import librosa
from deep_translator import GoogleTranslator
import numpy as np

model = whisper.load_model("medium")
translator = GoogleTranslator(source='auto', target='zh-TW')

sample_rate = 48000
device_id = 23
chunk_duration = 10

audio_queue = queue.Queue()
buffer_accum = []
warmup = 3 

def audio_callback(indata, frames, time, status):
    global warmup, buffer_accum
    if status:
        print(status)
    if warmup > 0:
        print("丟掉殘留音訊 buffer")
        warmup -= 1
        return

    buffer_accum.append(indata.copy())

    # 累積到 chunk_duration 秒再送
    if len(buffer_accum) * frames >= sample_rate * chunk_duration:
        audio_data = np.concatenate(buffer_accum, axis=0)
        buffer_accum = []  # 清空累積

        # 轉單聲道 + 重採樣
        audio_chunk = librosa.to_mono(audio_data.T)
        audio_chunk = librosa.resample(audio_chunk, orig_sr=sample_rate, target_sr=16000)
        audio_queue.put(audio_chunk)


def process_audio():
    while True:
        audio_chunk = audio_queue.get()
        result = model.transcribe(audio_chunk, fp16=False, language='en')
        text = result["text"].strip()
        if text:
            print(f"\n辨識結果：{text}")
            translated = translator.translate(text)
            print(f"中文翻譯：{translated}\n")
        audio_queue.task_done()

print("開始即時翻譯，按 Ctrl+C 停止")

# 啟動翻譯執行緒
threading.Thread(target=process_audio, daemon=True).start()

# 開啟持續錄音 stream
with sd.InputStream(samplerate=sample_rate, channels=2, device=device_id, callback=audio_callback):
    threading.Event().wait()  # 主程式保持運行
