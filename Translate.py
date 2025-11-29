import sounddevice as sd
import numpy as np
import whisper
import librosa
from deep_translator import GoogleTranslator

model = whisper.load_model("medium")
translator = GoogleTranslator(source='auto', target='zh-TW')

sample_rate = 48000
device_id = 23   # CABLE Output
chunk_duration = 10

print("🎙️ 開始即時翻譯，按 Ctrl+C 停止")
5
while True:
    print("⏺️ 錄音中...")
    recording = sd.rec(int(chunk_duration * sample_rate),
                       samplerate=sample_rate,
                       channels=2,
                       dtype='float32',
                       device=device_id)
    sd.wait()

    # 轉單聲道
    audio_chunk = librosa.to_mono(recording.T)
    audio_chunk = librosa.resample(audio_chunk, orig_sr=sample_rate, target_sr=16000)

    # Whisper 辨識
    result = model.transcribe(audio_chunk, fp16=False)

    english_text = result["text"].strip()

    if english_text:
        print(f"\n🗣️ 英文辨識結果：{english_text}")
        translated = translator.translate(english_text)
        print(f"🌐 中文翻譯：{translated}\n")
    else:
        print("🤷 沒有辨識到語音內容")