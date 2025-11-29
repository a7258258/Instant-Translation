# Instant-Translation
即時語音辨識 + 翻譯系統，支援英文、日文轉繁體中文。
使用 Whisper + deep-translator + sounddevice。

## Install
```bash
pip install -r requirements.txt
```

該程式需要安裝VB-CABLE Virtual Audio Device
[網站連結:https://vb-audio.com/Cable/](https://vb-audio.com/Cable/)

### First ###
執行device_check.py
將CABLE Output (VB-Audio Virtual Cable)編號設為device_id

Whisper 支援超過 50 種語言 在result後加上即可
result = model.transcribe(audio_chunk, fp16=False, language='en')  #英文
result = model.transcribe(audio_chunk, fp16=False, language='ja')  #日文
result = model.transcribe(audio_chunk, fp16=False, language='zh')  #中文
---

### Second
執行Translate.py