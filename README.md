# 🎙️ Youtube Vietnamese Speech To Text (STT)

Ứng dụng **Nhận dạng tiếng nói tiếng Việt** sử dụng mô hình **CNN – Transformer – CTC Decoder + KenLM**, triển khai giao diện bằng **Streamlit**.

---

## 🚀 Tính năng chính

- 🔊 Nhận dạng tiếng nói tiếng Việt từ file audio (`.wav`, `.mp3`, `.m4a`)
- 🧠 Mô hình học sâu CNN + Transformer
- 📖 Giải mã CTC bằng **pyctcdecode + KenLM Language Model**
- 🌐 Giao diện web trực quan với **Streamlit**
- 🎧 Hỗ trợ audio đơn / đa kênh, tự động resample về 16kHz

---

### 📊 Kết quả thực nghiệm
Mô hình được đánh giá trên tập dữ liệu tiếng Việt với hai chỉ số phổ biến:
- CER (Character Error Rate) – Tỷ lệ lỗi ký tự
- WER (Word Error Rate) – Tỷ lệ lỗi từ

| Mô hình                  | CER ↓  | WER ↓      |
| ------------------------ | ------ | ---------- |
| **CNN-LSTM**             | —      | —          |
| **CNN-Transformer**      | 0.1597 | 0.3406     |
| **CNN-Transformer + LM** | 0.1721 | **0.2935** |

### 📌 Nhận xét:
- Việc tích hợp Language Model (KenLM) giúp giảm đáng kể WER
- Mô hình CNN-Transformer-LM cho kết quả tốt nhất về mặt nhận dạng từ
- CER tăng nhẹ khi dùng LM do ưu tiên tính đúng ngữ cảnh từ

## 📁 Cấu trúc thư mục
```
Project_YoutubeVietnamese_S2T/
│
├── Youtube_Tool/
│   ├── main.py # Tải audio từ Youtube 
│   ├── urls.txt # Danh sách link Youtube
│   └── bin/
│       ├── ffmpeg.exe # tải từ Google Drive
│       └── yt-dlp.exe # tải từ Google Drive
│
├── model/
│   └── speech_model.py # Định nghĩa mô hình Speech Recognition 
│
├── utils/
│   └── text_transform.py # Xử lý text, mapping ký tự, CTC labels
│
├── app.py # Ứng dụng Streamlit chính
├── Vietnamese_char.txt # Danh sách ký tự tiếng Việt
├── best_model.pth # Model đã huấn luyện sẵn    
├── vi_lm_5grams.bin # tải từ Google Drive
├── requirements.txt # Danh sách thư viện Python
└── .gitignore
```
## ⚙️ Cài đặt môi trường

### 1️⃣ Tạo môi trường ảo
```bash
python -m venv venv
source venv/bin/activate     # Linux / Mac
venv\Scripts\activate        # Windows
```
### 2️⃣ Cài thư viện
```bash
pip install -r requirements.txt
```
### 📥 Tải các file cần thiết (Google Drive)
```bash
📌 Link Google Drive: https://drive.google.com/drive/folders/10EsoHqIEeRsrtWyMUgFdXfCDls38kqWM?usp=sharing
- Lưu các file vừa tải theo đúng trong cấu trúc thư mục
```
### ▶️ Chạy ứng dụng
```bash
python -m streamlit run app.py
```
### 🧠 Công nghệ sử dụng
- PyTorch
- TorchAudio
- Streamlit
- pyctcdecode
- KenLM
- FFmpeg
### ✨ Tác giả
SimpleAI04-Vietnamese Speech To Text Project 🇻🇳
