import streamlit as st
import torch
import librosa
import torchaudio
from model.speech_model import SpeechRecognitionModel
from utils.text_transform import TextTransform
from pyctcdecode import build_ctcdecoder

# Cấu hình Page
st.set_page_config(page_title="Vietnamese STT Demo", page_icon="🎙️")


@st.cache_resource
def init_app():
    tt = TextTransform()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SpeechRecognitionModel(
        n_mels=80,
        n_class=len(tt.all_char),
        d_model=384,
        num_layers=6,
        nhead=6,
        dropout=0.3,
    )
    model.load_state_dict(
        torch.load(r"D:\Dowloads_1\S2T\best_model.pth", map_location=device)
    )
    model.to(device)
    model.eval()

    # 3. Load KenLM Decoder
    labels = [c if c != "<BLANK>" else "" for c in tt.all_char]
    decoder = build_ctcdecoder(
        labels=labels,
        # kenlm_model_path=r"D:\Dowloads_1\S2TDemo\vi_lm_5grams.bin",
        # alpha=0.5,
        # beta=1.5,
    )

    return model, decoder, tt, device


model, decoder, tt, device = init_app()

st.title("🎙️ Vietnamese SPEECH TO TEXT")
st.info("Mô hình CNN-Transformer-LM | Ngôn ngữ: Tiếng Việt")

uploaded_file = st.file_uploader("Chọn file audio...", type=["wav", "mp3", "m4a"])

if uploaded_file:
    st.audio(uploaded_file)

    if st.button("Bắt đầu nhận dạng"):
        with st.spinner("Đang xử lý âm thanh..."):
            audio, sr = librosa.load(uploaded_file, sr=16000, mono=True)
            waveform = torch.from_numpy(audio).unsqueeze(0)
            spec = torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_mels=80)(
                waveform
            )

            spec = torchaudio.transforms.AmplitudeToDB()(spec).to(device)
            spec = spec.unsqueeze(0)

            with torch.no_grad():
                lengths = torch.tensor([spec.shape[-1]]).to(device)
                logits, _ = model(spec, lengths)  # (1, Time, Class)

            probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
            result = decoder.decode(probs, beam_width=20)

            st.success("Nhận dạng hoàn tất!")
            st.write("### Output:")
            st.markdown(f"### **{result}**")
