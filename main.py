from audiocraft.models import MusicGen
import streamlit as st
import torch
import torchaudio
import numpy
import base64
import os

def load_model():
    model = MusicGen.get_pretrained("small")
    return model

def generate_music_tensor(description,duration):
    model = load_model()
    model.set_generation_params(
        use_sampling = True,
        top_k = 200,
        duration = duration,
    )

    output = model.generate(
        descriptions = [description],
        progress = True,
        return_tokens = True
    )

    return output[0]

def save_audio(samples: torch.Tensor):
    sample_rate = 32000  # Adjusted to a more typical sample rate
    save_path = "audiocraft/"

    os.makedirs(save_path, exist_ok=True)

    assert samples.dim() == 2 or samples.dim() == 3

    sample = samples.detach().cpu()

    if samples.dim() == 2:
        samples = samples[None, ...]

    for idx, audio in enumerate(samples):
        audio_path = os.path.join(save_path, f"audio{idx}.wav")
        try:
            torchaudio.save(audio_path, audio, sample_rate)
        except Exception as e:
            print(f"Failed to save audio at {audio_path}: {e}")

def get_binary_file_downloader_html(bin_file, file_label='File'):
    with open(bin_file, 'rb') as f:
        data = f.read()
    bin_str = base64.b64encode(data).decode()
    href = f'<a href="data:application/octet-stream;base64,{bin_str}" download="{os.path.basename(bin_file)}">Download {file_label}</a>'
    return href

st.set_page_config(
    page_icon = "",
    page_title = "Music Generator"
)
def main():
    st.title("Text to Music")
    with st.expander("See explaination"):
        st.write("abcd")
    description = st.text_area("Enter your description")

    duration = st.slider('Select Duration (seconds):', min_value=2, max_value=10, value=5)
    if st.button("Generate"):
        music_tensor  = generate_music_tensor(description, duration)
        save_music_file = save_audio(music_tensor)
        audio_filepath = "audiocraft/audio0.wav"
        audio_file = open(audio_filepath, "rb")
        audio_bytes = audio_file.read()
        st.audio(audio_bytes)


if __name__ == "__main__":
    main()