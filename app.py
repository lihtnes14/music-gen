import streamlit as st
import requests

st.set_page_config(
    page_icon="",
    page_title="Music Generator"
)

def main():
    st.title("Text to Music")
    prompt = st.text_area("Enter your description")
    duration = st.slider('Select Duration (seconds):', min_value=2, max_value=10, value=5)

    if st.button("Generate"):
        response = requests.post(
            "http://192.168.1.8:8000/generate",
            json={"prompt": prompt, "duration": duration}
        )
        
        if response.status_code == 200:
            audio_path = response.json()["audio_path"]
            audio_file = open(audio_path, "rb")
            audio_bytes = audio_file.read()
            st.audio(audio_bytes)
        else:
            st.error("Failed to generate music")

if __name__ == "__main__":
    main()
