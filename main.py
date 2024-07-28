from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from audiocraft.models import MusicGen
import torch
import torchaudio
import os

app = FastAPI()

class MusicRequest(BaseModel):
    prompt: str
    duration: int

def load_model():
    model = MusicGen.get_pretrained("small")
    return model

def generate_tensor(prompt, duration):
    model = load_model()
    model.set_generation_params(
        use_sampling=True,
        top_k=200,
        duration=duration,
    )
    output = model.generate(
        descriptions=[prompt],
        progress=True,
        return_tokens=True
    )
    return output[0]

def save(samples: torch.Tensor):
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

    return audio_path

@app.post("/generate")
def generate_music(request: MusicRequest):
    try:
        music_tensor = generate_tensor(request.prompt, request.duration)
        audio_path = save(music_tensor)
        return {"audio_path": audio_path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
