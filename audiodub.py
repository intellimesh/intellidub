from fastapi import FastAPI, UploadFile, File, Response
import openai
import tempfile
import os
import requests
from llama_index.core import VectorStoreIndex, Document
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openai import OpenAI
from whisper import load_model
from transformers import AutoTokenizer

app = FastAPI()

# Load Whisper model for transcription
whisper_model = load_model("base")  # Replace "base" with the specific model you want to use

# Configure OpenAI API keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY

# Set up LlamaIndex configuration
llm = OpenAI(temperature=0.7, model_name="gpt-4")  # Use GPT-4 for better language capabilities
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

def transcribe_audio(file_path):
    """Transcribe audio using OpenAI's Whisper model."""
    result = whisper_model.transcribe(file_path)
    return result["text"]

def translate_text_with_context(text, target_language="te"):
    """Translate text using LlamaIndex with GPT-4 for context-aware translation."""
    document = Document(text=text)
    index = VectorStoreIndex.from_documents([document], llm=llm, embed_model=embed_model)
    query_engine = index.as_query_engine()
    
    prompt = f"Translate the following text to {target_language} with full context and accuracy: {text}"
    response = query_engine.query(prompt)
    return response.response.strip()

def text_to_speech(text, voice="alloy"):
    """Generate TTS output using OpenAI's new API in the target language."""
    url = "https://api.openai.com/v1/audio/speech"
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "tts-1",
        "input": text,
        "voice": voice
    }

    try:
        response = requests.post(url, headers=headers, json=data)
        if response.status_code == 200:
            return response.content  # Return the audio content
        else:
            print(f"Failed to convert text to speech: {response.status_code} - {response.text}")
            return None
    except requests.RequestException as e:
        print(f"Error during API request: {e}")
        return None

@app.post("/translate-audio/")
async def translate_audio(file: UploadFile = File(...), target_language: str = "te"):
    """Endpoint to handle the translation of an MP3 file."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio:
        temp_audio.write(await file.read())
        temp_audio_path = temp_audio.name

    try:
        # Step 1: Transcribe audio
        transcription = transcribe_audio(temp_audio_path)
        print(f"Transcription: {transcription}")

        # Step 2: Translate text using LlamaIndex with GPT-4 for context
        translated_text = translate_text_with_context(transcription, target_language)
        print(f"Translated Text: {translated_text}")

        # Step 3: Generate TTS audio
        tts_audio = text_to_speech(translated_text, voice="alloy")
        if tts_audio:
            # Return the audio data directly as a response
            headers = {"Content-Disposition": "attachment; filename=translated_audio.mp3"}
            return Response(content=tts_audio, media_type="audio/mpeg", headers=headers)
        else:
            return {"message": "TTS generation failed.", "translation": translated_text}

    finally:
        # Remove only the temporary input audio file
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

# Required pip installations:
# pip install fastapi uvicorn openai whisper requests llama-index-core llama-index-llms-openai llama-index-embeddings-huggingface transformers
