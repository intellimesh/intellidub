from fastapi import FastAPI, UploadFile, File, Response, Form
import openai
import tempfile
import os
import requests
from llama_index.core import VectorStoreIndex, Document, Settings
from llama_index.core.query_engine import CustomQueryEngine
from llama_index.core.retrievers import BaseRetriever
from llama_index.core import get_response_synthesizer
from llama_index.core.response_synthesizers import BaseSynthesizer
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.core.agent import ReActAgent
from llama_index.core.tools import QueryEngineTool
from whisper import load_model
from transformers import AutoTokenizer

app = FastAPI()

# Load Whisper model for transcription
whisper_model = load_model("base")  # Replace "base" with the specific model you want to use

# Configure OpenAI API keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
#openai.api_key = OPENAI_API_KEY
openai.api_key = NVIDIA_API_KEY
base_url = "https://integrate.api.nvidia.com/v1",
# model_name="gpt-4"
model_name = "meta/llama-3.1-405b-instruct"

# Set up LlamaIndex configuration
# Settings.llm = OpenAI(temperature=0.7, model_name="gpt-4")  # Use GPT-4 for better language capabilities
Settings.llm = OpenAI(base_url = base_url, model_name = model_name, temperature=0.7)

embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

def transcribe_audio(file_path):
    """Transcribe audio using OpenAI's Whisper model."""
    result = whisper_model.transcribe(file_path)
    return result["text"]

class RAGQueryEngine(CustomQueryEngine):
    """RAG Query Engine."""

    retriever: BaseRetriever
    response_synthesizer: BaseSynthesizer

    def custom_query(self, query_str: str):
        nodes = self.retriever.retrieve(query_str)
        response_obj = self.response_synthesizer.synthesize(query_str, nodes)
        return response_obj

def translate_text_with_context(text, target_language="es", context_file_path=None):
    """Translate text using LlamaIndex with GPT-4 for context-aware translation."""
    primary_document = Document(text=text)
    context_documents = []

    # Add context document if provided
    if context_file_path:
        with open(context_file_path, 'r') as context_file:
            context_text = context_file.read()
            context_documents.append(Document(text=context_text))
    
    # Create index using primary text and any context documents
    index = VectorStoreIndex.from_documents([primary_document] + context_documents, embed_model=embed_model)
    # query_engine = index.as_query_engine()
    retriever = index.as_retriever()
    synthesizer = get_response_synthesizer(response_mode="compact")
    query_engine = RAGQueryEngine(
        retriever=retriever, response_synthesizer=synthesizer
    )   
    translate_context_tool = QueryEngineTool.from_defaults(
        query_engine,
        name="context_aware_translation",
        description="A RAG engine for context-aware translation.",
    )
    agent = ReActAgent.from_tools([translate_context_tool], verbose=True)
    
    # Refine prompt to ensure only the source text is translated
    prompt = (
        f"Translate the following text to {target_language}: {text}. "
        f"You may request additional context using context_aware_translation tool. The context helps improve the translation "
        f"Do not include any context or additional information in the response."
    )
    
    response = agent.query(prompt)
    
    # Extract and return only the translated text
    translated_text = response.response.strip()
    return translated_text


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
async def translate_audio(
    file: UploadFile = File(...),
    target_language: str = Form("es"),
    context_file: UploadFile = File(None)
):
    """Endpoint to handle the translation of an MP3 file, with optional context upload."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio:
        temp_audio.write(await file.read())
        temp_audio_path = temp_audio.name

    context_file_path = None
    try:
        # Save context file if provided
        if context_file:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as temp_context:
                temp_context.write(await context_file.read())
                context_file_path = temp_context.name
        
        # Step 1: Transcribe audio
        transcription = transcribe_audio(temp_audio_path)
        print(f"Transcription: {transcription}")

        # Step 2: Translate text using LlamaIndex with GPT-4 for context
        translated_text = translate_text_with_context(transcription, target_language, context_file_path)
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
        # Remove temporary files
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)
        if context_file_path and os.path.exists(context_file_path):
            os.remove(context_file_path)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

# Required pip installations:
# pip install fastapi uvicorn openai whisper requests llama-index-core llama-index-llms-openai llama-index-embeddings-huggingface transformers
