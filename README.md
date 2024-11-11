# IntelliDub: Advanced Context-Aware Translation and Dubbing Platform

## Project Overview
This project provides an advanced solution for translating audio and text input, with **Context-Aware Translation** as its primary feature. This approach significantly enhances the quality of translations by leveraging contextual information to ensure that translations are not just literal but contextually relevant, maintaining meaning across various nuances of language.

### Advantages of Context-Aware Translation
Context-aware translation goes beyond basic word-for-word translation. By incorporating a context file provided by the user, the application can:
- **Preserve the meaning**: Ensure that idiomatic expressions and culturally nuanced content are translated accurately.
- **Enhance accuracy**: Adapt translations based on the context of preceding or related text, resulting in a coherent output.
- **Reduce ambiguity**: Clarify words with multiple meanings by considering the surrounding information.

This approach is particularly useful for technical, academic, or domain-specific translations where precision is paramount.

### Features
- **Audio Transcription and Translation**: Upload an audio file (e.g., MP3) for transcription and translation.
- **Text Translation**: Submit plain text for translation.
- **Context-Aware Translation**: Upload a context file to enhance the translation quality.
- **Integration with `meta/llama-3.1-405b-instruct` and LlamaIndex**: Uses `meta/llama-3.1-405b-instruct` for transcription and context-aware translation.

## Installation
1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. **Create and activate a conda environment**:
   ```bash
   conda create -n translation-api python=3.8
   conda activate translation-api
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set environment variables**:
   - Ensure `OPENAI_API_KEY` and `NVIDIA_API_KEY` are set in your environment.
   - **`OPENAI_API_KEY`** is required for speech-to-text and text-to-speech conversion (optional if only using text-to-text translation).
   - **`NVIDIA_API_KEY`** is required for text translation.

## API Endpoints
### 1. **/translate-audio/**
- **Description**: Transcribes and translates an uploaded audio file with optional context for enhanced accuracy.
- **Form Parameters**:
  - `file` (required): The audio file (e.g., MP3).
  - `target_language` (optional): The target language code (default: `es`).
  - `context_file` (optional): A text file providing context to improve translation.
- **Example cURL Command**:
  ```bash
  curl -X POST "http://localhost:8000/translate-audio/" \
       -H "Content-Type: multipart/form-data" \
       -F "file=@path/to/audio.mp3" \
       -F "target_language=es" \
       -F "context_file=@path/to/context.txt" \
       --output result.mp3
  ```

### 2. **/translate-text/**
- **Description**: Translates submitted text with the option to provide additional context for improved results.
- **Form Parameters**:
  - `text` (required): The input text to be translated.
  - `target_language` (optional): The target language code (default: `es`).
  - `context_file` (optional): A text file providing context to improve translation.
- **Example cURL Command**:
  ```bash
curl -X POST "http://localhost:8000/translate-text/" \
     -H "Content-Type: multipart/form-data" \
     -F "text=Hello, how are you?" \
     -F "target_language=es" \
     -F "context_file=@path/to/context.txt"
  ```

## Testing
To test the endpoints, you can use the provided example cURL commands or use tools like Postman.

## Dependencies
- `fastapi`
- `uvicorn`
- `openai` (if relevant to the integration)
- `whisper`
- `requests`
- `llama-index-core`
- `llama-index-llms-openai`
- `llama-index-embeddings-huggingface`
- `transformers`

Ensure these dependencies are listed in `requirements.txt` for easy installation.

## Notes
- This project assumes that the dependencies are installed and correctly set up.
- The necessary API keys must be valid and have the necessary permissions for accessing the relevant services.
- **Audio-to-audio dubbing** requires both `OPENAI_API_KEY` and `NVIDIA_API_KEY` for full functionality.
- **Text-to-text translation** only requires `NVIDIA_API_KEY`, making `OPENAI_API_KEY` optional in this case.

## License
MIT.

