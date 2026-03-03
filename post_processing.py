import os
import json
import requests
from openai import OpenAI
from anthropic import Anthropic

def process_with_ollama(text: str, model: str, prompt_template: str) -> str:
    """Send text to local Ollama instance."""
    url = "http://localhost:11434/api/generate"
    prompt = prompt_template.replace("{text}", text)
    
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False
    }
    
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        return response.json().get("response", "")
    except requests.exceptions.ConnectionError:
        raise ConnectionError(
            "Could not connect to Ollama. Make sure the Ollama app is running on your Mac "
            "and the model is downloaded (e.g. `ollama run llama3`)."
        )
    except Exception as e:
        raise Exception(f"Ollama API error: {e}")

def unload_ollama_model(model: str) -> None:
    """Force Ollama to unload the model from memory."""
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": model,
        "keep_alive": 0
    }
    try:
        # We don't need a prompt, just the keep_alive=0 to unload immediately.
        requests.post(url, json=payload, timeout=5)
        print(f"   🧹 [dim]Ollama model '{model}' explicitly unloaded from memory.[/dim]")
    except Exception as e:
        print(f"⚠️ [dim]Could not explicitly unload Ollama model: {e}[/dim]")


def process_with_openai(text: str, model: str, prompt_template: str, api_key: str) -> str:
    """Send text to OpenAI API."""
    if not api_key:
        raise ValueError("OpenAI API key is missing. Add it to your .env file as OPENAI_API_KEY")
        
    client = OpenAI(api_key=api_key)
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": prompt_template.replace("{text}", "")},
                {"role": "user", "content": text}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        raise Exception(f"OpenAI API error: {e}")

def process_with_anthropic(text: str, model: str, prompt_template: str, api_key: str) -> str:
    """Send text to Anthropic Claude API."""
    if not api_key:
        raise ValueError("Anthropic API key is missing. Add it to your .env file as ANTHROPIC_API_KEY")
        
    client = Anthropic(api_key=api_key)
    
    try:
        response = client.messages.create(
            model=model,
            max_tokens=4096,
            system=prompt_template.replace("{text}", ""),
            messages=[
                {"role": "user", "content": text}
            ]
        )
        return response.content[0].text
    except Exception as e:
        raise Exception(f"Anthropic API error: {e}")

def extract_podcast_metadata_with_llm(raw_text: str, config: dict) -> dict:
    """
    Extracts real speaker names and podcast metadata from the transcript context.
    Returns a dictionary with 'speakers' and 'metadata'.
    """
    pp_config = config.get("post_processing", {})
    provider = pp_config.get("provider", "ollama").lower()
    model = pp_config.get("model", "qwen2.5:3b")
    
    prompt = (
        "Analyze the following Russian transcript excerpt and extract metadata and speaker names. "
        "Identify the real names of the speakers labeled as GLOBAL_SPEAKER_1, GLOBAL_SPEAKER_2, etc. "
        "Also try to find the podcast's official show name, episode number, release date, and main topic. "
        "If you cannot find a specific piece of metadata, return null for it. "
        "Return the result ONLY as a valid JSON dictionary using the exact following schema:\n"
        "{\n"
        '  "speakers": {\n'
        '    "GLOBAL_SPEAKER_1": {"name": "Александр", "confidence": 95}\n'
        '  },\n'
        '  "metadata": {\n'
        '    "show_name": "Podcast Name",\n'
        '    "episode_number": "123",\n'
        '    "date": "December 12",\n'
        '    "topic": "Main discussion topic"\n'
        '  }\n'
        "}\n"
        "Where 'confidence' is an integer from 0 to 100 representing how sure you are based on the context. "
        "Do not include any other text.\n\n"
        f"Transcript:\n{raw_text}"
    )

    print(f"\n🔍 [LLM] Analyzing context for real speaker names using '{provider}' (model: {model})...")
    
    try:
        response_text = ""
        if provider == "ollama":
            response_text = process_with_ollama(prompt, model, "{text}")
        elif provider == "openai":
            api_key = os.environ.get("OPENAI_API_KEY", "")
            response_text = process_with_openai(prompt, model, "{text}", api_key)
        elif provider == "anthropic":
            api_key = os.environ.get("ANTHROPIC_API_KEY", "")
            response_text = process_with_anthropic(prompt, model, "{text}", api_key)
        else:
            return {}
            
        # Clean up possible markdown formatting from LLM response
        response_text = response_text.replace("```json", "").replace("```", "").strip()
        result_dict = json.loads(response_text)
        
        valid_data = {"speakers": {}, "metadata": {}}
        speakers_dict = result_dict.get("speakers", {})
        metadata_dict = result_dict.get("metadata", {})
        
        for k, v in speakers_dict.items():
            if isinstance(k, str) and k.startswith("GLOBAL_SPEAKER_") and isinstance(v, dict):
                name = v.get("name")
                conf = v.get("confidence", 0)
                if isinstance(name, str) and name.strip() and name != "Unknown":
                    valid_data["speakers"][k] = {"name": name.strip(), "confidence": conf}
                    
        valid_data["metadata"] = {
            "show_name": metadata_dict.get("show_name") if isinstance(metadata_dict.get("show_name"), str) else None,
            "episode_number": metadata_dict.get("episode_number") if isinstance(metadata_dict.get("episode_number"), str) else None,
            "date": metadata_dict.get("date") if isinstance(metadata_dict.get("date"), str) else None,
            "topic": metadata_dict.get("topic") if isinstance(metadata_dict.get("topic"), str) else None,
        }
                
        return valid_data
    except Exception as e:
        print(f"⚠️ [LLM] Name extraction failed or returned invalid JSON: {e}")
        return {"speakers": {}, "metadata": {}}


def run_post_processing(transcript_text: str, config: dict, is_single_speaker: bool = False) -> str:
    """
    Main entry point for LLM post-processing.
    Chunks the transcript to avoid context limits and routes to the correct provider.
    """
    pp_config = config.get("post_processing", {})
    if not pp_config.get("enabled", False):
        return transcript_text
        
    provider = pp_config.get("provider", "ollama").lower()
    model = pp_config.get("model", "llama3")
    
    if is_single_speaker:
        prompt_template = pp_config.get(
            "prompt_single_speaker", 
            "Format this single speaker transcript as an article:\n\n{text}"
        )
    else:
        prompt_template = pp_config.get(
            "prompt_multi_speaker", 
            "Fix punctuation and format this transcript:\n\n{text}"
        )
    
    chunk_size = int(pp_config.get("chunk_size_lines", 100))
    overlap = int(pp_config.get("overlap_lines", 10))
    
    lines = transcript_text.strip().split("\n")
    total_lines = len(lines)
    
    # If transcript is short, don't overcomplicate with chunking
    if total_lines <= chunk_size:
        print(f"\n🧠 [LLM] Post-processing transcript ({total_lines} lines) using '{provider}' (model: {model})...")
        try:
            return _process_chunk(transcript_text, provider, model, prompt_template)
        except Exception as e:
            print(f"❌ [LLM] Post-processing failed: {e}")
            return transcript_text

    # Chunking logic for long transcripts
    print(f"\n🧠 [LLM] Post-processing large transcript ({total_lines} lines) in chunks using '{provider}'...")
    
    final_output = []
    start_idx = 0
    chunk_index = 1
    
    while start_idx < total_lines:
        end_idx = min(start_idx + chunk_size, total_lines)
        chunk_lines = lines[start_idx:end_idx]
        chunk_text = "\n".join(chunk_lines)
        import sys
        sys.stdout.write(f"\n   ⏳ Processing chunk {chunk_index} (lines {start_idx+1}-{end_idx})... ")
        sys.stdout.flush()
        try:
            processed_chunk = _process_chunk(chunk_text, provider, model, prompt_template)
            sys.stdout.write("Done!\n")
            sys.stdout.flush()
            final_output.append(processed_chunk.strip())
        except Exception as e:
            print(f"❌ [LLM] Failed on chunk {chunk_index}: {e}")
            print("   ⚠️ Keeping original text for this chunk.")
            final_output.append(chunk_text)
            
        start_idx += (chunk_size - overlap)
        chunk_index += 1
        
    return "\n\n".join(final_output)


def _process_chunk(text: str, provider: str, model: str, prompt_template: str) -> str:
    """Helper to route a single chunk of text to the requested provider."""
    if provider == "ollama":
        return process_with_ollama(text, model, prompt_template)
    elif provider == "openai":
        api_key = os.environ.get("OPENAI_API_KEY", "")
        return process_with_openai(text, model, prompt_template, api_key)
    elif provider == "anthropic":
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        return process_with_anthropic(text, model, prompt_template, api_key)
    else:
        raise ValueError(f"Unknown provider '{provider}'")
