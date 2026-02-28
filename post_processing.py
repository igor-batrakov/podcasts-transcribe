import os
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

def run_post_processing(transcript_text: str, config: dict) -> str:
    """
    Main entry point for LLM post-processing.
    Routes the request to the correct provider based on config.
    """
    pp_config = config.get("post_processing", {})
    if not pp_config.get("enabled", False):
        return transcript_text
        
    provider = pp_config.get("provider", "ollama").lower()
    model = pp_config.get("model", "llama3")
    prompt_template = pp_config.get("prompt_template", "Fix punctuation and format this transcript:\n\n{text}")
    
    print(f"\n🧠 [LLM] Post-processing transcript using '{provider}' (model: {model})...")
    
    try:
        if provider == "ollama":
            return process_with_ollama(transcript_text, model, prompt_template)
        elif provider == "openai":
            api_key = os.environ.get("OPENAI_API_KEY", "")
            return process_with_openai(transcript_text, model, prompt_template, api_key)
        elif provider == "anthropic":
            api_key = os.environ.get("ANTHROPIC_API_KEY", "")
            return process_with_anthropic(transcript_text, model, prompt_template, api_key)
        else:
            print(f"⚠️ [LLM] Unknown provider '{provider}'. Skipping post-processing.")
            return transcript_text
            
    except Exception as e:
        print(f"❌ [LLM] Post-processing failed: {e}")
        print("⚠️ [LLM] Returning original unformatted transcript.")
        return transcript_text
