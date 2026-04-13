"""LLM adapter that auto-detects and uses either Gemini or OpenRouter."""
from __future__ import annotations

from . import config


def generate(prompt: str) -> str:
    """Generate response using configured LLM (Gemini, OpenRouter, or Ollama)."""
    provider = config.LLM_PROVIDER
    if provider == "gemini":
        if not config.GEMINI_API_KEY:
            raise ValueError("LLM_PROVIDER=gemini but GEMINI_API_KEY is not set")
        return _call_gemini(prompt)
    if provider == "openrouter":
        if not config.OPENROUTER_API_KEY:
            raise ValueError("LLM_PROVIDER=openrouter but OPENROUTER_API_KEY is not set")
        return _call_openrouter(prompt)
    if provider == "ollama":
        return _call_ollama(prompt)

    # Auto: prefer Gemini if configured, then OpenRouter, then Ollama
    if config.GEMINI_API_KEY:
        return _call_gemini(prompt)
    if config.OPENROUTER_API_KEY:
        return _call_openrouter(prompt)
    if config.OLLAMA_BASE_URL:
        return _call_ollama(prompt)
    raise ValueError("No LLM provider configured (set GEMINI_API_KEY, OPENROUTER_API_KEY, or OLLAMA_BASE_URL)")


def _call_gemini(prompt: str) -> str:
    """Call Google Gemini API."""
    import google.generativeai as genai

    genai.configure(api_key=config.GEMINI_API_KEY)
    model = genai.GenerativeModel(config.GEMINI_MODEL)
    resp = model.generate_content(prompt)
    return resp.text


def _call_openrouter(prompt: str) -> str:
    """Call OpenRouter API (OpenAI-compatible)."""
    from openai import OpenAI

    client = OpenAI(
        api_key=config.OPENROUTER_API_KEY,
        base_url=config.OPENROUTER_BASE_URL
    )
    resp = client.chat.completions.create(
        model=config.OPENROUTER_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1
    )
    return resp.choices[0].message.content


def _call_ollama(prompt: str) -> str:
    """Call Ollama-compatible API (local or remote via ngrok)."""
    import requests

    resp = requests.post(
        f"{config.OLLAMA_BASE_URL}/api/generate",
        headers={"ngrok-skip-browser-warning": "true"},
        json={
            "model": config.OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
        },
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["response"]
