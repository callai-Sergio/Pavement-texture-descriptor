"""
TextureLabDesktop – AI Draft Demo

Minimal demonstration showing how the Desktop module can import and use
the ``texturelab_llm`` package to generate project drafts via Ollama.

Usage:
    cd Pavement-texture-descriptor
    python TextureLabDesktop/ai_draft_demo.py

Requires Ollama running locally with a model pulled (e.g. ``ollama pull gemma3``).
"""
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from texturelab_llm import (
    LLMConfig,
    check_ollama_health,
    list_models,
    generate_project_draft,
    interpret_results,
)


def main():
    print("=" * 60)
    print("  TextureLabDesktop – AI Draft Demo")
    print("=" * 60)

    config = LLMConfig()

    # 1. Health check
    print(f"\n[1] Checking Ollama at {config.ollama_base_url} ...")
    if not check_ollama_health(config):
        print("    ❌ Ollama is not reachable.")
        print("    → Install Ollama: https://ollama.com/download")
        print("    → Start: ollama serve")
        print("    → Pull model: ollama pull gemma3")
        return
    print("    ✅ Ollama is running")

    # 2. List models
    models = list_models(config)
    print(f"\n[2] Available models: {', '.join(models) if models else '(none)'}")

    # 3. Generate a draft
    print(f"\n[3] Generating project draft with model '{config.model}' ...")
    sample_inputs = {
        "title": "AI-driven pavement texture optimization for airport runways",
        "context": (
            "Airport runway friction is critical for aircraft safety. "
            "Surface texture directly influences wet skid resistance."
        ),
        "objectives": [
            "Develop a real-time texture classification system",
            "Validate against ISO 13473-1 reference measurements",
            "Deploy a prototype on 3 test runway sections",
        ],
        "methodology": [
            "3D laser scanning of runway surfaces",
            "Machine learning model training on labelled texture data",
            "Field validation campaign with BPN and GripTester",
        ],
        "partners": "University of Padova, ANAS S.p.A., ENAC",
        "budget": "€450,000 over 2 years",
        "constraints": {"pages": 2, "tone": "formal", "language": "English"},
    }

    result = generate_project_draft(config, sample_inputs, stream=False)
    print("\n" + "─" * 60)
    print(result)
    print("─" * 60)

    # 4. Interpret example metrics
    print(f"\n[4] Interpreting example metrics ...")
    sample_metrics = {
        "MPD_mm": 1.2,
        "RMS_mm": 0.35,
        "Skewness": -0.12,
        "Kurtosis": 3.1,
        "Sa_um": 28.5,
        "Sdr_pct": 12.3,
        "Notes": "Dry measurements, 1m segment, SMA 11",
    }
    interp = interpret_results(config, sample_metrics, stream=False)
    print("\n" + "─" * 60)
    print(interp)
    print("─" * 60)

    print("\n✅ Demo complete.")


if __name__ == "__main__":
    main()
