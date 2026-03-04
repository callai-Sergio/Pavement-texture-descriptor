"""
TextureLab – AI Assistant page (Streamlit).

Provides three tabs:
  1. Draft Generator – create project proposals / reports from form inputs.
  2. Results Interpreter – interpret texture measurement data.
  3. Chat – interactive chat with project context.

Requires Ollama running locally (http://localhost:11434).
"""
import sys
import json
from pathlib import Path

import streamlit as st
import pandas as pd

# ── Ensure project root is on the path ────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from texturelab_llm import (
    LLMConfig,
    check_ollama_health,
    list_models,
    generate_project_draft,
    interpret_results,
    chat_turn,
)


# ===================================================================
# Session state helpers
# ===================================================================

def _ensure_ai_state():
    """Initialise AI Assistant session state keys."""
    if "ai_chat_history" not in st.session_state:
        st.session_state["ai_chat_history"] = []
    if "ai_draft_output" not in st.session_state:
        st.session_state["ai_draft_output"] = ""
    if "ai_interp_output" not in st.session_state:
        st.session_state["ai_interp_output"] = ""


def _build_config(model: str, temperature: float, stream: bool) -> LLMConfig:
    """Build an LLMConfig from sidebar / form values."""
    return LLMConfig(
        model=model,
        temperature=temperature,
        stream_default=stream,
    )


# ===================================================================
# Page renderer (called from app.py router)
# ===================================================================

def page_ai_assistant():
    """Render the AI Assistant page."""
    _ensure_ai_state()

    st.markdown("## 🤖 AI Assistant")
    st.caption(
        "Local LLM integration via **Ollama**. "
        "All processing happens on your machine – no data is sent externally."
    )

    # ── Health check ──────────────────────────────────────────────────
    healthy = check_ollama_health(LLMConfig())
    if not healthy:
        st.error("⚠️ **Ollama is not reachable** at `http://localhost:11434`.")
        st.markdown("""
**To get started:**

1. **Install Ollama** → [ollama.com/download](https://ollama.com/download)
2. **Start the server:**
   ```bash
   ollama serve
   ```
3. **Pull a model:**
   ```bash
   ollama pull gemma3
   ```
4. **Reload this page** once Ollama is running.
        """)
        return

    # ── Model selector (sidebar-like) in columns ─────────────────────
    st.success("✅ Ollama is running")

    col_m, col_t, col_s = st.columns([2, 1, 1])
    available_models = list_models(LLMConfig())
    if not available_models:
        available_models = ["gemma3", "gemma2:2b"]

    with col_m:
        default_idx = 0
        for i, m in enumerate(available_models):
            if m.startswith("gemma3"):
                default_idx = i
                break
        selected_model = st.selectbox(
            "🧠 Model",
            available_models,
            index=default_idx,
            key="ai_model",
        )
    with col_t:
        temperature = st.slider(
            "🌡️ Temperature",
            0.0, 1.0, 0.2, 0.05,
            key="ai_temp",
        )
    with col_s:
        use_stream = st.toggle("⚡ Stream", value=False, key="ai_stream")

    st.markdown("---")

    # ── Tabs ──────────────────────────────────────────────────────────
    tab_draft, tab_interp, tab_chat = st.tabs([
        "📝 Draft Generator",
        "📊 Results Interpreter",
        "💬 Chat",
    ])

    # ==================================================================
    # TAB 1 – Draft Generator
    # ==================================================================
    with tab_draft:
        st.markdown("### Generate a Project Draft")
        st.caption(
            "Fill in the fields below and click **Generate**. "
            "The AI will produce a structured proposal draft."
        )

        with st.form("draft_form"):
            title = st.text_input(
                "Title / Topic",
                placeholder="AI-driven pavement texture optimization for airport runways",
            )
            context = st.text_area(
                "Context / Background",
                placeholder="Describe the problem, existing gaps, and motivation...",
                height=100,
            )
            objectives = st.text_area(
                "Objectives (one per line)",
                placeholder=(
                    "Develop a real-time texture classification system\n"
                    "Validate against ISO 13473-1 reference measurements\n"
                    "Deploy a prototype on 3 test sections"
                ),
                height=100,
            )
            methodology = st.text_area(
                "Methodology (one per line)",
                placeholder=(
                    "3D laser scanning of pavement surfaces\n"
                    "Machine learning model training on labelled data\n"
                    "Field validation campaign"
                ),
                height=100,
            )
            col_p, col_b = st.columns(2)
            with col_p:
                partners = st.text_input(
                    "Partners",
                    placeholder="University X, Company Y, Road Authority Z",
                )
            with col_b:
                budget = st.text_input(
                    "Budget",
                    placeholder="€500,000 over 3 years",
                )

            col_c1, col_c2, col_c3 = st.columns(3)
            with col_c1:
                pages = st.number_input("Target pages", 1, 50, 2, key="draft_pages")
            with col_c2:
                tone = st.selectbox(
                    "Tone",
                    ["formal", "semi-formal", "technical"],
                    key="draft_tone",
                )
            with col_c3:
                language = st.selectbox(
                    "Language",
                    ["English", "Portuguese", "Italian", "Spanish", "French", "German"],
                    key="draft_lang",
                )

            submitted = st.form_submit_button(
                "🚀 Generate Draft", use_container_width=True
            )

        if submitted and title:
            inputs = {
                "title": title,
                "context": context,
                "objectives": objectives,
                "methodology": methodology,
                "partners": partners,
                "budget": budget,
                "constraints": {
                    "pages": pages,
                    "tone": tone,
                    "language": language,
                },
            }
            config = _build_config(selected_model, temperature, use_stream)

            try:
                if use_stream:
                    placeholder = st.empty()
                    full_text = ""
                    for chunk in generate_project_draft(config, inputs, stream=True):
                        full_text += chunk
                        placeholder.markdown(full_text + "▌")
                    placeholder.markdown(full_text)
                    st.session_state["ai_draft_output"] = full_text
                else:
                    with st.spinner("Generating draft…"):
                        result = generate_project_draft(config, inputs, stream=False)
                    st.session_state["ai_draft_output"] = result
            except (ConnectionError, RuntimeError, TimeoutError) as exc:
                st.error(f"❌ {exc}")

        # Show output
        if st.session_state.get("ai_draft_output"):
            st.markdown("---")
            st.markdown("### 📄 Generated Draft")
            st.markdown(st.session_state["ai_draft_output"])
            st.download_button(
                "⬇️ Download as .md",
                st.session_state["ai_draft_output"],
                file_name="draft.md",
                mime="text/markdown",
            )

    # ==================================================================
    # TAB 2 – Results Interpreter
    # ==================================================================
    with tab_interp:
        st.markdown("### Interpret Texture Results")
        st.caption(
            "Paste a JSON of metrics **or** select a processed DataFrame "
            "from the current session."
        )

        source = st.radio(
            "Data source",
            ["Paste JSON", "Session data (aggregated results)"],
            horizontal=True,
            key="interp_source",
        )

        metrics_data = None

        if source == "Paste JSON":
            json_text = st.text_area(
                "Metrics JSON",
                placeholder=json.dumps({
                    "MPD_mm": 1.2,
                    "RMS_mm": 0.35,
                    "Skewness": -0.12,
                    "Kurtosis": 3.1,
                    "Notes": "Dry measurements, 1m segment",
                }, indent=2),
                height=200,
                key="interp_json",
            )
            if json_text.strip():
                try:
                    metrics_data = json.loads(json_text)
                except json.JSONDecodeError as e:
                    st.warning(f"Invalid JSON: {e}")
        else:
            # Try to get aggregated results from session
            agg = st.session_state.get("aggregated", {})
            if agg:
                fname = st.selectbox(
                    "Select file",
                    list(agg.keys()),
                    key="interp_file",
                )
                if fname:
                    merged = dict(agg[fname])
                    areal = st.session_state.get("results_areal", {}).get(fname, {})
                    merged.update(areal)
                    metrics_data = merged
                    st.json(metrics_data)
            else:
                st.info(
                    "No processed data in session. "
                    "Run an analysis first, or use **Paste JSON**."
                )

        interp_ctx = st.text_input(
            "Additional context (optional)",
            placeholder="SMA 11 surface, 3D laser scan at 0.011 mm resolution",
            key="interp_ctx",
        )

        if st.button("🔍 Interpret", use_container_width=True, key="btn_interp"):
            if metrics_data is None:
                st.warning("Please provide metrics data first.")
            else:
                config = _build_config(selected_model, temperature, use_stream)
                ctx = {"description": interp_ctx} if interp_ctx else None

                try:
                    if use_stream:
                        placeholder = st.empty()
                        full_text = ""
                        for chunk in interpret_results(
                            config, metrics_data, context=ctx, stream=True
                        ):
                            full_text += chunk
                            placeholder.markdown(full_text + "▌")
                        placeholder.markdown(full_text)
                        st.session_state["ai_interp_output"] = full_text
                    else:
                        with st.spinner("Interpreting results…"):
                            result = interpret_results(
                                config, metrics_data, context=ctx, stream=False
                            )
                        st.session_state["ai_interp_output"] = result
                except (ConnectionError, RuntimeError, TimeoutError) as exc:
                    st.error(f"❌ {exc}")

        if st.session_state.get("ai_interp_output"):
            st.markdown("---")
            st.markdown("### 📋 Interpretation")
            st.markdown(st.session_state["ai_interp_output"])

    # ==================================================================
    # TAB 3 – Chat
    # ==================================================================
    with tab_chat:
        st.markdown("### Chat with AI Assistant")

        # Context settings
        with st.expander("⚙️ Chat context", expanded=False):
            col_cx1, col_cx2, col_cx3 = st.columns(3)
            with col_cx1:
                chat_project = st.text_input(
                    "Project context",
                    placeholder="Pavement texture analysis of highway A1",
                    key="chat_project",
                )
            with col_cx2:
                chat_standard = st.text_input(
                    "Target standard",
                    placeholder="ISO 13473-1",
                    key="chat_standard",
                )
            with col_cx3:
                chat_language = st.selectbox(
                    "Response language",
                    ["English", "Portuguese", "Italian", "Spanish", "French", "German"],
                    key="chat_language",
                )

        # Display history
        for msg in st.session_state["ai_chat_history"]:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        # Chat input
        user_input = st.chat_input(
            "Ask about pavement textures, ISO standards, or your data…",
            key="chat_input",
        )

        if user_input:
            # Show user message
            st.session_state["ai_chat_history"].append(
                {"role": "user", "content": user_input}
            )
            with st.chat_message("user"):
                st.markdown(user_input)

            # Build context
            ctx = {}
            if chat_project:
                ctx["project"] = chat_project
            if chat_standard:
                ctx["standard"] = chat_standard
            if chat_language:
                ctx["language"] = chat_language

            config = _build_config(selected_model, temperature, use_stream)

            try:
                with st.chat_message("assistant"):
                    if use_stream:
                        placeholder = st.empty()
                        full_text = ""
                        for chunk in chat_turn(
                            config,
                            st.session_state["ai_chat_history"][:-1],
                            user_input,
                            context=ctx or None,
                            stream=True,
                        ):
                            full_text += chunk
                            placeholder.markdown(full_text + "▌")
                        placeholder.markdown(full_text)
                    else:
                        with st.spinner("Thinking…"):
                            full_text = chat_turn(
                                config,
                                st.session_state["ai_chat_history"][:-1],
                                user_input,
                                context=ctx or None,
                                stream=False,
                            )
                        st.markdown(full_text)

                st.session_state["ai_chat_history"].append(
                    {"role": "assistant", "content": full_text}
                )
            except (ConnectionError, RuntimeError, TimeoutError) as exc:
                st.error(f"❌ {exc}")

        # Clear chat button
        if st.session_state["ai_chat_history"]:
            if st.button("🗑️ Clear chat history", key="btn_clear_chat"):
                st.session_state["ai_chat_history"] = []
                st.rerun()
