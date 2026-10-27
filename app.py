import streamlit as st
import torch
import numpy as np
import soundfile as sf
import io
import time
from peft import PeftConfig, PeftModel
from transformers import MusicgenForConditionalGeneration, AutoProcessor

# ─────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="MusicGen DreamBooth",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# Custom CSS — dark Indian-classical aesthetic
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700;900&family=DM+Sans:wght@300;400;500&display=swap');

:root {
    --bg:        #0a0a0f;
    --surface:   #12121a;
    --border:    #2a2a3a;
    --gold:      #c9a84c;
    --gold-dim:  #7a6230;
    --saffron:   #e8650a;
    --teal:      #2ec4b6;
    --text:      #e8e4d8;
    --muted:     #6b6880;
    --punk-red:  #e8102a;
    --punk-pink: #ff2d78;
}

html, body, [data-testid="stAppViewContainer"] {
    background-color: var(--bg) !important;
    font-family: 'DM Sans', sans-serif;
    color: var(--text);
}

[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border);
}

h1, h2, h3 {
    font-family: 'Playfair Display', serif !important;
}

/* Title */
.app-title {
    font-family: 'Playfair Display', serif;
    font-size: 2.6rem;
    font-weight: 900;
    letter-spacing: -0.02em;
    background: linear-gradient(135deg, var(--gold) 0%, var(--saffron) 60%, var(--punk-pink) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0;
    line-height: 1.1;
}
.app-subtitle {
    font-size: 0.85rem;
    color: var(--muted);
    letter-spacing: 0.18em;
    text-transform: uppercase;
    margin-top: 4px;
    margin-bottom: 2rem;
}

/* Mode cards */
.mode-card {
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1rem 1.2rem;
    background: var(--surface);
    margin-bottom: 0.5rem;
    transition: border-color 0.2s;
}
.mode-card.carnatic  { border-left: 3px solid var(--gold); }
.mode-card.punk      { border-left: 3px solid var(--punk-red); }
.mode-card.fusion    { border-left: 3px solid var(--teal); }

/* Generate button */
.stButton > button {
    background: linear-gradient(135deg, var(--gold) 0%, var(--saffron) 100%) !important;
    color: #0a0a0f !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.65rem 2rem !important;
    letter-spacing: 0.05em;
    width: 100%;
}
.stButton > button:hover {
    opacity: 0.88 !important;
}

/* Selectbox / radio / slider */
[data-testid="stSelectbox"] label,
[data-testid="stSlider"] label,
.stRadio label { color: var(--text) !important; font-size: 0.9rem; }

/* Audio player */
audio { width: 100%; border-radius: 8px; }

/* Section headers */
.section-label {
    font-size: 0.72rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: var(--gold-dim);
    margin-bottom: 0.4rem;
    margin-top: 1.2rem;
}

/* Tag pills */
.pill {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 20px;
    font-size: 0.72rem;
    margin: 2px;
    font-family: 'DM Sans', sans-serif;
}
.pill-gold   { background: #2a2210; color: var(--gold);     border: 1px solid var(--gold-dim); }
.pill-red    { background: #2a0a10; color: var(--punk-red); border: 1px solid #6a1020; }
.pill-teal   { background: #0a2220; color: var(--teal);     border: 1px solid #1a5250; }

/* Info box */
.info-box {
    background: #12121a;
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 0.8rem 1rem;
    font-size: 0.82rem;
    color: var(--muted);
    line-height: 1.6;
}

/* divider */
hr { border-color: var(--border) !important; }

/* Streamlit element tweaks */
[data-testid="stMarkdownContainer"] p { color: var(--text); }
.stAlert { border-radius: 8px; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Model loading — cached
# ─────────────────────────────────────────────
CARNATIC_REPO = "fahrahmn/saraga-dreambooth-musicgen"
PUNK_REPO     = "fahrahmn/musicgen-melody-lora_joel"

@st.cache_resource(show_spinner=False)
def load_model(repo_id: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device_gotcha", device)
    config = PeftConfig.from_pretrained(repo_id)
    model  = MusicgenForConditionalGeneration.from_pretrained(
        config.base_model_name_or_path,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
    )
    model     = PeftModel.from_pretrained(model, repo_id).to(device)
    processor = AutoProcessor.from_pretrained(repo_id)
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return model, processor, device


def audio_to_bytes(audio_np: np.ndarray, sr: int = 32000) -> bytes:
    buf = io.BytesIO()
    # Cast to float32 — soundfile does not support float16
    audio_np = audio_np.astype(np.float32)
    sf.write(buf, audio_np, sr, format="WAV")
    buf.seek(0)
    return buf.read()


def generate_audio(model, processor, device, prompt: str,
                   max_new_tokens: int, guidance_scale: float,
                   do_sample: bool = True) -> np.ndarray:
    inputs = processor(text=[prompt], padding=True, return_tensors="pt").to(device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            do_sample=do_sample,
            guidance_scale=guidance_scale,
            max_new_tokens=max_new_tokens,
        )
    # Always return float32 numpy array
    return output[0, 0].cpu().float().numpy()


def generate_chunked(model, processor, device, prompt: str,
                     total_tokens: int, chunk_tokens: int,
                     guidance_scale: float, crossfade_ms: int = 100) -> np.ndarray:
    """Chunked generation with crossfade to avoid hallucination on long clips."""
    sr = 32000
    overlap = int(sr * crossfade_ms / 1000)
    chunks  = []
    n       = max(1, total_tokens // chunk_tokens)

    prog = st.progress(0, text="Generating chunks…")
    for i in range(n):
        prog.progress((i) / n, text=f"Chunk {i+1} / {n}")
        chunk = generate_audio(model, processor, device, prompt,
                               chunk_tokens, guidance_scale)
        chunks.append(chunk)
    prog.progress(1.0, text="Stitching…")

    result = chunks[0]
    fade_out = np.linspace(1, 0, overlap)
    fade_in  = np.linspace(0, 1, overlap)
    for chunk in chunks[1:]:
        tail = result[-overlap:] * fade_out
        head = chunk[:overlap]   * fade_in
        result = np.concatenate([result[:-overlap], tail + head, chunk[overlap:]])

    prog.empty()
    return result


# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="app-title">MusicGen<br>DreamBooth</div>', unsafe_allow_html=True)
    st.markdown('<div class="app-subtitle">Style Transfer · Indian Classical × Western</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-label">Generation Mode</div>', unsafe_allow_html=True)
    mode = st.radio(
        "mode",
        ["🪘  Punk / Western", "🪕  Carnatic Classical", "🌀  Fusion Experiment"],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown('<div class="section-label">Generation Settings</div>', unsafe_allow_html=True)

    guidance = st.slider("Guidance Scale", 1.0, 10.0, 5.0, 0.5,
                         help="Higher = closer to prompt. Recommended: 3–5 for punk, 5–7 for Carnatic.")
    tokens   = st.slider("Max Tokens (length)", 64, 512, 256, 32,
                         help="256 ≈ 5 sec · 512 ≈ 10 sec · use Fusion for longer clips")

    st.markdown("---")
    st.markdown("""
    <div class="info-box">
    <b style="color:#c9a84c">Token → Duration</b><br>
    256 ≈ 5 sec &nbsp;·&nbsp; 512 ≈ 10 sec<br><br>
    <b style="color:#c9a84c">Models</b><br>
    Carnatic: <code>saraga-dreambooth-musicgen</code><br>
    Punk: <code>musicgen-melody-lora-punk-colab</code><br><br>
    <b style="color:#c9a84c">sks tokens</b><br>
    <code>sks0</code> → Flute (Saraga)<br>
    <code>sks1</code> → Veena (Ramana Radhakrishnan)
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Device indicator in sidebar
    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    if device_type == "cuda":
        st.success(f"🟢 GPU: {torch.cuda.get_device_name(0)}")
    else:
        st.warning("🟡 Running on GPU — generation will be slow")

    st.caption("Final Year Project · Dept. of CS · 2025–26")


# ─────────────────────────────────────────────
# Main area
# ─────────────────────────────────────────────

# ── PUNK MODE ──────────────────────────────────────────────────
if "Punk" in mode:
    st.markdown("## 🪘 Punk & Western Generation")
    st.markdown('<span class="pill pill-red">LoRA Fine-tuned</span> <span class="pill pill-red">musicgen-melody-lora-punk-colab</span>', unsafe_allow_html=True)
    st.markdown("")

    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown('<div class="section-label">Prompt</div>', unsafe_allow_html=True)
        punk_preset = st.selectbox("Quick Presets", [
            "Custom…",
            "80s punk and pop track with bassy drums and synth",
            "aggressive punk guitar riff with distortion and fast drums",
            "new wave synth pop with driving bassline and reverb vocals",
            "classic rock power chords with stadium drums",
            "post-punk dark atmosphere with jangly guitar and echo",
        ])
        if punk_preset == "Custom…":
            prompt = st.text_area("Describe your track", placeholder="e.g. lo-fi punk with fuzzy bass and shouted vocals", height=80)
        else:
            prompt = punk_preset
            st.text_area("Prompt (editable)", value=prompt, height=80, key="punk_edit")
            prompt = st.session_state.get("punk_edit", prompt)

    with col2:
        st.markdown('<div class="section-label">Style Tags</div>', unsafe_allow_html=True)
        st.markdown("""
        <span class="pill pill-red">punk</span>
        <span class="pill pill-red">distortion</span>
        <span class="pill pill-red">synth</span>
        <span class="pill pill-red">bassy drums</span>
        <span class="pill pill-red">new wave</span>
        <span class="pill pill-red">lo-fi</span>
        <span class="pill pill-red">power chords</span>
        """, unsafe_allow_html=True)

    st.markdown("")
    if st.button("⚡ Generate Punk Track"):
        if not prompt.strip():
            st.warning("Please enter a prompt.")
        else:
            with st.spinner("Loading Punk model…"):
                model, processor, device = load_model(PUNK_REPO)
            with st.spinner("🎸 Generating…"):
                t0    = time.time()
                audio = generate_audio(model, processor, device, prompt, tokens, guidance)
                dt    = time.time() - t0
            st.success(f"Generated in {dt:.1f}s · {len(audio)/32000:.1f} sec of audio")
            st.audio(audio_to_bytes(audio), format="audio/wav")
            st.download_button("⬇ Download WAV", audio_to_bytes(audio),
                               file_name="punk_output.wav", mime="audio/wav")

# ── CARNATIC MODE ──────────────────────────────────────────────
elif "Carnatic" in mode:
    st.markdown("## 🪕 Carnatic Classical Generation")
    st.markdown('<span class="pill pill-gold">DreamBooth</span> <span class="pill pill-gold">saraga-dreambooth-musicgen</span> <span class="pill pill-gold">Saraga Dataset</span>', unsafe_allow_html=True)
    st.markdown("")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown('<div class="section-label">Instrument</div>', unsafe_allow_html=True)
        instrument = st.selectbox("Instrument", [
            "sks1 — Veena (Ramana Radhakrishnan style)",
            "sks0 — Flute (Saraga timbre)",
            "Generic — No sks token (base model)",
        ])

        token_map = {
            "sks1 — Veena (Ramana Radhakrishnan style)": "sks1",
            "sks0 — Flute (Saraga timbre)":              "sks0",
            "Generic — No sks token (base model)":       "",
        }
        sks = token_map[instrument]

    with col2:
        st.markdown('<div class="section-label">Style / Raga Mood</div>', unsafe_allow_html=True)
        style = st.selectbox("Style", [
            "Calm, Carnatic",
            "Hindustani",
            "Devotional, Bhajan",
            "Classical, Meditative",
            "Folk, Festive",
            "Custom…",
        ])

    if style == "Custom…":
        style = st.text_input("Describe mood / raga", placeholder="e.g. Raga Bhairavi, evening calm")

    full_prompt = f"{sks} {style}".strip()
    st.markdown(f'<div class="section-label">Final Prompt</div>', unsafe_allow_html=True)
    st.code(full_prompt, language=None)

    st.markdown("")
    if st.button("🎵 Generate Carnatic Music"):
        with st.spinner("Loading Carnatic model…"):
            model, processor, device = load_model(CARNATIC_REPO)
        with st.spinner("🪕 Generating…"):
            t0    = time.time()
            audio = generate_audio(model, processor, device, full_prompt, tokens, guidance)
            dt    = time.time() - t0
        st.success(f"Generated in {dt:.1f}s · {len(audio)/32000:.1f} sec of audio")
        st.audio(audio_to_bytes(audio), format="audio/wav")
        st.download_button("⬇ Download WAV", audio_to_bytes(audio),
                           file_name="carnatic_output.wav", mime="audio/wav")

    st.markdown("---")
    st.markdown("#### 🔁 A/B Comparison — sks vs Baseline")
    st.caption("Generate the same prompt with and without the sks token to hear what DreamBooth learned.")

    if st.button("Run A/B Comparison"):
        if not sks:
            st.warning("Select an sks instrument above to compare.")
        else:
            with st.spinner("Loading model…"):
                model, processor, device = load_model(CARNATIC_REPO)

            c1, c2 = st.columns(2)
            with c1:
                with st.spinner(f"Generating sks version…"):
                    a_sks = generate_audio(model, processor, device,
                                           full_prompt, tokens, guidance)
                st.markdown(f"**🟡 With `{sks}` token**")
                st.audio(audio_to_bytes(a_sks), format="audio/wav")

            with c2:
                baseline = style
                with st.spinner("Generating baseline…"):
                    a_base = generate_audio(model, processor, device,
                                            baseline, tokens, guidance)
                st.markdown("**⚪ Baseline (no sks)**")
                st.audio(audio_to_bytes(a_base), format="audio/wav")


# ── FUSION MODE ────────────────────────────────────────────────
else:
    st.markdown("## 🌀 Fusion Experiment — Long Music")
    st.markdown('<span class="pill pill-teal">Chunked Generation</span> <span class="pill pill-teal">Crossfade Stitching</span> <span class="pill pill-teal">Experimental</span>', unsafe_allow_html=True)
    st.markdown("")

    st.info("⚠️ Long generation is experimental. Chunked mode avoids hallucination by generating short clips and crossfading them. Expect some discontinuity at boundaries.", icon="🔬")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown('<div class="section-label">Carnatic Element</div>', unsafe_allow_html=True)
        fusion_instrument = st.selectbox("Indian Style", [
            "sks1 Veena, Carnatic, Calm",
            "sks0 Flute, Carnatic, Meditative",
            "sks1 Veena, Hindustani",
            "sks1 Veena, Devotional",
        ])
        fusion_model_choice = st.radio("Model for Carnatic part", ["Carnatic DreamBooth", "Punk LoRA (cross-style)"])

    with col2:
        st.markdown('<div class="section-label">Western Element</div>', unsafe_allow_html=True)
        western_style = st.selectbox("Western Style", [
            "ambient synth pads",
            "soft electronic beats",
            "minimal guitar arpeggios",
            "lo-fi hip hop drums",
            "orchestral strings",
        ])

    fusion_prompt = f"{fusion_instrument} with {western_style}"
    st.markdown('<div class="section-label">Fusion Prompt</div>', unsafe_allow_html=True)
    st.code(fusion_prompt, language=None)

    custom_fusion = st.text_input("Or write your own fusion prompt", placeholder="sks1 Veena Carnatic with soft jazz brushes")
    if custom_fusion.strip():
        fusion_prompt = custom_fusion

    st.markdown('<div class="section-label">Long Generation Settings</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        total_tokens  = st.selectbox("Total Length", [512, 768, 1024, 1500], index=1,
                                     help="512≈10s · 768≈15s · 1024≈20s · 1500≈30s")
    with c2:
        chunk_tokens  = st.selectbox("Chunk Size", [128, 256, 384], index=1,
                                     help="256 is the safe window per chunk")
    with c3:
        crossfade_ms  = st.slider("Crossfade (ms)", 50, 500, 150, 50)

    st.markdown("")
    if st.button("🌀 Generate Fusion Track"):
        repo = CARNATIC_REPO if "Carnatic" in fusion_model_choice else PUNK_REPO
        with st.spinner("Loading model…"):
            model, processor, device = load_model(repo)
        t0    = time.time()
        audio = generate_chunked(model, processor, device,
                                 fusion_prompt, total_tokens, chunk_tokens,
                                 guidance, crossfade_ms)
        dt    = time.time() - t0
        st.success(f"Generated in {dt:.1f}s · {len(audio)/32000:.1f} sec of audio")
        st.audio(audio_to_bytes(audio), format="audio/wav")
        st.download_button("⬇ Download WAV", audio_to_bytes(audio),
                           file_name="fusion_output.wav", mime="audio/wav")


# ─────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align:center; color:#3a3850; font-size:0.78rem; padding: 0.5rem 0 1rem;">
    MusicGen DreamBooth &nbsp;·&nbsp; Style Transfer in Indian Classical Music &nbsp;·&nbsp;
    Final Year Project &nbsp;·&nbsp; Built with 🤗 Transformers + PEFT + Streamlit
</div>
""", unsafe_allow_html=True)