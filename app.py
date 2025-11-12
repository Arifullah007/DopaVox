# app.py
# Parkinson's Voice Detector - Gradio app
# Robust, cross-version audio input, safe model loading, Parselmouth+librosa feature extraction,
# polished animated UI (percentage + emoji).

import os
import time
import traceback
import warnings
from typing import Tuple, Optional

import numpy as np
import joblib
import gradio as gr

# NOTE: we import librosa & parselmouth lazily inside functions so that the app can start even
# if building those heavy wheels fails initially. We'll try importing them where needed.
# However requirements.txt/postbuild should install them.

# ---------- Helper: create audio component that works across Gradio versions ----------
def make_audio_component(label: str = "Record or upload voice (WAV preferred)"):
    """
    Construct a gr.Audio element that supports both microphone+upload across different
    Gradio versions. Returns the gr.Audio component instance.
    """
    # Prefer the newest name 'sources' (recent releases), otherwise try 'source', finally fallback.
    try:
        return gr.Audio(sources=["microphone", "upload"], type="filepath", label=label)
    except TypeError:
        pass
    try:
        return gr.Audio(source=["microphone", "upload"], type="filepath", label=label)
    except TypeError:
        pass
    # Last-ditch fallback: upload-only (always supported)
    try:
        return gr.Audio(type="filepath", label=label)
    except Exception as e:
        raise RuntimeError("Failed to construct gr.Audio component: " + str(e))


# ---------- Model loading (safe, with fallbacks) ----------
MODEL_PATH = "pipeline_calibrated.pkl"

def load_model_safe(path: str):
    """
    Try to load the sklearn pipeline with joblib. If that fails try dill,
    and if that also fails return a lightweight dummy model that gives 50/50.
    """
    if not os.path.exists(path):
        print(f"[WARN] Model file not found at {path} - running with dummy fallback.")
        return None

    # Suppress sklearn cross-version warnings (we'll still log exceptions)
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)

    # 1) Try joblib (typical)
    try:
        model = joblib.load(path)
        print(f"[INFO] Loaded model via joblib from {path}")
        return model
    except Exception as e_joblib:
        print("[WARN] joblib.load failed:", repr(e_joblib))
        traceback.print_exc()

    # 2) Try dill if available
    try:
        import dill
        with open(path, "rb") as f:
            model = dill.load(f)
        print(f"[INFO] Loaded model via dill from {path}")
        return model
    except Exception as e_dill:
        print("[WARN] dill load failed (or dill not present):", repr(e_dill))
        traceback.print_exc()

    # 3) If all fails - return None (app will use fallback dummy)
    print("[ERROR] Failed to load the model with joblib/dill. App will run with a safe dummy predictor.")
    return None


class DummyModel:
    """Simple fallback model that returns 50/50 probability and can be replaced later."""
    def predict_proba(self, X):
        # Return balanced probabilities
        n = X.shape[0]
        return np.tile(np.array([0.5, 0.5]), (n, 1))


# Load model (or fallback)
_model = load_model_safe(MODEL_PATH)
if _model is None:
    model = DummyModel()
else:
    model = _model


# ---------- Feature extraction ----------
def extract_features_from_file(path: str, sr_target: int = 16000) -> Optional[np.ndarray]:
    """
    Load WAV (filepath) and extract features matching the UCI Parkinson dataset:
      - mean F0 (Hz)
      - jitter (local)
      - shimmer (local)
      - HNR (harmonicity)
      - MFCC means (13)
    Returns numpy array shape (1, n_features) or None on failure.
    """
    # Lazy imports
    try:
        import librosa
    except Exception as e:
        print("[ERROR] librosa import failed:", e)
        traceback.print_exc()
        return None

    # parlselmouth import inside function (can be optional)
    have_parselmouth = True
    try:
        import parselmouth
        from parselmouth.praat import call
    except Exception as e:
        have_parselmouth = False
        print("[WARN] parselmouth import failed, using limited features:", e)

    try:
        # Load audio as mono, resampled
        y, sr = librosa.load(path, sr=sr_target, mono=True)
        if y is None or y.size == 0:
            print("[ERROR] librosa.load returned empty audio")
            return None
        # Normalize amplitude
        if np.max(np.abs(y)) > 0:
            y = y / np.max(np.abs(y))
        # Trim silence
        y_trim, _ = librosa.effects.trim(y, top_db=30)
        if y_trim.size == 0:
            print("[ERROR] audio trimmed to zero length")
            return None
    except Exception as e:
        print("[ERROR] Failed to load/process audio with librosa:", e)
        traceback.print_exc()
        return None

    # Defaults
    mean_f0 = 0.0
    jitter_val = 0.0
    shimmer_val = 0.0
    hnr_val = 0.0

    # If parselmouth available, compute these
    if have_parselmouth:
        try:
            import parselmouth
            from parselmouth.praat import call

            snd = parselmouth.Sound(y_trim, sampling_frequency=sr)
            # Pitch (F0)
            pitch = snd.to_pitch()
            freqs = pitch.selected_array['frequency']
            freqs = freqs[freqs > 0]
            if freqs.size > 0:
                mean_f0 = float(np.mean(freqs))
            else:
                mean_f0 = 0.0

            # PointProcess required for jitter/shimmer
            pp = call(snd, "To PointProcess (periodic, cc)", 75, 500)

            # jitter local
            try:
                jitter_val = float(call(pp, "Get jitter (local)", 0.0001, 0.02, 0.02, 1.3))
            except Exception:
                jitter_val = 0.0

            # shimmer local: note the call signature expects [sound, point_process]
            try:
                shimmer_val = float(call([snd, pp], "Get shimmer (local)", 0.0001, 0.02, 1.3, 1.6))
            except Exception:
                shimmer_val = 0.0

            # Harmonics-to-noise ratio
            try:
                hnr_val = float(call(snd, "Get harmonicity (cc)", 0.01, 75, 0.1, 1.0))
            except Exception:
                hnr_val = 0.0

        except Exception as e:
            print("[WARN] parselmouth feature extraction failed:", e)
            traceback.print_exc()
            # keep defaults and continue

    # MFCCs via librosa (13 coefficients average)
    try:
        import librosa
        mfccs = librosa.feature.mfcc(y=y_trim, sr=sr, n_mfcc=13)
        mfccs_mean = np.mean(mfccs, axis=1)
        mfccs_mean = np.nan_to_num(mfccs_mean).tolist()
    except Exception as e:
        print("[WARN] MFCC extraction failed:", e)
        traceback.print_exc()
        mfccs_mean = [0.0] * 13

    # Build feature vector (order must match training pipeline)
    features = [mean_f0, jitter_val, shimmer_val, hnr_val] + mfccs_mean
    features = np.array(features, dtype=float).reshape(1, -1)
    return features


# ---------- Prediction + UI output formatting ----------
def build_result_html(risk_percent: float) -> Tuple[str, str]:
    """
    Build an animated progress bar HTML and a textual percentage label with emoji.
    risk_percent: 0..100 (float)
    Returns: (bar_html, text_html)
    """
    # Clamp
    rp = float(max(0.0, min(100.0, risk_percent)))
    # Color scale: green -> yellow -> red
    # hue 120 (green) -> 60 (yellow) -> 0 (red)
    if rp <= 50:
        # green -> yellow interpolation
        hue = 120 - (rp / 50.0) * 60
    else:
        # yellow -> red interpolation
        hue = 60 - ((rp - 50.0) / 50.0) * 60
    text_color = f"hsl({hue:.0f}, 85%, 45%)"

    # Emoji/tone
    if rp < 30:
        emoji = "🟢"  # low risk
        tone = "Low risk — likely healthy voice"
    elif rp < 65:
        emoji = "🟡"  # moderate
        tone = "Moderate — consider clinical follow-up"
    else:
        emoji = "🔴"  # high
        tone = "High risk — seek medical evaluation"

    # Animated progress bar with CSS (smooth)
    bar_html = f"""
    <div style="width:100%; max-width:520px;">
      <div style="font-size:14px; color: #ddd; margin-bottom:6px;">Estimated PD risk {emoji}</div>
      <div style="position:relative; height:36px; background:linear-gradient(90deg,#222,#1b1b1b); border-radius:10px; padding:4px;">
        <div id="pd-inner" style="width:0%; height:100%; border-radius:6px; background: linear-gradient(90deg, rgba(76,175,80,0.95), rgba(244,67,54,0.95)); box-shadow: 0 4px 12px rgba(0,0,0,0.45); transition: width 1.6s ease-out;"></div>
        <div style="position:absolute; right:12px; top:6px; font-size:16px; color:#fff; font-weight:700;">{rp:.1f}%</div>
      </div>
      <div style="margin-top:8px; color:{text_color}; font-weight:700;">{tone}</div>
    </div>
    <script>
      setTimeout(function() {{
         const el = document.getElementById('pd-inner');
         if (el) el.style.width = '{rp:.1f}%';
      }}, 80);
    </script>
    """

    text_html = f"<div style='font-size:28px; color:{text_color}; font-weight:800; margin-top:6px;'>{rp:.1f}%</div>"
    return bar_html, text_html

def extract_features_from_file(path: str, sr_target: int = 16000) -> Optional[np.ndarray]:
    """
    Build a 22-element feature vector in the common UCI order (see docstring above).
    Uses parselmouth (Praat) for pitch/jitter/shimmer/HNR where available, and librosa for MFCCs
    but only keeps MFCCs out of the UCI set indirectly by using PRAAT measures instead.
    Features that are not straightforward to compute (RPDE, DFA, spread1/2, D2, PPE) are set to 0.0.
    """
    try:
        import librosa
    except Exception as e:
        print("[ERROR] librosa import failed:", e)
        traceback.print_exc()
        return None

    have_parselmouth = True
    try:
        import parselmouth
        from parselmouth.praat import call
    except Exception as e:
        have_parselmouth = False
        print("[WARN] parselmouth import failed (some features unavailable):", e)

    try:
        y, sr = librosa.load(path, sr=sr_target, mono=True)
        if y is None or y.size == 0:
            print("[ERROR] librosa.load returned empty audio")
            return None
        if np.max(np.abs(y)) > 0:
            y = y / np.max(np.abs(y))
        y_trim, _ = librosa.effects.trim(y, top_db=30)
        if y_trim.size == 0:
            print("[ERROR] audio trimmed to zero length")
            return None
    except Exception as e:
        print("[ERROR] Failed to load/process audio with librosa:", e)
        traceback.print_exc()
        return None

    # Default placeholders
    MDVP_Fo = 0.0
    MDVP_Fhi = 0.0
    MDVP_Flo = 0.0
    MDVP_Jitter_perc = 0.0
    MDVP_Jitter_Abs = 0.0
    MDVP_RAP = 0.0
    MDVP_PPQ = 0.0
    Jitter_DDP = 0.0
    MDVP_Shimmer = 0.0
    MDVP_Shimmer_dB = 0.0
    Shimmer_APQ3 = 0.0
    Shimmer_APQ5 = 0.0
    MDVP_APQ = 0.0
    Shimmer_DDA = 0.0
    NHR = 0.0
    HNR = 0.0

    if have_parselmouth:
        try:
            snd = parselmouth.Sound(y_trim, sampling_frequency=sr)
            # Pitch (F0) and statistics
            pitch = snd.to_pitch()
            freqs = pitch.selected_array['frequency']
            freqs = freqs[freqs > 0]
            if freqs.size > 0:
                MDVP_Fo = float(np.mean(freqs))
                MDVP_Fhi = float(np.percentile(freqs, 95))
                MDVP_Flo = float(np.percentile(freqs, 5))
            else:
                MDVP_Fo = MDVP_Fhi = MDVP_Flo = 0.0

            # PointProcess for jitter/shimmer calculations
            try:
                pp = call(snd, "To PointProcess (periodic, cc)", 75, 500)
            except Exception:
                pp = None

            # Jitter measures
            try:
                MDVP_Jitter_perc = float(call(pp, "Get jitter (local)", 0.0001, 0.02, 0.02, 1.3)) * 100.0
            except Exception:
                MDVP_Jitter_perc = 0.0
            try:
                MDVP_Jitter_Abs = float(call(pp, "Get jitter (local, absolute)", 0.0001, 0.02, 0.02, 1.3))
            except Exception:
                # fallback: set to jitter_local (abs) if available or zero
                MDVP_Jitter_Abs = 0.0
            try:
                MDVP_RAP = float(call(pp, "Get jitter (rap)", 0.0001, 0.02, 0.02, 1.3))
            except Exception:
                MDVP_RAP = 0.0
            try:
                MDVP_PPQ = float(call(pp, "Get jitter (ppq5)", 0.0001, 0.02, 0.02, 1.3))
            except Exception:
                MDVP_PPQ = 0.0
            try:
                Jitter_DDP = float(call(pp, "Get jitter (ddp)", 0.0001, 0.02, 0.02, 1.3))
            except Exception:
                Jitter_DDP = 0.0

            # Shimmer measures
            try:
                MDVP_Shimmer = float(call([snd, pp], "Get shimmer (local)", 0.0001, 0.02, 1.3, 1.6))
            except Exception:
                MDVP_Shimmer = 0.0
            try:
                MDVP_Shimmer_dB = float(call([snd, pp], "Get shimmer (dB)", 0.0001, 0.02, 1.3, 1.6))
            except Exception:
                MDVP_Shimmer_dB = 0.0
            try:
                Shimmer_APQ3 = float(call([snd, pp], "Get shimmer (apq3)", 0.0001, 0.02, 1.3, 1.6))
            except Exception:
                Shimmer_APQ3 = 0.0
            try:
                Shimmer_APQ5 = float(call([snd, pp], "Get shimmer (apq5)", 0.0001, 0.02, 1.3, 1.6))
            except Exception:
                Shimmer_APQ5 = 0.0
            try:
                MDVP_APQ = float(call([snd, pp], "Get shimmer (apq)", 0.0001, 0.02, 1.3, 1.6))
            except Exception:
                MDVP_APQ = 0.0
            try:
                Shimmer_DDA = float(call([snd, pp], "Get shimmer (dda)", 0.0001, 0.02, 1.3, 1.6))
            except Exception:
                Shimmer_DDA = 0.0

            # Harmonicity / HNR
            try:
                # 'Get harmonicity (cc)' often returns HNR-like values
                HNR = float(call(snd, "Get harmonicity (cc)", 0.01, 75, 0.1, 1.0))
            except Exception:
                HNR = 0.0
            try:
                # try a noise-to-harmonics call if present – else approximate NHR as inverse (small epsilon)
                NHR = float(call(snd, "Get noise-to-harmonics ratio", 0.01, 75, 0.1, 1.0))
            except Exception:
                # approximate (bounded) fallback: if HNR>0 then NHR ~= 1/HNR (scale safely)
                NHR = 1.0 / (HNR + 1e-8) if HNR > 0.0 else 0.0

        except Exception as e:
            print("[WARN] parselmouth extraction partially failed:", e)
            traceback.print_exc()

    # Features that are nontrivial and not computed above: RPDE, DFA, spread1, spread2, D2, PPE
    # Set to 0.0 placeholders so the feature vector length matches model expectation.
    RPDE = 0.0
    DFA = 0.0
    spread1 = 0.0
    spread2 = 0.0
    D2 = 0.0
    PPE = 0.0

    # Build final vector in UCI order (22 features)
    features = [
        MDVP_Fo,
        MDVP_Fhi,
        MDVP_Flo,
        MDVP_Jitter_perc,
        MDVP_Jitter_Abs,
        MDVP_RAP,
        MDVP_PPQ,
        Jitter_DDP,
        MDVP_Shimmer,
        MDVP_Shimmer_dB,
        Shimmer_APQ3,
        Shimmer_APQ5,
        MDVP_APQ,
        Shimmer_DDA,
        NHR,
        HNR,
        RPDE,
        DFA,
        spread1,
        spread2,
        D2,
        PPE,
    ]
    features = np.array(features, dtype=float).reshape(1, -1)
    return features



def analyze_audio(audio_path: str):
    """
    Full pipeline: validate input -> extract features -> predict -> return HTML + text outputs.
    """
    if not audio_path:
        return ("<div style='color:#ff6b6b; font-weight:bold;'>No audio provided — please upload or record a 5–10s sample.</div>", "")

    # Extract features
    features = extract_features_from_file(audio_path)
    if features is None:
        return ("<div style='color:#ff6b6b; font-weight:bold;'>Failed to extract audio features. See container logs.</div>", "")

    # Predict probability for positive class index 1
    try:
        proba = model.predict_proba(features)[0]
        # Find positive class probability – assume second column index 1 is PD
        if proba.shape[0] == 2:
            prob_pd = float(proba[1])
        else:
            # If only one output or unknown shape, fallback to mean of vector
            prob_pd = float(np.mean(proba))
    except Exception as e:
        print("[ERROR] model.predict_proba failed:", e)
        traceback.print_exc()
        return ("<div style='color:#ff6b6b; font-weight:bold;'>Prediction failed — check container logs.</div>", "")

    # Make percentage
    risk_percent = prob_pd * 100.0
    bar_html, text_html = build_result_html(risk_percent)
    return bar_html, text_html


# ---------- Gradio UI ----------
css = """
.app-title { font-weight:900; font-size:26px; color:#ffffff; }
.small-muted { color: #cfcfcf; font-size:13px; }
.big-button { padding: 12px 20px; border-radius:10px; }
"""

def build_ui():
    with gr.Blocks(css=css, theme=None, analytics_enabled=False) as demo:
        gr.Markdown("<div class='app-title'>🧠 Parkinson's Voice Detector</div>")
        gr.Markdown("<div class='small-muted'>Record or upload a ~5–10s sustained vowel or reading sample. Research demo — not medical advice.</div>")

        with gr.Row():
            with gr.Column(scale=6):
                audio_in = make_audio_component("🎙️ Record or upload voice sample (WAV preferred)")
                with gr.Row():
                    analyze_btn = gr.Button("Analyze Voice", variant="primary", elem_classes="big-button")
                    clear_btn = gr.Button("Clear", variant="secondary", elem_classes="big-button")
            with gr.Column(scale=6):
                gr.Markdown("### 📊 Prediction")
                result_bar = gr.HTML("<div style='color:#bbb; font-style:italic;'>Awaiting audio input...</div>")
                result_text = gr.HTML("")

        # Button actions
        def on_clear():
            return "<div style='color:#bbb; font-style:italic;'>Awaiting audio input...</div>", ""

        analyze_btn.click(fn=analyze_audio, inputs=audio_in, outputs=[result_bar, result_text])
        clear_btn.click(fn=on_clear, inputs=None, outputs=[result_bar, result_text])

        # Expose the demo
        return demo

demo = build_ui()

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", share=False)
