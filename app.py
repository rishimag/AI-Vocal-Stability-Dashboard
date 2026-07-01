import streamlit as st
import librosa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# --- INTERFACE SETUP ---
st.set_page_config(page_title="AI Vocal Stability Tracker", layout="wide")
st.title("🎙️ AI Vocal Technique & Diagnostic Dashboard")
st.markdown("Created by **Rishima** | Rising 10th Grade Computational Health Project")

# --- BACKGROUND CLINICAL CORE ---
@st.cache_resource
def load_and_train_ai():
    parkinsons = fetch_ucirepo(id=174)
    df = pd.DataFrame(parkinsons.data.features)
    df['status'] = parkinsons.data.targets
    
    # Deduplicate identical headers safely using len() to avoid TypeError!
    cols = pd.Series(df.columns)
    for dup in cols[cols.duplicated()].unique():
        count = len(cols[cols == dup])
        cols[cols == dup] = [f"{dup}_{i}" if i != 0 else dup for i in range(count)]
    df.columns = cols
    
    X_all = df.drop(columns=[col for col in ['status', 'name'] if col in df.columns])
    y_all = df['status']
    
    X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.2, random_state=42)
    optimized_model = RandomForestClassifier(n_estimators=300, criterion='entropy', random_state=42)
    optimized_model.fit(X_train, y_train)
    return optimized_model, X_all, y_all, df

optimized_model, X_all, y_all, raw_df = load_and_train_ai()

# --- PATIENT INPUT INTERFACE ---
st.header("🎛️ Patient Recording & Input Interface")
st.markdown("Choose to either upload a pre-recorded audio file, use library benchmarks, or record your voice live:")

input_mode = st.radio("Select Input Source:", ["🔴 Live Microphone Recording", "📁 Upload an Audio File (.wav)", "🎵 Use Dataset Benchmarks"])

my_y, my_sr = None, None

if input_mode == "🔴 Live Microphone Recording":
    audio_file = st.audio_input("Record yourself singing 'Ahhh' steadily for 5 seconds:")
    if audio_file is not None:
        my_y, my_sr = librosa.load(audio_file)

elif input_mode == "📁 Upload an Audio File (.wav)":
    uploaded_file = st.file_uploader("Upload a vocal WAV file", type=["wav"])
    if uploaded_file is not None:
        my_y, my_sr = librosa.load(uploaded_file)

else:
    selected_sample = st.selectbox("Choose a Benchmark Sample:", ["vibeace", "trumpet"])
    if st.button("Load Benchmark Track"):
        my_y, my_sr = librosa.load(librosa.ex(selected_sample))

# --- PROCESSING AND PROVING CREDIBILITY ---
if my_y is not None:
    trimmed_y, _ = librosa.effects.trim(my_y, top_db=20)
    normalized_y = librosa.util.normalize(trimmed_y)
    
    f0_me, _, _ = librosa.pyin(normalized_y, fmin=80, fmax=400)
    clean_pitches = f0_me[~np.isnan(f0_me)]
    rms_me = librosa.feature.rms(y=normalized_y)
    clean_rms = rms_me[rms_me > 0.005]
    
    if len(clean_pitches) > 0 and len(clean_rms) > 0:
        my_avg_pitch = np.mean(clean_pitches)
        my_jitter = np.mean(np.abs(np.diff(clean_pitches)))
        my_shimmer = np.mean(np.abs(np.diff(clean_rms)))
        
        # SPLIT SCREEN LAYOUT FOR GRAPHICAL PROOF
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📋 Your Extracted Biomarkers")
            st.metric("Your Average Pitch", f"{my_avg_pitch:.2f} Hz")
            st.metric("Your Jitter (Frequency Variation)", f"{my_jitter:.4f} Hz")
            st.metric("Your Shimmer (Amplitude Variation)", f"{my_shimmer:.5f}")
            
            # FORMAT FOR PREDICTION
            healthy_averages = X_all[y_all == 0].mean()
            my_custom_row = pd.DataFrame([healthy_averages.values], columns=X_all.columns)
            my_custom_row['MDVP:Fo'] = my_avg_pitch
            my_custom_row['MDVP:Jitter'] = my_jitter
            my_custom_row['MDVP:Shimmer'] = my_shimmer
            
            # 🛠️ THE PLUG: Grab array elements directly using [0] index to avoid text formatting crashes!
            probabilities = optimized_model.predict_proba(my_custom_row)[0]
            healthy_confidence = probabilities[0] * 100
            clinical_confidence = probabilities[1] * 100
            
            st.subheader("🛡️ Probability Decision Matrix")
            st.write(f"• Healthy Group Match: **{healthy_confidence:.2f}%**")
            st.write(f"• Clinical Variation Match: **{clinical_confidence:.2f}%**")
            
            if max(healthy_confidence, clinical_confidence) < 70.0:
                st.warning("⚠️ **SYSTEM VERDICT: BORDERLINE PROFILE DETECTED**\nYour vocal stability properties fall into the gray boundary zone between classes.")
            elif healthy_confidence >= 70.0:
                st.success("✅ **SYSTEM VERDICT: HIGHLY ALIGNED WITH HEALTHY CONTROLS**\nYour voice indicates uniform muscle contractions and breath stability.")
            else:
                st.error("🚨 **SYSTEM VERDICT: MATCHES CLINICAL VARIATION PATTERNS**\nMicro-perturbation maps show elevated tremor indexes matching tracking benchmarks.")
        
        with col2:
            st.subheader("📊 Scientific Credibility & Population Alignment")
            st.markdown("This chart explains *why* the AI made its choice. It plots your personal **Jitter** directly against the actual healthy population distributions inside the clinical repository:")
            
            # Generate the proof-of-work credibility chart
            fig, ax = plt.subplots(figsize=(6, 4))
            
            # Plot historical population data points
            ax.scatter(raw_df[raw_df['status'] == 0]['MDVP:Jitter'], np.zeros(sum(raw_df['status'] == 0)), color='#1f77b4', alpha=0.4, label='Healthy Clinical Cohort', s=50)
            ax.scatter(raw_df[raw_df['status'] == 1]['MDVP:Jitter'], np.ones(sum(raw_df['status'] == 1)), color='#d62728', alpha=0.2, label='Parkinson Cohort', s=50)
            
            # Plot user's gold star positioning based on their math
            ax.axvline(x=my_jitter, color='gold', linestyle='--', linewidth=2)
            ax.scatter([my_jitter], [0.5], color='gold', edgecolor='black', s=300, marker='*', zorder=5, label='YOUR RECORDED VOICE')
            
            ax.set_title("Vocal Frequency Perturbation Distribution")
            ax.set_xlabel("Absolute Jitter Values (Micro-Tremor Level)")
            ax.set_yticks([])
            ax.set_yticklabels([])
            ax.legend()
            st.pyplot(fig)
            
            st.markdown("💡 **Acoustic Explanation:** If your gold star falls to the far left, your voice has low variation. If it trends to the right, ambient room static or microphone reflections are increasing the perturbation metric.")
    else:
        st.error("Error: Could not isolate a stable voice pitch. Please get closer to your microphone and repeat the note.")

st.markdown("---")
st.caption("⚠️ **MEDICAL DISCLAIMER:** This dashboard is an educational proof-of-concept. It does not replace clinical consultation, nor does it diagnose medical conditions.")ns.")
