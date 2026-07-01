import streamlit as st
import librosa
import numpy as np
import pandas as pd
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# --- INTERFACE SETUP ---
st.title("🎙️ AI Vocal Technique & Stability Dashboard")
st.markdown("Created by **Rishima** | Rising 10th Grade STEM Project")

# --- DATASET AND AI LOADING ---
parkinsons = fetch_ucirepo(id=174)
df = pd.DataFrame(parkinsons.data.features)
df['status'] = parkinsons.data.targets

# 🛠️ THE PLUG: This automatically fixes the duplicate column name error!
cols = pd.Series(df.columns)
for dup in cols[cols.duplicated()].unique():
    cols[cols == dup] = [f"{dup}_{i}" if i != 0 else dup for i in range(cols[cols == dup].shape[0])]
df.columns = cols

X_all = df.drop(columns=[col for col in ['status', 'name'] if col in df.columns])
y_all = df['status']

# Train the exact 300-tree model you optimized today
X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.2, random_state=42)
optimized_model = RandomForestClassifier(n_estimators=300, criterion='entropy', random_state=42)
optimized_model.fit(X_train, y_train)

# --- TRACK SELECTOR PANEL ---
st.header("🎛️ Audio Input Panel")
selected_sample = st.selectbox("Choose a Voice Sample Track to Test:", ["vibeace", "trumpet"])

if st.button("🚀 Run AI Diagnosis"):
    audio_path = librosa.ex(selected_sample)
    my_y, my_sr = librosa.load(audio_path)
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
        
        # Display the report card using web metrics
        st.subheader("📋 Your Extracted Biomarkers")
        st.metric("Your Average Pitch", f"{my_avg_pitch:.2f} Hz")
        st.metric("Your Absolute Jitter", f"{my_jitter:.4f} Hz")
        st.metric("Your Absolute Shimmer", f"{my_shimmer:.5f}")
        
        # --- PROBABILISTIC SAFETY BUFFER LOGIC ---
        healthy_averages = X_all[y_all == 0].mean()
        my_custom_row = pd.DataFrame([healthy_averages.values], columns=X_all.columns)
        my_custom_row['MDVP:Fo'] = my_avg_pitch
        my_custom_row['MDVP:Jitter'] = my_jitter
        my_custom_row['MDVP:Shimmer'] = my_shimmer
        
        probabilities = optimized_model.predict_proba(my_custom_row)[0]
        healthy_confidence = probabilities[0] * 100
        clinical_confidence = probabilities[1] * 100
        
        st.subheader("🛡️ Probability Decision Matrix")
        st.write(f"• Healthy Confidence: **{healthy_confidence:.2f}%**")
        st.write(f"• Variation Confidence: **{clinical_confidence:.2f}%**")
        
        if max(healthy_confidence, clinical_confidence) < 70.0:
            st.warning("⚠️ **SYSTEM VERDICT: BORDERLINE PROFILE DETECTED**\nAcoustic data sits near the population split boundary.")
        elif healthy_confidence >= 70.0:
            st.success("✅ **SYSTEM VERDICT: HIGHLY ALIGNED WITH HEALTHY CONTROLS**\nVoice shows high stability and strong motor consistency.")
        else:
            st.error("🚨 **SYSTEM VERDICT: MATCHES CLINICAL VARIATION PATTERNS**\nMicro-tremor profiles align with patterns seen in tracking studies.")
    else:
        st.error("Error: Could not extract features from this audio file.")

st.markdown("---")
st.caption("⚠️ **MEDICAL DISCLAIMER:** This dashboard is an educational proof-of-concept. It does not replace clinical consultation, nor does it diagnose medical conditions.")
