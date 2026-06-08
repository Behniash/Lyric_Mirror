import numpy as np
import librosa
import librosa.display
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def extract_audio_features(audio_path: str) -> dict:
    """
    Extracts low-level acoustic features from an audio file.

    Returns a dict with tempo, energy, valence proxy, key, mode,
    and a sonic mood label.
    """
    print(f"Loading audio: {audio_path}")
    y, sr = librosa.load(audio_path, mono=True)

    # Tempo
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    tempo = round(float(np.asarray(tempo).flat[0]), 1)

    # Energy (RMS normalized)
    rms = librosa.feature.rms(y=y)[0]
    energy_norm = round(float(np.mean(rms) / (np.max(rms) + 1e-9)), 4)

    # Brightness (spectral centroid)
    spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    brightness = round(float(np.mean(spectral_centroids)), 2)

    # Dominant key
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    dominant_pitch_class = int(np.argmax(np.mean(chroma, axis=1)))
    pitch_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    dominant_key = pitch_names[dominant_pitch_class]

    # Major vs minor (valence proxy)
    major_profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
                               2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
    minor_profile = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53,
                               2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
    chroma_mean = np.mean(chroma, axis=1)
    major_corr = float(np.corrcoef(chroma_mean, major_profile)[0, 1])
    minor_corr = float(np.corrcoef(chroma_mean, minor_profile)[0, 1])
    mode = "major" if major_corr > minor_corr else "minor"
    valence = round((major_corr + 1) / 2, 4)

    # Sonic mood label
    if tempo > 130 and energy_norm > 0.6:
        mood = "energetic"
    elif tempo > 110 and mode == "major":
        mood = "upbeat"
    elif tempo < 80 and mode == "minor":
        mood = "melancholic"
    elif mode == "minor" and energy_norm < 0.4:
        mood = "dark"
    else:
        mood = "balanced"

    return {
        "tempo_bpm": tempo,
        "energy": energy_norm,
        "brightness_hz": brightness,
        "dominant_key": dominant_key,
        "mode": mode,
        "valence": valence,
        "sonic_mood": mood,
    }


def plot_waveform_and_spectrum(audio_path: str, output_path: str = "audio_analysis.png") -> None:
    """
    Saves a 2-panel figure: waveform + mel spectrogram.
    """
    y, sr = librosa.load(audio_path, mono=True)

    fig, axes = plt.subplots(2, 1, figsize=(10, 6))
    fig.patch.set_facecolor("#0f0f0f")
    for ax in axes:
        ax.set_facecolor("#1a1a1a")

    librosa.display.waveshow(y, sr=sr, ax=axes[0], color="#a78bfa")
    axes[0].set_title("Waveform", color="white", fontsize=12)
    axes[0].tick_params(colors="gray")

    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    S_db = librosa.power_to_db(S, ref=np.max)
    img = librosa.display.specshow(S_db, sr=sr, x_axis="time",
                                   y_axis="mel", ax=axes[1], cmap="magma")
    axes[1].set_title("Mel Spectrogram", color="white", fontsize=12)
    axes[1].tick_params(colors="gray")
    fig.colorbar(img, ax=axes[1], format="%+2.0f dB")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"Audio plot saved to: {output_path}")


if __name__ == "__main__":
    AUDIO_FILE = "diary_of_dreams_she_and_her_darkness 128.mp3"  # change this to your audio file

    features = extract_audio_features(AUDIO_FILE)
    for key, value in features.items():
        print(f"{key:<20} {value}")

    plot_waveform_and_spectrum(AUDIO_FILE)
