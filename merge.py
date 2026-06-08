import json
from nlp import run_nlp_analysis
from audio import extract_audio_features


THEME_COLORS = {
    "love":         "#f472b6",
    "pain":         "#818cf8",
    "hope":         "#34d399",
    "anger":        "#f87171",
    "nostalgia":    "#fb923c",
    "joy":          "#facc15",
    "unclassified": "#94a3b8",
}

SONIC_MOOD_COLORS = {
    "energetic":   "#f87171",
    "upbeat":      "#facc15",
    "melancholic": "#818cf8",
    "dark":        "#475569",
    "balanced":    "#34d399",
}


def merge_signals(transcript: dict, audio_path: str) -> dict:
    """
    Combines NLP analysis and audio feature extraction
    into a single emotional portrait dict.

    Args:
        transcript: output of week1_transcribe.transcribe_audio()
        audio_path: path to the same audio file

    Returns:
        portrait dict ready for visualisation or the Gradio UI
    """
    print("Running NLP pipeline...")
    nlp = run_nlp_analysis(transcript)

    print("Running audio pipeline...")
    audio = extract_audio_features(audio_path)

    sentiment_score = nlp["sentiment"]["score"]
    if nlp["sentiment"]["label"] == "NEGATIVE":
        sentiment_score = 1 - sentiment_score

    combined_valence = round((sentiment_score + audio["valence"]) / 2, 4)

    primary_theme = nlp["themes"][0] if nlp["themes"] else "unclassified"
    primary_color = THEME_COLORS.get(primary_theme, "#94a3b8")
    sonic_color = SONIC_MOOD_COLORS.get(audio["sonic_mood"], "#94a3b8")

    portrait = {
        "combined_valence": combined_valence,
        "primary_theme": primary_theme,
        "all_themes": nlp["themes"],
        "keywords": nlp["keywords"],
        "sentiment": nlp["sentiment"],
        "audio": audio,
        "colors": {
            "primary": primary_color,
            "sonic": sonic_color,
        },
        "summary": _build_summary(nlp, audio, combined_valence),
    }

    return portrait


def _build_summary(nlp: dict, audio: dict, valence: float) -> str:
    mood_word = "positive" if valence > 0.55 else "negative" if valence < 0.45 else "ambivalent"
    themes_str = ", ".join(nlp["themes"])
    keys_str = ", ".join(nlp["keywords"][:5])

    return (
        f"This track carries a {mood_word} emotional signature "
        f"(valence: {valence}). "
        f"Lyrically, it revolves around themes of {themes_str}, "
        f"with recurring imagery around: {keys_str}. "
        f"Sonically it feels {audio['sonic_mood']}, "
        f"sitting at {audio['tempo_bpm']} BPM in {audio['dominant_key']} {audio['mode']}."
    )


if __name__ == "__main__":
    from transcribe import transcribe_audio

    AUDIO_FILE = "diary_of_dreams_she_and_her_darkness 128.mp3"

    transcript = transcribe_audio(AUDIO_FILE)
    portrait = merge_signals(transcript, AUDIO_FILE)
    print(json.dumps(portrait, indent=2))