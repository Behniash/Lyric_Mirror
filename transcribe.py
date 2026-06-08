import whisper
import os


def transcribe_audio(audio_path: str, language: str = None) -> dict:
    """
    Transcribes an audio file using OpenAI Whisper.

    Args:
        audio_path: Path to the audio file (mp3, wav, m4a, ...)
        language: Optional language code e.g. 'en', 'fa'
                  If None, Whisper auto-detects.

    Returns:
        dict with keys: text, language, segments
    """
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"File not found: {audio_path}")

    print("Loading Whisper model...")
    model = whisper.load_model("base")

    print(f"Transcribing: {audio_path}")
    result = model.transcribe(audio_path, language=language, verbose=False)

    return {
        "text": result["text"].strip(),
        "language": result["language"],
        "segments": [
            {
                "start": round(seg["start"], 2),
                "end": round(seg["end"], 2),
                "text": seg["text"].strip(),
            }
            for seg in result["segments"]
        ],
    }


def save_transcript(transcript: dict, output_path: str = "transcript.txt") -> None:
    """Saves the transcript dict to a plain text file."""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"Detected language: {transcript['language']}\n")
        f.write("=" * 50 + "\n\n")
        f.write(transcript["text"])
        f.write("\n\n" + "=" * 50 + "\n")
        f.write("Timed segments:\n\n")
        for seg in transcript["segments"]:
            f.write(f"[{seg['start']}s -> {seg['end']}s]  {seg['text']}\n")
    print(f"Transcript saved to: {output_path}")


def print_transcript(transcript: dict) -> None:
    print("\n" + "=" * 50)
    print(f"Language: {transcript['language']}")
    print("=" * 50)
    print(transcript["text"])
    print(f"\nSegments: {len(transcript['segments'])}")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    AUDIO_FILE = "diary_of_dreams_she_and_her_darkness 128.mp3"

    transcript = transcribe_audio(AUDIO_FILE)
    print_transcript(transcript)
    save_transcript(transcript)