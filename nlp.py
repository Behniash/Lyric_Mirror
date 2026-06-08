import json
import spacy


POSITIVE_WORDS = {
    "love", "hope", "joy", "happy", "smile", "light", "dream", "free",
    "beautiful", "good", "peace", "strong", "rise", "bright", "warm",
    "laugh", "dance", "sing", "thank", "forgive", "heal", "better"
}

NEGATIVE_WORDS = {
    "pain", "hurt", "cry", "dark", "lost", "alone", "break", "hate",
    "fear", "die", "dying", "empty", "cold", "tear", "sad", "wrong",
    "vanish", "hunger", "violence", "silence", "distance", "defeat",
    "broken", "miss", "lie", "lying"
}


def analyze_sentiment(text: str) -> dict:
    """
    Lexicon-based sentiment analysis — no model download needed.
    Counts positive vs negative words and returns a label + score.
    """
    words = text.lower().split()
    pos = sum(1 for w in words if w.strip(".,!?\"'") in POSITIVE_WORDS)
    neg = sum(1 for w in words if w.strip(".,!?\"'") in NEGATIVE_WORDS)
    total = pos + neg or 1

    score = round(pos / total, 4)
    label = "POSITIVE" if score >= 0.5 else "NEGATIVE"
    return {"label": label, "score": score, "positive_hits": pos, "negative_hits": neg}


def extract_keywords(text: str, top_n: int = 15) -> list:
    """
    Extracts the most meaningful words using spaCy POS tagging.
    Keeps nouns, verbs, adjectives — filters stopwords and short tokens.
    """
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)

    allowed_pos = {"NOUN", "VERB", "ADJ", "ADV"}
    words = [
        token.lemma_.lower()
        for token in doc
        if token.pos_ in allowed_pos
        and not token.is_stop
        and not token.is_punct
        and len(token.text) > 2
    ]

    freq = {}
    for w in words:
        freq[w] = freq.get(w, 0) + 1

    ranked = sorted(freq.items(), key=lambda x: x[1], reverse=True)
    return [word for word, _ in ranked[:top_n]]


def detect_themes(keywords: list) -> list:
    """
    Maps extracted keywords to high-level emotional themes.
    """
    theme_map = {
        "love":      ["love", "heart", "kiss", "hold", "touch", "feel", "want", "need", "miss"],
        "pain":      ["pain", "hurt", "cry", "tear", "break", "lost", "alone", "empty", "dark"],
        "hope":      ["hope", "light", "rise", "dream", "believe", "fly", "free", "strong", "new"],
        "anger":     ["hate", "rage", "fire", "fight", "war", "burn", "scream", "break", "wrong"],
        "nostalgia": ["remember", "old", "time", "past", "memory", "used", "once", "back", "gone"],
        "joy":       ["happy", "smile", "laugh", "dance", "sing", "bright", "sun", "good", "day"],
    }

    found_themes = []
    kw_set = set(keywords)
    for theme, words in theme_map.items():
        if kw_set & set(words):
            found_themes.append(theme)
    return found_themes if found_themes else ["unclassified"]


def run_nlp_analysis(transcript: dict) -> dict:
    """
    Full NLP pipeline on a transcript dict (output of week1_transcribe).
    """
    text = transcript["text"]

    print("Running sentiment analysis...")
    sentiment = analyze_sentiment(text)

    print("Extracting keywords...")
    keywords = extract_keywords(text)

    print("Detecting themes...")
    themes = detect_themes(keywords)

    return {
        "sentiment": sentiment,
        "keywords": keywords,
        "themes": themes,
    }


if __name__ == "__main__":
    from transcribe import transcribe_audio

    AUDIO_FILE = "diary_of_dreams_she_and_her_darkness 128.mp3" 

    transcript = transcribe_audio(AUDIO_FILE)
    result = run_nlp_analysis(transcript)
    print(json.dumps(result, indent=2))
