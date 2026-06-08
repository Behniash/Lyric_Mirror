import gradio as gr
import json

from transcribe import transcribe_audio
from merge import merge_signals
from portrait import build_portrait


def analyze(audio_file, language_choice: str):
    if audio_file is None:
        raise gr.Error("Please upload an audio file first.")

    lang_map = {
        "Auto-detect": None,
        "English": "en",
        "Persian (Farsi)": "fa",
    }
    lang = lang_map.get(language_choice)

    transcript = transcribe_audio(audio_file, language=lang)
    portrait   = merge_signals(transcript, audio_file)

    portrait_path = "portrait.png"
    build_portrait(portrait, output_path=portrait_path)

    summary         = portrait["summary"]
    transcript_text = transcript["text"]
    portrait_json   = json.dumps(
        {k: v for k, v in portrait.items() if k != "colors"},
        indent=2,
    )
    audio_info = (
        f"Key: {portrait['audio']['dominant_key']} {portrait['audio']['mode']}   |   "
        f"BPM: {portrait['audio']['tempo_bpm']}   |   "
        f"Mood: {portrait['audio']['sonic_mood'].capitalize()}   |   "
        f"Valence: {portrait['combined_valence']}"
    )

    return portrait_path, transcript_text, summary, audio_info, portrait_json


CSS = """
body, .gradio-container {
    background-color: #080810 !important;
    font-family: 'Inter', sans-serif !important;
}
"""

THEME = gr.themes.Base(
    primary_hue="violet",
    secondary_hue="indigo",
    neutral_hue="slate",
).set(
    body_background_fill="#080810",
    body_text_color="#e2e8f0",
    block_background_fill="#0f0f1a",
    block_border_color="#1e1e30",
    block_label_text_color="#94a3b8",
    input_background_fill="#0a0a14",
    input_border_color="#1e1e30",
    button_primary_background_fill="*primary_500",
    button_primary_text_color="#ffffff",
)

with gr.Blocks(title="Lyric Mirror") as demo:

    gr.HTML("""
    <div style="text-align:center; padding: 2rem 0 1rem;">
        <h1 style="font-size:2.2rem; font-weight:700;
                   background: linear-gradient(135deg, #a78bfa, #60a5fa);
                   -webkit-background-clip: text; -webkit-text-fill-color: transparent;
                   margin:0; letter-spacing:-0.02em;">
            Lyric Mirror
        </h1>
        <p style="color:#64748b; font-size:0.95rem; margin-top:0.5rem;">
            Upload a song &mdash; get an emotional portrait of its artist
        </p>
        <div style="width:60px; height:2px; background:linear-gradient(90deg,#7c3aed,#4f46e5);
                    margin:1rem auto 0;"></div>
    </div>
    """)

    with gr.Row(equal_height=False):
        with gr.Column(scale=1, min_width=280):
            audio_input = gr.Audio(
                label="Audio File",
                type="filepath",
            )
            language_input = gr.Radio(
                choices=["Auto-detect", "English", "Persian (Farsi)"],
                value="Auto-detect",
                label="Language",
            )
            run_btn = gr.Button("Analyze", variant="primary", size="lg")

            gr.HTML("""
            <div style="margin-top:1.5rem; padding:1rem; background:#0a0a14;
                        border:1px solid #1e1e30; border-radius:10px;">
                <p style="color:#64748b; font-size:11px; letter-spacing:0.08em;
                          text-transform:uppercase; margin:0 0 0.75rem;">How it works</p>
                <div style="color:#94a3b8; font-size:12px; line-height:1.8;">
                    <span style="color:#a78bfa;">01</span> &nbsp;Whisper transcribes the audio<br>
                    <span style="color:#a78bfa;">02</span> &nbsp;NLP extracts themes and sentiment<br>
                    <span style="color:#a78bfa;">03</span> &nbsp;librosa analyzes acoustic features<br>
                    <span style="color:#a78bfa;">04</span> &nbsp;Signals merge into an emotional portrait
                </div>
            </div>
            """)

        with gr.Column(scale=2):
            portrait_output = gr.Image(
                label="Emotional Portrait",
                type="filepath",
            )
            audio_info_output = gr.Textbox(
                label="Track Analysis",
                interactive=False,
            )
            summary_output = gr.Textbox(
                label="Summary",
                lines=3,
                interactive=False,
            )

    with gr.Accordion("Full transcript and raw data", open=False):
        with gr.Row():
            transcript_output = gr.Textbox(
                label="Transcribed Lyrics",
                lines=10,
                interactive=False,
            )
            json_output = gr.Code(
                label="Portrait JSON",
                language="json",
            )

    run_btn.click(
        fn=analyze,
        inputs=[audio_input, language_input],
        outputs=[portrait_output, transcript_output, summary_output,
                 audio_info_output, json_output],
    )

    gr.HTML("""
    <div style="text-align:center; padding:1.5rem 0 0.5rem; color:#334155; font-size:11px;">
        Lyric Mirror &nbsp;&mdash;&nbsp; Whisper &middot; spaCy &middot; librosa &middot; Gradio
    </div>
    """)


if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        show_error=True,
        theme=THEME,
        css=CSS,
    )