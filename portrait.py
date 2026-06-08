import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from wordcloud import WordCloud
from merge import THEME_COLORS, SONIC_MOOD_COLORS


def hex_to_rgb_norm(hex_color: str) -> tuple:
    h = hex_color.lstrip("#")
    return tuple(int(h[i:i+2], 16) / 255 for i in (0, 2, 4))


def make_gradient_cmap(hex_color: str) -> LinearSegmentedColormap:
    rgb = hex_to_rgb_norm(hex_color)
    return LinearSegmentedColormap.from_list("custom", ["#0f0f0f", hex_color])


def build_portrait(portrait: dict, output_path: str = "portrait.png") -> str:
    keywords        = portrait["keywords"]
    summary         = portrait["summary"]
    themes          = portrait["all_themes"]
    audio           = portrait["audio"]
    colors          = portrait["colors"]
    combined_val    = portrait["combined_valence"]

    primary_hex  = colors["primary"]
    sonic_hex    = colors["sonic"]
    primary_norm = hex_to_rgb_norm(primary_hex)

    BG      = "#0a0a0f"
    CARD_BG = "#12121a"
    TEXT    = "#e2e8f0"
    MUTED   = "#64748b"

    fig = plt.figure(figsize=(14, 9), facecolor=BG)

    # Title bar
    fig.text(0.5, 0.96, "LYRIC MIRROR", color=TEXT,
             fontsize=20, fontweight="bold", ha="center", va="top",
             fontfamily="monospace")
    fig.text(0.5, 0.925, "emotional portrait", color=MUTED,
             fontsize=10, ha="center", va="top", fontfamily="monospace")

    # Thin accent line under title
    line = plt.Line2D([0.1, 0.9], [0.905, 0.905],
                      transform=fig.transFigure,
                      color=primary_hex, linewidth=0.8, alpha=0.6)
    fig.add_artist(line)

    gs = gridspec.GridSpec(
        3, 3,
        figure=fig,
        left=0.05, right=0.97,
        top=0.89, bottom=0.05,
        hspace=0.45, wspace=0.35,
        height_ratios=[2.2, 2.2, 0.8],
    )

    # ------------------------------------------------------------------ #
    # Panel 1 — Word Cloud (top-left, spans 2 cols)
    # ------------------------------------------------------------------ #
    ax_wc = fig.add_subplot(gs[0, :2])
    ax_wc.set_facecolor(CARD_BG)
    for spine in ax_wc.spines.values():
        spine.set_edgecolor(primary_hex)
        spine.set_linewidth(0.5)

    freq = {w: (len(keywords) - i) ** 1.5 for i, w in enumerate(keywords)}
    wc = WordCloud(
        width=700, height=260,
        background_color=None, mode="RGBA",
        colormap=make_gradient_cmap(primary_hex),
        max_words=40,
        prefer_horizontal=0.8,
    ).generate_from_frequencies(freq)
    ax_wc.imshow(wc, interpolation="bilinear")
    ax_wc.axis("off")
    ax_wc.set_title("LYRICAL IMAGERY", color=MUTED, fontsize=8,
                    fontfamily="monospace", pad=6, loc="left")

    # ------------------------------------------------------------------ #
    # Panel 2 — Valence gauge (top-right)
    # ------------------------------------------------------------------ #
    ax_gauge = fig.add_subplot(gs[0, 2])
    ax_gauge.set_facecolor(CARD_BG)
    for spine in ax_gauge.spines.values():
        spine.set_edgecolor(sonic_hex)
        spine.set_linewidth(0.5)
    ax_gauge.set_xlim(0, 1)
    ax_gauge.set_ylim(0, 1)
    ax_gauge.axis("off")
    ax_gauge.set_title("VALENCE", color=MUTED, fontsize=8,
                       fontfamily="monospace", pad=6, loc="left")

    # Background track
    track = mpatches.FancyBboxPatch((0.1, 0.35), 0.8, 0.18,
                                     boxstyle="round,pad=0.02",
                                     facecolor="#1e1e2e", edgecolor="none",
                                     transform=ax_gauge.transAxes)
    ax_gauge.add_patch(track)

    # Fill
    fill_w = max(0.02, 0.8 * combined_val)
    fill = mpatches.FancyBboxPatch((0.1, 0.35), fill_w, 0.18,
                                    boxstyle="round,pad=0.02",
                                    facecolor=primary_hex, edgecolor="none",
                                    transform=ax_gauge.transAxes, alpha=0.9)
    ax_gauge.add_patch(fill)

    ax_gauge.text(0.5, 0.72, f"{combined_val:.2f}", color=TEXT,
                  fontsize=22, fontweight="bold", ha="center",
                  transform=ax_gauge.transAxes, fontfamily="monospace")
    ax_gauge.text(0.5, 0.18,
                  "negative" if combined_val < 0.45 else
                  "positive" if combined_val > 0.55 else "ambivalent",
                  color=primary_hex, fontsize=9, ha="center",
                  transform=ax_gauge.transAxes, fontfamily="monospace")

    # ------------------------------------------------------------------ #
    # Panel 3 — Audio features (middle-left)
    # ------------------------------------------------------------------ #
    ax_bars = fig.add_subplot(gs[1, :2])
    ax_bars.set_facecolor(CARD_BG)
    for spine in ax_bars.spines.values():
        spine.set_edgecolor("#1e293b")
        spine.set_linewidth(0.5)
    ax_bars.set_title("SONIC DIMENSIONS", color=MUTED, fontsize=8,
                      fontfamily="monospace", pad=6, loc="left")

    features = {
        "Valence":    combined_val,
        "Energy":     audio["energy"],
        "Tempo":      min(audio["tempo_bpm"] / 200, 1.0),
        "Brightness": min(audio["brightness_hz"] / 8000, 1.0),
    }
    bar_colors_hex = [primary_hex, sonic_hex, "#60a5fa", "#fb923c"]
    bar_colors_norm = [hex_to_rgb_norm(c) for c in bar_colors_hex]

    y_pos = range(len(features))
    bars = ax_bars.barh(
        list(features.keys()),
        list(features.values()),
        color=bar_colors_norm,
        height=0.45,
        edgecolor="none",
    )

    for bar, val, color in zip(bars, features.values(), bar_colors_hex):
        ax_bars.text(bar.get_width() + 0.02,
                     bar.get_y() + bar.get_height() / 2,
                     f"{val:.2f}", va="center", color=color,
                     fontsize=9, fontfamily="monospace")

    ax_bars.set_xlim(0, 1.15)
    ax_bars.tick_params(colors=MUTED, labelsize=9)
    ax_bars.set_facecolor(CARD_BG)
    for spine in ax_bars.spines.values():
        spine.set_visible(False)
    ax_bars.axvline(x=0, color=MUTED, linewidth=0.5)
    ax_bars.tick_params(left=False)
    ax_bars.yaxis.label.set_color(MUTED)
    plt.setp(ax_bars.get_yticklabels(), color=TEXT,
             fontfamily="monospace", fontsize=9)
    plt.setp(ax_bars.get_xticklabels(), color=MUTED, fontsize=7)

    # ------------------------------------------------------------------ #
    # Panel 4 — Track info (middle-right)
    # ------------------------------------------------------------------ #
    ax_info = fig.add_subplot(gs[1, 2])
    ax_info.set_facecolor(CARD_BG)
    for spine in ax_info.spines.values():
        spine.set_edgecolor("#1e293b")
        spine.set_linewidth(0.5)
    ax_info.axis("off")
    ax_info.set_title("TRACK INFO", color=MUTED, fontsize=8,
                      fontfamily="monospace", pad=6, loc="left")

    info_lines = [
        ("KEY",   f"{audio['dominant_key']} {audio['mode']}"),
        ("BPM",   str(audio["tempo_bpm"])),
        ("MOOD",  audio["sonic_mood"].upper()),
        ("THEME", ", ".join(themes).upper()),
    ]
    for i, (label, value) in enumerate(info_lines):
        y = 0.82 - i * 0.22
        ax_info.text(0.08, y, label, color=MUTED, fontsize=8,
                     transform=ax_info.transAxes, fontfamily="monospace")
        ax_info.text(0.08, y - 0.1, value, color=TEXT, fontsize=11,
                     fontweight="bold", transform=ax_info.transAxes,
                     fontfamily="monospace")

    # ------------------------------------------------------------------ #
    # Panel 5 — Summary + palette (bottom, full width)
    # ------------------------------------------------------------------ #
    ax_bottom = fig.add_subplot(gs[2, :])
    ax_bottom.set_facecolor(CARD_BG)
    for spine in ax_bottom.spines.values():
        spine.set_edgecolor("#1e293b")
        spine.set_linewidth(0.5)
    ax_bottom.axis("off")

    # Color swatches
    all_labels = themes + [f"sonic:{audio['sonic_mood']}"]
    all_colors = [THEME_COLORS.get(t, "#94a3b8") for t in themes] + \
                 [SONIC_MOOD_COLORS.get(audio["sonic_mood"], "#94a3b8")]
    n = len(all_labels)
    swatch_w = 0.06
    start_x = 0.02
    for i, (label, color) in enumerate(zip(all_labels, all_colors)):
        x = start_x + i * (swatch_w + 0.01)
        rect = mpatches.FancyBboxPatch(
            (x, 0.55), swatch_w, 0.3,
            boxstyle="round,pad=0.01",
            facecolor=color, edgecolor="none",
            transform=ax_bottom.transAxes, clip_on=False,
        )
        ax_bottom.add_patch(rect)
        ax_bottom.text(x + swatch_w / 2, 0.35, label,
                       color=MUTED, fontsize=7, ha="center",
                       transform=ax_bottom.transAxes,
                       fontfamily="monospace")

    # Summary text
    summary_short = summary[:180] + "..." if len(summary) > 180 else summary
    ax_bottom.text(start_x + n * (swatch_w + 0.01) + 0.02, 0.5,
                   summary_short, color=MUTED, fontsize=7.5,
                   transform=ax_bottom.transAxes, va="center",
                   wrap=True, fontfamily="monospace")

    plt.savefig(output_path, dpi=180, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"Portrait saved to: {output_path}")
    return output_path


if __name__ == "__main__":
    from transcribe import transcribe_audio
    from merge import merge_signals

    AUDIO_FILE = "diary_of_dreams_she_and_her_darkness 128.mp3"  # change this to your audio file

    transcript = transcribe_audio(AUDIO_FILE)
    portrait   = merge_signals(transcript, AUDIO_FILE)
    build_portrait(portrait)
