"""Hinglish Shield Gradio demo (HF Space scaffold).

Demo mode: mock outputs until model endpoints are wired.
All chart series are MOCK / PLACEHOLDER from mock_data.py.
"""

from __future__ import annotations

import random
from typing import Any

import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

import mock_data as md

DEMO_BANNER = (
    "**Demo mode: mock outputs until model endpoints are wired.** "
    "All charts use MOCK / PLACEHOLDER numbers (not real eval)."
)

CUSTOM_CSS = """
.verdict-safe { background:#14532d !important; color:#86efac !important;
  border:1px solid #22c55e !important; border-radius:999px !important;
  padding:4px 12px !important; display:inline-block !important; font-weight:600 !important; }
.verdict-unsafe { background:#7f1d1d !important; color:#fca5a5 !important;
  border:1px solid #ef4444 !important; border-radius:999px !important;
  padding:4px 12px !important; display:inline-block !important; font-weight:600 !important; }
.verdict-refused { background:#374151 !important; color:#d1d5db !important;
  border:1px solid #9ca3af !important; border-radius:999px !important;
  padding:4px 12px !important; display:inline-block !important; font-weight:600 !important; }
.gap-headline { font-size:2.4rem !important; font-weight:700 !important;
  color:#2DD4BF !important; line-height:1.2 !important; }
.mock-note { opacity:0.85; font-size:0.9rem; }
"""

_attack_idx = {"i": 0}


def _style_axes(ax: Any) -> None:
    ax.set_facecolor(md.COLOR_BG)
    ax.tick_params(colors=md.COLOR_FG)
    ax.xaxis.label.set_color(md.COLOR_FG)
    ax.yaxis.label.set_color(md.COLOR_FG)
    ax.title.set_color(md.COLOR_FG)
    for spine in ax.spines.values():
        spine.set_color(md.COLOR_GRID)
    ax.grid(True, color=md.COLOR_GRID, alpha=0.4, linestyle="--", linewidth=0.6)


def _new_fig(h: float = 4.2, w: float = 7.5) -> tuple[Figure, Any]:
    fig, ax = plt.subplots(figsize=(w, h), dpi=120)
    fig.patch.set_facecolor(md.COLOR_BG)
    _style_axes(ax)
    return fig, ax


def _chip_html(verdict: str) -> str:
    key = verdict.lower()
    cls = {
        "safe": "verdict-safe",
        "unsafe": "verdict-unsafe",
        "refused": "verdict-refused",
    }.get(key, "verdict-refused")
    return f'<span class="{cls}">{verdict}</span>'


def random_attack(lang: str) -> str:
    pool = [p for p in md.MOCK_ATTACK_PROMPTS if p["lang"] == lang]
    if not pool:
        pool = md.MOCK_ATTACK_PROMPTS
    _attack_idx["i"] = (_attack_idx["i"] + 1) % len(pool)
    return pool[_attack_idx["i"]]["prompt"]


def run_compare(prompt: str, lang: str) -> tuple[str, str, str, str, str, str]:
    lang = lang if lang in md.MOCK_RESPONSES else "EN"
    base = md.MOCK_RESPONSES[lang]["base"]
    shield = md.MOCK_RESPONSES[lang]["shield"]
    lo, hi = md.MOCK_LATENCY_MS.get(lang, (200, 400))
    base_ms = random.randint(lo, hi)
    shield_ms = random.randint(lo, hi) + random.randint(10, 60)
    return (
        base["text"],
        _chip_html(base["verdict"]),
        f"**Latency:** {base_ms} ms (MOCK)",
        shield["text"],
        _chip_html(shield["verdict"]),
        f"**Latency:** {shield_ms} ms (MOCK)",
    )


def plot_asr_bars() -> Figure:
    langs = list(md.MOCK_ASR_BEFORE.keys())
    x = np.arange(len(langs))
    width = 0.35
    before = [md.MOCK_ASR_BEFORE[l] for l in langs]
    after = [md.MOCK_ASR_AFTER[l] for l in langs]
    ci = md.MOCK_CI_ASR

    fig, ax = _new_fig()
    ax.bar(
        x - width / 2, before, width, yerr=ci, capsize=4,
        label="Base (MOCK)", color=md.COLOR_BASE, ecolor=md.COLOR_FG,
    )
    ax.bar(
        x + width / 2, after, width, yerr=ci, capsize=4,
        label="Hinglish Shield (MOCK)", color=md.COLOR_REPAIRED, ecolor=md.COLOR_FG,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(langs)
    ax.set_ylabel("Attack Success Rate")
    ax.set_ylim(0, 0.6)
    ax.set_title(f"ASR by language before vs after  |  n={md.MOCK_N_ASR}  |  95% CI +/-{ci:.2f}  |  MOCK")
    ax.legend(facecolor=md.COLOR_BG, edgecolor=md.COLOR_GRID, labelcolor=md.COLOR_FG)
    fig.tight_layout()
    return fig


def plot_over_refusal_bars() -> Figure:
    langs = list(md.MOCK_OVER_REFUSAL_BEFORE.keys())
    x = np.arange(len(langs))
    width = 0.35
    before = [md.MOCK_OVER_REFUSAL_BEFORE[l] for l in langs]
    after = [md.MOCK_OVER_REFUSAL_AFTER[l] for l in langs]
    ci = md.MOCK_CI_OVER_REFUSAL

    fig, ax = _new_fig()
    ax.bar(
        x - width / 2, before, width, yerr=ci, capsize=4,
        label="Base (MOCK)", color=md.COLOR_BASE, ecolor=md.COLOR_FG,
    )
    ax.bar(
        x + width / 2, after, width, yerr=ci, capsize=4,
        label="Hinglish Shield (MOCK)", color=md.COLOR_REPAIRED, ecolor=md.COLOR_FG,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(langs)
    ax.set_ylabel("Over-refusal rate")
    ax.set_ylim(0, 0.35)
    ax.set_title(
        f"Over-refusal by language  |  n={md.MOCK_N_OVER_REFUSAL}  |  "
        f"95% CI +/-{ci:.3f}  |  MOCK"
    )
    ax.legend(facecolor=md.COLOR_BG, edgecolor=md.COLOR_GRID, labelcolor=md.COLOR_FG)
    fig.tight_layout()
    return fig


def plot_tradeoff_curve() -> Figure:
    strengths = md.MOCK_STEERING_STRENGTHS
    asr = md.MOCK_TRADEOFF_ASR
    over = md.MOCK_TRADEOFF_OVER_REFUSAL
    ci = md.MOCK_CI_TRADEOFF

    fig, ax = _new_fig(h=4.5)
    ax.errorbar(
        over, asr, xerr=ci, yerr=ci, fmt="-o", color=md.COLOR_REPAIRED,
        ecolor=md.COLOR_FG, capsize=3, label="Steering path (MOCK)",
    )
    for s, x, y in zip(strengths, over, asr):
        ax.annotate(
            f"s={s:g}", (x, y), textcoords="offset points", xytext=(6, 4),
            color=md.COLOR_FG, fontsize=8,
        )
    ax.scatter(
        [md.MOCK_OVER_REFUSAL_BEFORE["Hinglish"]],
        [md.MOCK_ASR_BEFORE["Hinglish"]],
        color=md.COLOR_BASE, s=80, zorder=5, label="Base Hinglish (MOCK)",
    )
    ax.set_xlabel("Over-refusal rate (Hinglish)")
    ax.set_ylabel("ASR (Hinglish)")
    ax.set_title(
        f"Safety vs over-refusal tradeoff  |  n={md.MOCK_N_TRADEOFF} strengths  |  "
        f"95% CI +/-{ci:.2f}  |  MOCK"
    )
    ax.legend(facecolor=md.COLOR_BG, edgecolor=md.COLOR_GRID, labelcolor=md.COLOR_FG)
    fig.tight_layout()
    return fig


def capability_table() -> pd.DataFrame:
    rows = []
    for r in md.MOCK_CAPABILITY_ROWS:
        rows.append(
            {
                "Benchmark": r["benchmark"],
                "Base (MOCK)": f"{r['base']:.2f}",
                "Repaired (MOCK)": f"{r['repaired']:.2f}",
                "n": r["n"],
            }
        )
    return pd.DataFrame(rows)


def plot_refusal_projection() -> Figure:
    layers = md.MOCK_LAYERS
    fig, ax = _new_fig()
    ax.plot(
        layers, md.MOCK_REFUSAL_PROJ_EN, "-o", color=md.COLOR_BASE,
        label="EN (MOCK)", markersize=4,
    )
    ax.plot(
        layers, md.MOCK_REFUSAL_PROJ_HINGLISH, "-o", color=md.COLOR_REPAIRED,
        label="Hinglish (MOCK)", markersize=4,
    )
    ax.fill_between(
        layers,
        np.array(md.MOCK_REFUSAL_PROJ_EN) - 0.03,
        np.array(md.MOCK_REFUSAL_PROJ_EN) + 0.03,
        color=md.COLOR_BASE, alpha=0.2,
    )
    ax.fill_between(
        layers,
        np.array(md.MOCK_REFUSAL_PROJ_HINGLISH) - 0.03,
        np.array(md.MOCK_REFUSAL_PROJ_HINGLISH) + 0.03,
        color=md.COLOR_REPAIRED, alpha=0.2,
    )
    ax.set_xlabel("Layer")
    ax.set_ylabel("Refusal-direction projection")
    ax.set_title(
        f"Layer-wise refusal projection  |  n={md.MOCK_N_LAYER} prompts  |  "
        f"band ~95% CI  |  MOCK"
    )
    ax.legend(facecolor=md.COLOR_BG, edgecolor=md.COLOR_GRID, labelcolor=md.COLOR_FG)
    fig.tight_layout()
    return fig


def plot_circuit_heatmap() -> Figure:
    scores = np.array(md.MOCK_CIRCUIT_SCORES)
    fig, ax = _new_fig(h=5.0, w=8.0)
    from matplotlib.colors import LinearSegmentedColormap
    teal_cmap = LinearSegmentedColormap.from_list(
        "shield_teal", ["#0D1117", "#134E4A", "#2DD4BF", "#99F6E4"]
    )
    im = ax.imshow(scores, aspect="auto", cmap=teal_cmap, origin="lower")
    ax.set_xticks(range(len(md.MOCK_CIRCUIT_NODES)))
    ax.set_xticklabels(md.MOCK_CIRCUIT_NODES, rotation=30, ha="right")
    ax.set_yticks(range(len(md.MOCK_LAYERS)))
    ax.set_yticklabels([str(l) for l in md.MOCK_LAYERS])
    ax.set_xlabel("CircuitKIT node")
    ax.set_ylabel("Layer")
    ax.set_title(f"CircuitKIT node scores  |  n={md.MOCK_N_LAYER}  |  MOCK / PLACEHOLDER")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.yaxis.set_tick_params(color=md.COLOR_FG)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color=md.COLOR_FG)
    cbar.outline.set_edgecolor(md.COLOR_GRID)
    fig.tight_layout()
    return fig


def sample_rows_df() -> pd.DataFrame:
    return pd.DataFrame(md.MOCK_SAMPLE_ROWS)


def dataset_markdown() -> str:
    lines = ["### HF datasets (PLACEHOLDER links)", ""]
    for d in md.MOCK_HF_DATASETS:
        lines.append(f"- **[{d['name']}]({d['url']})**: {d['desc']}")
    lines.extend(["", "### Pipeline", "", md.PIPELINE_MERMAID])
    return "\n".join(lines)


def build_app() -> gr.Blocks:
    theme = gr.themes.Soft(
        primary_hue=gr.themes.colors.teal,
        neutral_hue=gr.themes.colors.gray,
    ).set(
        body_background_fill="#0D1117",
        body_text_color="#E6EDF3",
        block_background_fill="#161B22",
        block_border_color="#30363D",
        border_color_primary="#30363D",
        button_primary_background_fill="#0D9488",
        button_primary_text_color="#FFFFFF",
    )

    with gr.Blocks(title="Hinglish Shield", theme=theme, css=CUSTOM_CSS) as demo:
        gr.Markdown("# Hinglish Shield")
        gr.Markdown(
            "Cross-lingual safety repair for Tiny Aya on EN / HI / Hinglish "
            "(Lexsi Hackathon Track 2)."
        )
        gr.Markdown(DEMO_BANNER)

        with gr.Tabs():
            # ---- Tab 1: Compare ----
            with gr.Tab("Compare"):
                gr.Markdown(
                    "Side-by-side base Tiny Aya vs Hinglish Shield. "
                    "**No model is loaded**; responses are canned MOCK outputs."
                )
                with gr.Row():
                    lang = gr.Radio(
                        choices=["EN", "HI", "Hinglish"],
                        value="Hinglish",
                        label="Language",
                    )
                    rand_btn = gr.Button("Random attack", variant="secondary")
                prompt = gr.Textbox(
                    label="Prompt",
                    lines=3,
                    value=md.MOCK_ATTACK_PROMPTS[-1]["prompt"],
                )
                run_btn = gr.Button("Run compare", variant="primary")

                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### Base Tiny Aya")
                        base_out = gr.Textbox(label="Response", lines=6)
                        base_chip = gr.HTML(label="Verdict")
                        base_lat = gr.Markdown("Latency: --")
                    with gr.Column():
                        gr.Markdown("### Hinglish Shield")
                        shield_out = gr.Textbox(label="Response", lines=6)
                        shield_chip = gr.HTML(label="Verdict")
                        shield_lat = gr.Markdown("Latency: --")

                rand_btn.click(fn=random_attack, inputs=[lang], outputs=[prompt])
                run_btn.click(
                    fn=run_compare,
                    inputs=[prompt, lang],
                    outputs=[base_out, base_chip, base_lat, shield_out, shield_chip, shield_lat],
                )

            # ---- Tab 2: Results ----
            with gr.Tab("Results"):
                gr.Markdown(
                    f'<p class="mock-note">MOCK / PLACEHOLDER metrics. '
                    f"Replace from <code>runs/&lt;id&gt;/metrics.json</code>.</p>"
                )
                gr.Markdown("### Language gap (base ASR Hinglish minus EN)")
                gap_pp = md.MOCK_LANGUAGE_GAP * 100
                gr.HTML(
                    f'<div class="gap-headline">{gap_pp:.0f} pp</div>'
                    f'<div class="mock-note">MOCK headline  |  n={md.MOCK_N_ASR}</div>'
                )
                gr.Markdown("### ASR by language before vs after")
                gr.Plot(value=plot_asr_bars())
                gr.Markdown("### Over-refusal")
                gr.Plot(value=plot_over_refusal_bars())
                gr.Markdown("### Safety vs over-refusal tradeoff (money plot)")
                gr.Plot(value=plot_tradeoff_curve())
                gr.Markdown("### Capability retention")
                gr.Dataframe(value=capability_table(), interactive=False)

            # ---- Tab 3: Inside the model ----
            with gr.Tab("Inside the model"):
                gr.Markdown(
                    "### Where safety lives\n"
                    "Refusal-direction projection peaks in mid layers for EN, "
                    "but stays weaker and flatter for Hinglish (MOCK). "
                    "CircuitKIT highlights attn.o / mlp.up in the same band."
                )
                gr.Plot(value=plot_refusal_projection())
                gr.Markdown("### CircuitKIT node scores (heatmap)")
                gr.Plot(value=plot_circuit_heatmap())

            # ---- Tab 4: Data ----
            with gr.Tab("Data"):
                gr.Markdown(dataset_markdown())
                gr.Markdown("### Sample rows (MOCK)")
                gr.Dataframe(value=sample_rows_df(), interactive=False)
                gr.Markdown(md.MOCK_LABELING_GUIDE)

        gr.Markdown(
            "<span class='mock-note'>Hinglish Shield scaffold. "
            "Charts always show n= and CI. Series colors: base gray, repaired teal; "
            "unsafe verdicts in red.</span>"
        )

    return demo


demo = build_app()

if __name__ == "__main__":
    demo.launch()
