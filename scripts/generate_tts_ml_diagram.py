"""Generates the Machine Learning Architecture Diagram for StyleTTS 2 in Samvyas 2.0.
Creates both docs/styletts2_ml_architecture.excalidraw and docs/styletts2_ml_architecture.svg.
"""

from __future__ import annotations

import json
import random
from pathlib import Path


def create_ml_excalidraw() -> dict:
    elements = []
    seed_gen = random.Random(101)

    def next_seed() -> int:
        return seed_gen.randint(100000, 999999)

    def add_box(
        box_id: str,
        x: int,
        y: int,
        w: int,
        h: int,
        stroke_color: str,
        bg_color: str,
        title: str,
        lines: list[str],
        stroke_style: str = "solid",
        stroke_width: int = 2,
    ):
        elements.append(
            {
                "id": box_id,
                "type": "rectangle",
                "x": x,
                "y": y,
                "width": w,
                "height": h,
                "angle": 0,
                "strokeColor": stroke_color,
                "backgroundColor": bg_color,
                "fillStyle": "solid",
                "strokeWidth": stroke_width,
                "strokeStyle": stroke_style,
                "roughness": 1,
                "opacity": 100,
                "groupIds": [],
                "roundness": {"type": 3},
                "seed": next_seed(),
                "version": 1,
                "versionNonce": 1,
                "isDeleted": False,
                "boundElements": [],
                "updated": 1,
                "link": None,
                "locked": False,
            }
        )

        elements.append(
            {
                "id": f"{box_id}_title",
                "type": "text",
                "x": x + 12,
                "y": y + 12,
                "width": w - 24,
                "height": 22,
                "angle": 0,
                "strokeColor": stroke_color,
                "backgroundColor": "transparent",
                "fillStyle": "solid",
                "strokeWidth": 1,
                "strokeStyle": "solid",
                "roughness": 0,
                "opacity": 100,
                "groupIds": [],
                "roundness": None,
                "seed": next_seed(),
                "version": 1,
                "versionNonce": 1,
                "isDeleted": False,
                "boundElements": [],
                "updated": 1,
                "text": title,
                "fontSize": 15,
                "fontFamily": 1,
                "textAlign": "left",
                "verticalAlign": "top",
                "baseline": 15,
                "containerId": None,
                "originalText": title,
            }
        )

        body = "\n".join(lines)
        elements.append(
            {
                "id": f"{box_id}_body",
                "type": "text",
                "x": x + 12,
                "y": y + 38,
                "width": w - 24,
                "height": h - 45,
                "angle": 0,
                "strokeColor": "#334155",
                "backgroundColor": "transparent",
                "fillStyle": "solid",
                "strokeWidth": 1,
                "strokeStyle": "solid",
                "roughness": 0,
                "opacity": 100,
                "groupIds": [],
                "roundness": None,
                "seed": next_seed(),
                "version": 1,
                "versionNonce": 1,
                "isDeleted": False,
                "boundElements": [],
                "updated": 1,
                "text": body,
                "fontSize": 12,
                "fontFamily": 1,
                "textAlign": "left",
                "verticalAlign": "top",
                "baseline": 12,
                "containerId": None,
                "originalText": body,
            }
        )

    def add_arrow(
        arrow_id: str,
        start_x: int,
        start_y: int,
        end_x: int,
        end_y: int,
        stroke_color: str = "#64748b",
        label: str | None = None,
        stroke_style: str = "solid",
    ):
        dx = end_x - start_x
        dy = end_y - start_y
        elements.append(
            {
                "id": arrow_id,
                "type": "arrow",
                "x": start_x,
                "y": start_y,
                "width": abs(dx),
                "height": abs(dy),
                "angle": 0,
                "strokeColor": stroke_color,
                "backgroundColor": "transparent",
                "fillStyle": "solid",
                "strokeWidth": 2,
                "strokeStyle": stroke_style,
                "roughness": 1,
                "opacity": 100,
                "groupIds": [],
                "roundness": {"type": 2},
                "seed": next_seed(),
                "version": 1,
                "versionNonce": 1,
                "isDeleted": False,
                "boundElements": [],
                "updated": 1,
                "points": [[0, 0], [dx, dy]],
                "lastCommittedPoint": None,
                "startBinding": None,
                "endBinding": None,
                "startArrowhead": None,
                "endArrowhead": "arrow",
            }
        )

        if label:
            mid_x = start_x + dx // 2 - 25
            mid_y = start_y + dy // 2 - 14
            elements.append(
                {
                    "id": f"{arrow_id}_label",
                    "type": "text",
                    "x": mid_x,
                    "y": mid_y,
                    "width": 100,
                    "height": 18,
                    "angle": 0,
                    "strokeColor": stroke_color,
                    "backgroundColor": "transparent",
                    "fillStyle": "solid",
                    "strokeWidth": 1,
                    "strokeStyle": "solid",
                    "roughness": 0,
                    "opacity": 100,
                    "groupIds": [],
                    "roundness": None,
                    "seed": next_seed(),
                    "version": 1,
                    "versionNonce": 1,
                    "isDeleted": False,
                    "boundElements": [],
                    "updated": 1,
                    "text": label,
                    "fontSize": 11,
                    "fontFamily": 1,
                    "textAlign": "center",
                    "verticalAlign": "middle",
                    "baseline": 11,
                    "containerId": None,
                    "originalText": label,
                }
            )

    # Header Title
    elements.append(
        {
            "id": "title_main",
            "type": "text",
            "x": 80,
            "y": 40,
            "width": 1200,
            "height": 40,
            "angle": 0,
            "strokeColor": "#0f172a",
            "backgroundColor": "transparent",
            "fillStyle": "solid",
            "strokeWidth": 1,
            "strokeStyle": "solid",
            "roughness": 0,
            "opacity": 100,
            "groupIds": [],
            "roundness": None,
            "seed": next_seed(),
            "version": 1,
            "versionNonce": 1,
            "isDeleted": False,
            "boundElements": [],
            "updated": 1,
            "text": "StyleTTS 2 Deep Learning Architecture (ML Internal Diagram)",
            "fontSize": 26,
            "fontFamily": 1,
            "textAlign": "left",
            "verticalAlign": "top",
            "baseline": 26,
            "containerId": None,
            "originalText": "StyleTTS 2 Deep Learning Architecture (ML Internal Diagram)",
        }
    )

    # 1. Phoneme & Text Encoder
    add_box(
        "box_text_enc",
        x=80,
        y=120,
        w=260,
        h=190,
        stroke_color="#2563eb",
        bg_color="#eff6ff",
        title="1. Bengali Text Encoder",
        lines=[
            "Input: IPA Phonemes",
            "Shape: (B, T_text)",
            "• Phoneme Embedding (D=512)",
            "• Transformer / PL-BERT",
            "• Bi-directional Self-Attention",
            "Output: H_text",
            "Shape: (B, T_text, 512)",
        ],
    )

    # 2. Alignment & Duration Predictor
    add_box(
        "box_alignment",
        x=400,
        y=120,
        w=270,
        h=190,
        stroke_color="#0891b2",
        bg_color="#ecfeff",
        title="2. Duration & Length Regulator",
        lines=[
            "• Conditioned on: H_text & Style s",
            "• Duration Predictor (Bi-GRU)",
            "• Predicts phoneme duration d_i",
            "• Alignment Expansion / MAS:",
            "  T_text ──➔ T_frames (frame-level)",
            "Output: H_frame",
            "Shape: (B, T_frames, 512)",
        ],
    )

    # 3. Style Encoder & Style Bank
    add_box(
        "box_style",
        x=80,
        y=350,
        w=260,
        h=220,
        stroke_color="#db2777",
        bg_color="#fdf2f8",
        title="3. Acoustic Style Conditioning",
        lines=[
            "Option A: Style Bank (s_emo)",
            "  [warm, excited, whisper, ...]",
            "Option B: Style Encoder (Mel Spec)",
            "  Conv1D ResBlks + AvgPool",
            "Disentangled Style Vector:",
            "  s = [s_timbre (Jeet) || s_prosody]",
            "Shape: s ∈ ℝ^(B × 64)",
            "L2-Normalized",
        ],
    )

    # 4. Prosody Predictor (F0 & Energy)
    add_box(
        "box_prosody",
        x=400,
        y=350,
        w=270,
        h=220,
        stroke_color="#d97706",
        bg_color="#fffbeb",
        title="4. Pitch (F0) & Energy Predictor",
        lines=[
            "Input: [H_frame + Linear(s)]",
            "• Multi-layer Bi-GRU + Convolutions",
            "• Pitch Predictor: log(F0) contour",
            "• Energy Predictor: RMS norm",
            "• Pitch Shift Dial (semitone scale)",
            "Outputs:",
            "  F0: (B, T_frames), E: (B, T_frames)",
        ],
    )

    # 5. AdaIN Neural Generator (Core Decoder)
    add_box(
        "box_decoder",
        x=740,
        y=180,
        w=330,
        h=290,
        stroke_color="#16a34a",
        bg_color="#f0fdf4",
        title="5. AdaIN Residual Generator",
        lines=[
            "Core Non-Autoregressive Synthesis Engine",
            "Fused Inputs:",
            "  • H_frame (Acoustic content)",
            "  • F0 contour embedding & Energy",
            "  • Style vector s via AdaIN:",
            "    AdaIN(x, s) = σ(s) * (x - μ) / σ + μ(s)",
            "Architecture:",
            "  • 4x Cascaded Up-sampling Conv1D",
            "  • Multi-Receptive Field (MRF) ResBlocks",
            "  • Dilated Conv filters (kernel 3, 7, 11)",
            "Latency: ~220 ms (Single Forward Pass!)",
        ],
    )

    # 6. Vocoder & Audio Output
    add_box(
        "box_vocoder",
        x=1140,
        y=220,
        w=260,
        h=210,
        stroke_color="#7c3aed",
        bg_color="#faf5ff",
        title="6. Vocoder / Waveform Head",
        lines=[
            "Vocos / HiFi-GAN Vocoder",
            "• ISTFT / Polyphase synthesis",
            "• Peak Attenuation (-1 dB limiter)",
            "Outputs:",
            "  • Native: 24 kHz Hi-Fi PCM",
            "  • Resampled: 16 kHz PCM",
            "Shape: (B, 1, T_audio) in [-1, 1]",
        ],
    )

    # 7. Discriminators & Training Losses (Bottom Box)
    add_box(
        "box_loss",
        x=80,
        y=610,
        w=1320,
        h=160,
        stroke_color="#475569",
        bg_color="#f8fafc",
        title="⚖️ Training Objective & Adversarial Feedback (Discriminators)",
        lines=[
            "• Multi-Period Discriminator (MPD): Analyzes periodic harmonic patterns at periods [2, 3, 5, 7, 11]",
            "• Multi-Scale Spectrogram Discriminator (MSD): Evaluates consecutive structural audio frames across multiple temporal resolutions",
            "• Joint Loss Function:  L_total = L_adv(G; D) + λ_fm * L_fm(G; D) + λ_mel * L_mel(y, ŷ) + λ_F0 * L_F0 + λ_dur * L_dur",
            "  ➔ Enables single-pass naturalness WITHOUT slow diffusion sampling at runtime!",
        ],
        stroke_style="dashed",
    )

    # Arrows
    add_arrow(
        "arr_1_2",
        start_x=340,
        start_y=215,
        end_x=400,
        end_y=215,
        stroke_color="#2563eb",
        label="H_text",
    )
    add_arrow(
        "arr_2_5",
        start_x=670,
        start_y=215,
        end_x=740,
        end_y=240,
        stroke_color="#0891b2",
        label="H_frame",
    )
    add_arrow(
        "arr_3_4",
        start_x=340,
        start_y=460,
        end_x=400,
        end_y=460,
        stroke_color="#db2777",
        label="Style s",
    )
    add_arrow(
        "arr_3_5",
        start_x=340,
        start_y=400,
        end_x=740,
        end_y=350,
        stroke_color="#db2777",
        label="AdaIN Modulation",
    )
    add_arrow(
        "arr_4_5",
        start_x=670,
        start_y=460,
        end_x=740,
        end_y=410,
        stroke_color="#d97706",
        label="F0 & Energy",
    )
    add_arrow(
        "arr_5_6",
        start_x=1070,
        start_y=325,
        end_x=1140,
        end_y=325,
        stroke_color="#16a34a",
        label="Audio Tensor",
    )
    add_arrow(
        "arr_dec_loss",
        start_x=905,
        start_y=470,
        end_x=905,
        end_y=610,
        stroke_color="#64748b",
        label="Adversarial Feedback",
        stroke_style="dashed",
    )

    return {
        "type": "excalidraw",
        "version": 2,
        "source": "https://excalidraw.com",
        "elements": elements,
        "appState": {
            "viewBackgroundColor": "#ffffff",
            "gridSize": None,
        },
        "files": {},
    }


def generate_ml_svg() -> str:
    return """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1550 820" width="100%" height="100%" style="background-color: #0b1120; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;">
  <defs>
    <filter id="shadow" x="-5%" y="-5%" width="110%" height="115%" filterUnits="userSpaceOnUse">
      <feGaussianBlur stdDeviation="6" result="blur"/>
      <feOffset dx="0" dy="6" result="offset"/>
      <feComponentTransfer><feFuncA type="linear" slope="0.35"/></feComponentTransfer>
      <feMerge><feMergeNode/><feMergeNode in="SourceGraphic"/></feMerge>
    </filter>

    <marker id="m-blue" markerWidth="8" markerHeight="8" refX="6" refY="4" orient="auto"><path d="M 0 0 L 8 4 L 0 8 Z" fill="#60a5fa"/></marker>
    <marker id="m-cyan" markerWidth="8" markerHeight="8" refX="6" refY="4" orient="auto"><path d="M 0 0 L 8 4 L 0 8 Z" fill="#38bdf8"/></marker>
    <marker id="m-pink" markerWidth="8" markerHeight="8" refX="6" refY="4" orient="auto"><path d="M 0 0 L 8 4 L 0 8 Z" fill="#f472b6"/></marker>
    <marker id="m-amber" markerWidth="8" markerHeight="8" refX="6" refY="4" orient="auto"><path d="M 0 0 L 8 4 L 0 8 Z" fill="#fbbf24"/></marker>
    <marker id="m-green" markerWidth="8" markerHeight="8" refX="6" refY="4" orient="auto"><path d="M 0 0 L 8 4 L 0 8 Z" fill="#4ade80"/></marker>
    <marker id="m-purple" markerWidth="8" markerHeight="8" refX="6" refY="4" orient="auto"><path d="M 0 0 L 8 4 L 0 8 Z" fill="#c084fc"/></marker>
    <marker id="m-slate" markerWidth="8" markerHeight="8" refX="6" refY="4" orient="auto"><path d="M 0 0 L 8 4 L 0 8 Z" fill="#94a3b8"/></marker>

    <!-- Gradients -->
    <linearGradient id="g-text" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#1e3a8a" stop-opacity="0.9"/><stop offset="100%" stop-color="#2563eb" stop-opacity="0.6"/>
    </linearGradient>
    <linearGradient id="g-dur" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#164e63" stop-opacity="0.9"/><stop offset="100%" stop-color="#0891b2" stop-opacity="0.6"/>
    </linearGradient>
    <linearGradient id="g-style" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#831843" stop-opacity="0.9"/><stop offset="100%" stop-color="#db2777" stop-opacity="0.6"/>
    </linearGradient>
    <linearGradient id="g-pro" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#78350f" stop-opacity="0.9"/><stop offset="100%" stop-color="#d97706" stop-opacity="0.6"/>
    </linearGradient>
    <linearGradient id="g-gen" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#064e3b" stop-opacity="0.95"/><stop offset="100%" stop-color="#16a34a" stop-opacity="0.7"/>
    </linearGradient>
    <linearGradient id="g-voc" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#4c1d95" stop-opacity="0.9"/><stop offset="100%" stop-color="#7c3aed" stop-opacity="0.6"/>
    </linearGradient>
  </defs>

  <!-- Title -->
  <text x="60" y="55" fill="#f8fafc" font-size="27" font-weight="700" letter-spacing="-0.5px">StyleTTS 2 Deep Learning Architecture (ML Internal Diagram)</text>
  <text x="60" y="85" fill="#94a3b8" font-size="14" font-weight="400">Decoupled Neural Speech Synthesis: Bengali G2P ➔ Duration Expansion ➔ AdaIN ResBlock Generator ➔ Vocos</text>

  <!-- 1. Text Encoder -->
  <g filter="url(#shadow)">
    <rect x="60" y="120" width="260" height="200" rx="12" fill="url(#g-text)" stroke="#60a5fa" stroke-width="2"/>
    <rect x="75" y="135" width="230" height="30" rx="6" fill="#1d4ed8" fill-opacity="0.5"/>
    <text x="88" y="156" fill="#eff6ff" font-size="15" font-weight="700">1. Bengali Text Encoder</text>
    <text x="75" y="190" fill="#bfdbfe" font-size="12" font-weight="600">Input: IPA Phonemes</text>
    <text x="75" y="210" fill="#f8fafc" font-size="12">• Phoneme Embedding (D=512)</text>
    <text x="75" y="230" fill="#f8fafc" font-size="12">• Multi-layer Transformer / PL-BERT</text>
    <text x="75" y="250" fill="#f8fafc" font-size="12">• Bi-directional Self-Attention</text>
    <text x="75" y="275" fill="#60a5fa" font-size="12" font-weight="700">Output: H_text ∈ ℝ^(B × T_text × 512)</text>
  </g>

  <!-- Arrow 1 -> 2 -->
  <path d="M 320 220 L 370 220" stroke="#60a5fa" stroke-width="2.5" fill="none" marker-end="url(#m-blue)"/>
  <text x="325" y="210" fill="#60a5fa" font-size="11" font-weight="600">H_text</text>

  <!-- 2. Alignment & Duration Regulator -->
  <g filter="url(#shadow)">
    <rect x="380" y="120" width="280" height="200" rx="12" fill="url(#g-dur)" stroke="#38bdf8" stroke-width="2"/>
    <rect x="395" y="135" width="250" height="30" rx="6" fill="#0e7490" fill-opacity="0.5"/>
    <text x="408" y="156" fill="#ecfeff" font-size="15" font-weight="700">2. Duration &amp; Alignment</text>
    <text x="395" y="190" fill="#a5f3fc" font-size="12" font-weight="600">Conditioned on: H_text &amp; Style s</text>
    <text x="395" y="210" fill="#f8fafc" font-size="12">• Duration Predictor (Bi-GRU)</text>
    <text x="395" y="230" fill="#f8fafc" font-size="12">• Predicts discrete durations d_i</text>
    <text x="395" y="250" fill="#f8fafc" font-size="12">• Length Regulator / MAS Expansion</text>
    <text x="395" y="275" fill="#38bdf8" font-size="12" font-weight="700">Output: H_frame ∈ ℝ^(B × T_frames × 512)</text>
  </g>

  <!-- 3. Style Conditioning & Style Bank -->
  <g filter="url(#shadow)">
    <rect x="60" y="360" width="260" height="230" rx="12" fill="url(#g-style)" stroke="#f472b6" stroke-width="2"/>
    <rect x="75" y="375" width="230" height="30" rx="6" fill="#be185d" fill-opacity="0.5"/>
    <text x="88" y="396" fill="#fdf2f8" font-size="15" font-weight="700">3. Acoustic Style Engine</text>
    <text x="75" y="430" fill="#fbcfe8" font-size="12" font-weight="600">Decoupled Prosody Vector</text>
    <text x="75" y="450" fill="#f8fafc" font-size="12">• Style Bank: [warm, excited, ...]</text>
    <text x="75" y="470" fill="#f8fafc" font-size="12">• Style Encoder: Conv1D + Pool</text>
    <text x="75" y="490" fill="#f8fafc" font-size="12">• Disentangled Vector:</text>
    <text x="75" y="510" fill="#f8fafc" font-size="12">  s = [s_timbre (Jeet) || s_prosody]</text>
    <text x="75" y="535" fill="#f472b6" font-size="12" font-weight="700">Output: s ∈ ℝ^(B × 64) (Unit Norm)</text>
  </g>

  <!-- 4. Pitch & Energy Predictor -->
  <g filter="url(#shadow)">
    <rect x="380" y="360" width="280" height="230" rx="12" fill="url(#g-pro)" stroke="#fbbf24" stroke-width="2"/>
    <rect x="395" y="375" width="250" height="30" rx="6" fill="#b45309" fill-opacity="0.5"/>
    <text x="408" y="396" fill="#fffbeb" font-size="15" font-weight="700">4. Pitch &amp; Energy Predictor</text>
    <text x="395" y="430" fill="#fde68a" font-size="12" font-weight="600">Conditioned on [H_frame, s]</text>
    <text x="395" y="450" fill="#f8fafc" font-size="12">• Multi-layer Bi-GRU + Convolutions</text>
    <text x="395" y="470" fill="#f8fafc" font-size="12">• Pitch Predictor: log(F0) contour</text>
    <text x="395" y="490" fill="#f8fafc" font-size="12">• Energy Predictor: RMS norm</text>
    <text x="395" y="510" fill="#f8fafc" font-size="12">• Semitone Pitch Shift Dial</text>
    <text x="395" y="535" fill="#fbbf24" font-size="12" font-weight="700">Outputs: F0, Energy ∈ ℝ^(B × T_frames)</text>
  </g>

  <!-- Arrow 3 -> 4 -->
  <path d="M 320 475 L 370 475" stroke="#f472b6" stroke-width="2.5" fill="none" marker-end="url(#m-pink)"/>
  <text x="330" y="465" fill="#f472b6" font-size="11" font-weight="600">Style s</text>

  <!-- Connective Arrows into Generator (5) -->
  <path d="M 660 220 C 700 220, 700 250, 730 250" stroke="#38bdf8" stroke-width="2.5" fill="none" marker-end="url(#m-cyan)"/>
  <text x="675" y="230" fill="#38bdf8" font-size="11" font-weight="600">H_frame</text>

  <path d="M 320 420 C 500 420, 680 340, 730 340" stroke="#f472b6" stroke-width="2.5" fill="none" marker-end="url(#m-pink)"/>
  <text x="500" y="380" fill="#f472b6" font-size="11" font-weight="600">AdaIN Style s</text>

  <path d="M 660 475 C 700 475, 700 420, 730 420" stroke="#fbbf24" stroke-width="2.5" fill="none" marker-end="url(#m-amber)"/>
  <text x="675" y="455" fill="#fbbf24" font-size="11" font-weight="600">F0, Energy</text>

  <!-- 5. AdaIN Neural Generator (The Core Model) -->
  <g filter="url(#shadow)">
    <rect x="740" y="160" width="360" height="370" rx="14" fill="url(#g-gen)" stroke="#4ade80" stroke-width="2.5"/>
    <rect x="755" y="175" width="330" height="34" rx="8" fill="#15803d" fill-opacity="0.6"/>
    <text x="770" y="198" fill="#f0fdf4" font-size="16" font-weight="700">5. AdaIN Residual Generator</text>
    <text x="758" y="235" fill="#bbf7d0" font-size="13" font-weight="600">Non-Autoregressive Waveform Synthesizer</text>
    
    <rect x="758" y="250" width="324" height="42" rx="6" fill="#052e16" stroke="#22c55e" stroke-width="1"/>
    <text x="768" y="268" fill="#86efac" font-size="11" font-weight="600">AdaIN Operation (Feature Modulation):</text>
    <text x="768" y="284" fill="#dcfce7" font-size="11">AdaIN(x, s) = σ(s) · (x - μ(x)) / σ(x) + μ(s)</text>

    <text x="758" y="315" fill="#f0fdf4" font-size="12">• F0 Sine Source / Harmonic Carrier Excitations</text>
    <text x="758" y="335" fill="#f0fdf4" font-size="12">• 4x Cascaded Up-sampling Conv1D Transposed</text>
    <text x="758" y="355" fill="#f0fdf4" font-size="12">• Multi-Receptive Field (MRF) ResBlocks:</text>
    <text x="758" y="375" fill="#dcfce7" font-size="11">  Parallel Dilated Convs (kernel sizes: 3, 7, 11)</text>
    <text x="758" y="395" fill="#f0fdf4" font-size="12">• Modulates Timbre &amp; Prosody per layer</text>
    
    <rect x="758" y="420" width="324" height="40" rx="6" fill="#14532d"/>
    <text x="768" y="438" fill="#4ade80" font-size="12" font-weight="700">⚡ Single Forward Pass: ~220 ms (RTF &lt; 0.1!)</text>
    <text x="768" y="452" fill="#bbf7d0" font-size="11">No iterative 32-step diffusion on audio latents!</text>
  </g>

  <!-- Arrow 5 -> 6 -->
  <path d="M 1100 345 L 1150 345" stroke="#4ade80" stroke-width="2.5" fill="none" marker-end="url(#m-green)"/>
  <text x="1105" y="335" fill="#4ade80" font-size="11" font-weight="600">Wave</text>

  <!-- 6. Vocoder / Waveform Head -->
  <g filter="url(#shadow)">
    <rect x="1160" y="200" width="280" height="240" rx="12" fill="url(#g-voc)" stroke="#c084fc" stroke-width="2"/>
    <rect x="1175" y="215" width="250" height="30" rx="6" fill="#6d28d9" fill-opacity="0.5"/>
    <text x="1188" y="236" fill="#faf5ff" font-size="15" font-weight="700">6. Neural Vocoder Output</text>
    <text x="1175" y="270" fill="#e9d5ff" font-size="12" font-weight="600">Vocos / HiFi-GAN Vocoder</text>
    <text x="1175" y="295" fill="#f8fafc" font-size="12">• Inverse STFT (iSTFT) / Synthesis</text>
    <text x="1175" y="315" fill="#f8fafc" font-size="12">• Peak Attenuation (-1.0 dB limiter)</text>
    <text x="1175" y="340" fill="#c084fc" font-size="12" font-weight="700">Dual Export Streams:</text>
    <text x="1175" y="360" fill="#f8fafc" font-size="12">  - Native: 24 kHz Hi-Fi PCM-16/24</text>
    <text x="1175" y="380" fill="#f8fafc" font-size="12">  - Resampled: 16 kHz Mono PCM</text>
    <text x="1175" y="405" fill="#e9d5ff" font-size="11">Shape: (B, 1, T_samples) in [-1.0, 1.0]</text>
  </g>

  <!-- 7. Discriminators & Training Losses (Bottom Box) -->
  <g filter="url(#shadow)">
    <rect x="60" y="630" width="1380" height="150" rx="12" fill="#0f172a" stroke="#64748b" stroke-width="2" stroke-dasharray="6 4"/>
    <rect x="80" y="645" width="460" height="28" rx="6" fill="#1e293b"/>
    <text x="95" y="664" fill="#f1f5f9" font-size="14" font-weight="700">⚖️ Training Objective &amp; Discriminator Losses (Offline Kaggle Adaptation)</text>
    
    <text x="80" y="700" fill="#cbd5e1" font-size="12">• Multi-Period Discriminator (MPD): Evaluates periodic harmonic patterns at prime intervals [p = 2, 3, 5, 7, 11] to capture human pitch periodicity.</text>
    <text x="80" y="720" fill="#cbd5e1" font-size="12">• Multi-Scale Spectrogram Discriminator (MSD): Evaluates consecutive audio frames across multiple temporal resolutions to enforce spectral realism.</text>
    <text x="80" y="745" fill="#38bdf8" font-size="12" font-weight="700">Total Objective: L = L_adv(G; D) + λ_fm · L_fm(G; D) + λ_mel · L_mel(y, ŷ) + λ_F0 · L_F0 + λ_dur · L_dur</text>
  </g>

  <!-- Dashed feedback arrow from Generator to Discriminator -->
  <path d="M 920 530 L 920 620" stroke="#94a3b8" stroke-width="2" stroke-dasharray="5 5" fill="none" marker-end="url(#m-slate)"/>
  <text x="930" y="580" fill="#94a3b8" font-size="11" font-weight="600">Adversarial Gradients</text>

</svg>
"""


def main():
    excal_data = create_ml_excalidraw()
    excal_path = Path("docs/styletts2_ml_architecture.excalidraw")
    with open(excal_path, "w", encoding="utf-8") as f:
        json.dump(excal_data, f, indent=2)
    print(f"[+] Created ML Excalidraw diagram: {excal_path.resolve()}")

    svg_data = generate_ml_svg()
    svg_path = Path("docs/styletts2_ml_architecture.svg")
    with open(svg_path, "w", encoding="utf-8") as f:
        f.write(svg_data.strip())
    print(f"[+] Created ML high-res SVG diagram: {svg_path.resolve()}")


if __name__ == "__main__":
    main()
