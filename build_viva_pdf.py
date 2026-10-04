"""
Build SPECTRA_Viva_Prep.pdf — a short, dense, professor-facing prep pack.
"""
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle,
    KeepTogether, HRFlowable, Preformatted,
)

OUT = "SPECTRA_Viva_Prep.pdf"

# ----- Colors -----
ACCENT = colors.HexColor("#1F6FEB")      # blue
ACCENT_DARK = colors.HexColor("#0B3D91")
INK = colors.HexColor("#0F172A")          # near-black
MUTED = colors.HexColor("#475569")
PANEL = colors.HexColor("#F1F5F9")
LINE = colors.HexColor("#CBD5E1")
GOOD = colors.HexColor("#0F766E")

# ----- Styles -----
styles = getSampleStyleSheet()

title_style = ParagraphStyle(
    "Title", parent=styles["Title"],
    fontName="Helvetica-Bold", fontSize=28, textColor=ACCENT_DARK,
    alignment=TA_CENTER, spaceAfter=6, leading=32,
)
subtitle_style = ParagraphStyle(
    "Subtitle", parent=styles["Normal"],
    fontName="Helvetica", fontSize=13, textColor=MUTED,
    alignment=TA_CENTER, spaceAfter=4, leading=16,
)
small_center = ParagraphStyle(
    "SmallCenter", parent=styles["Normal"],
    fontName="Helvetica", fontSize=10, textColor=MUTED,
    alignment=TA_CENTER, leading=14,
)
h1 = ParagraphStyle(
    "H1", parent=styles["Heading1"],
    fontName="Helvetica-Bold", fontSize=18, textColor=ACCENT_DARK,
    spaceBefore=4, spaceAfter=8, leading=22,
)
h2 = ParagraphStyle(
    "H2", parent=styles["Heading2"],
    fontName="Helvetica-Bold", fontSize=13, textColor=INK,
    spaceBefore=8, spaceAfter=4, leading=16,
)
body = ParagraphStyle(
    "Body", parent=styles["Normal"],
    fontName="Helvetica", fontSize=10.5, textColor=INK,
    alignment=TA_JUSTIFY, leading=15, spaceAfter=6,
)
body_left = ParagraphStyle(
    "BodyLeft", parent=body, alignment=TA_LEFT,
)
quote = ParagraphStyle(
    "Quote", parent=body,
    fontName="Helvetica-Oblique", textColor=MUTED,
    leftIndent=14, rightIndent=14, spaceBefore=4, spaceAfter=6,
)
script_style = ParagraphStyle(
    "Script", parent=body,
    fontName="Helvetica", fontSize=10.5, textColor=INK,
    leftIndent=10, rightIndent=10, spaceBefore=4, spaceAfter=6,
    leading=15, backColor=PANEL, borderPadding=8,
)
code = ParagraphStyle(
    "Code", parent=styles["Code"],
    fontName="Courier", fontSize=9, textColor=INK, leading=12,
)
q_style = ParagraphStyle(
    "Q", parent=body_left,
    fontName="Helvetica-Bold", fontSize=11, textColor=ACCENT_DARK,
    spaceBefore=10, spaceAfter=2,
)
a_style = ParagraphStyle(
    "A", parent=body_left,
    fontSize=10.5, textColor=INK, spaceAfter=4, leading=15,
)

def rule():
    return HRFlowable(width="100%", thickness=0.6, color=LINE,
                      spaceBefore=4, spaceAfter=8)

# ============================================================
# Build the story
# ============================================================
story = []

# -------- COVER --------
story.append(Spacer(1, 4.5*cm))
story.append(Paragraph("SPECTRA", title_style))
story.append(Paragraph("Viva Voce Preparation Pack", subtitle_style))
story.append(Spacer(1, 0.2*cm))
story.append(Paragraph(
    "A 10-minute guide to defending your final-year project with confidence",
    small_center))
story.append(Spacer(1, 1.5*cm))
story.append(HRFlowable(width="40%", thickness=1, color=ACCENT,
                        hAlign="CENTER"))
story.append(Spacer(1, 0.6*cm))
story.append(Paragraph(
    "<b>Causal Physically-Guided Transparent-Object Segmentation</b>",
    small_center))
story.append(Spacer(1, 0.3*cm))
story.append(Paragraph(
    "Lalith Gona &nbsp;·&nbsp; Lovely Professional University &nbsp;·&nbsp; 2026",
    small_center))
story.append(Spacer(1, 6*cm))
story.append(Paragraph(
    "How to use this pack: read Section 1 out loud once. Skim "
    "Section 2 (architecture). Memorise the file walkthrough in Section 4. "
    "Practise the answers in Section 5 — these are the questions a professor "
    "actually asks.",
    small_center))
story.append(PageBreak())

# -------- 1. THE 60-SECOND PITCH --------
story.append(Paragraph("1.&nbsp; The 60-Second Pitch", h1))
story.append(Paragraph(
    "If the professor opens with <i>“In one minute, what is your project?”</i> — "
    "say exactly this:", body))
story.append(Paragraph(
    "“SPECTRA is a deep-learning system that detects and segments "
    "<b>transparent objects</b> — glass bottles, windows, clear plastic — in images. "
    "Standard segmentation networks fail at this because transparent surfaces have "
    "<b>no texture of their own</b>; they show whatever is behind them. My system "
    "solves this by giving the network a <b>physical signal</b> that opaque objects "
    "cannot produce: when the camera moves, light passing through glass refracts, so "
    "the optical flow behind the glass <b>disagrees</b> with the rigid-body camera "
    "motion. I expose this disagreement as a per-patch <b>consistency-violation</b> "
    "score and use it to <b>gate the features</b> inside a DINOv2 vision transformer "
    "before the segmentation head makes its decision. On the Trans10K benchmark my "
    "model reaches <b>0.9217 IoU</b> — beating U-Net, DeepLabV3+, SegFormer, and "
    "zero-shot SAM under identical training conditions.”",
    script_style))
story.append(rule())

# -------- 2. PROBLEM & IDEA --------
story.append(Paragraph("2.&nbsp; Problem &amp; Core Idea", h1))

story.append(Paragraph("The Problem", h2))
story.append(Paragraph(
    "Glass is invisible to a camera. The pixels behind a transparent surface look "
    "almost identical to the pixels around it, so any network that learns from "
    "texture and edges alone will silently segment the <i>background</i> instead "
    "of the glass. Standard CNN/ViT models plateau around IoU 0.87–0.89 on the "
    "Trans10K benchmark for this reason.", body))

story.append(Paragraph("The Idea — Two Physics Signals", h2))
idea_data = [
    [Paragraph("<b>OFCV</b><br/>Optical Flow<br/>Consistency Violation", body_left),
     Paragraph("<b>Physics:</b> Snell’s law — refraction bends light, so motion "
               "<i>behind</i> glass violates the rigid-body assumption of optical "
               "flow.<br/><br/><b>Effect:</b> Fires on glass surfaces, stays dark on "
               "opaque objects — even in the same scene.", body_left)],
    [Paragraph("<b>BRF</b><br/>Boundary<br/>Resonance Field", body_left),
     Paragraph("<b>Physics:</b> Glass edges produce a characteristic "
               "<i>double-edge</i> frequency signature (inner + outer surface) "
               "detectable by a Gabor filter bank.<br/><br/><b>Effect:</b> "
               "Sharpens glass boundaries as a full-resolution structural prior.",
               body_left)],
]
idea_tbl = Table(idea_data, colWidths=[4*cm, 12*cm])
idea_tbl.setStyle(TableStyle([
    ("VALIGN", (0,0), (-1,-1), "TOP"),
    ("BACKGROUND", (0,0), (0,-1), PANEL),
    ("BOX", (0,0), (-1,-1), 0.5, LINE),
    ("INNERGRID", (0,0), (-1,-1), 0.5, LINE),
    ("LEFTPADDING", (0,0), (-1,-1), 10),
    ("RIGHTPADDING", (0,0), (-1,-1), 10),
    ("TOPPADDING", (0,0), (-1,-1), 8),
    ("BOTTOMPADDING", (0,0), (-1,-1), 8),
]))
story.append(idea_tbl)
story.append(Spacer(1, 0.3*cm))
story.append(Paragraph(
    "<b>Key phrase to memorise:</b> <i>“OFCV and BRF are not learned features "
    "that re-discover physics — they ARE the physics, handed to the network "
    "directly.”</i>", body))

story.append(PageBreak())

# -------- 3. SYSTEM ARCHITECTURE --------
story.append(Paragraph("3.&nbsp; System Architecture", h1))
story.append(Paragraph(
    "Three input streams feed a fusion head; everything below is a single "
    "forward pass at inference.", body))

arch_diagram = """  RGB frame I_t   ----> DINOv2 ViT-S/14 (frozen)
                             |
                             v
                       Patch tokens (B, 384, 32, 32)
                             |
                             v
                       FPN multi-scale features
                             |
  RGB frame I_t+1 ----> RAFT (frozen)
                             |
                             v
                       OFCV Detector (residual + flow consistency)
                             |
                             v             ofcv_map (B, 1, 32, 32)
                             |
  RGB frame I_t   ----> Gabor bank (8 orient x 3 scales)
                             |
                             v             BRF map (B, 1, 448, 448)
                             |
                             v
                  +-------------------------+
                  |     FUSION HEAD         |
                  | tokens * sigmoid(OFCV)  |
                  |  (+) BRF projection     |
                  +-----------+-------------+
                              |
                              v
                seg_logits  +  mat_logits
                (mask)        (material class)
"""
arch_box = Preformatted(arch_diagram, code)
story.append(arch_box)
story.append(Spacer(1, 0.2*cm))

story.append(Paragraph("One-line summary per block", h2))
arch_tbl = Table([
    ["DINOv2 ViT-S/14", "Frozen self-supervised backbone — strong patch features without retraining"],
    ["RAFT", "Pretrained optical-flow estimator — produces forward/backward flow between two frames"],
    ["OFCV Detector", "Computes photometric residual + flow inconsistency, outputs a per-patch violation score"],
    ["Gabor Bank", "Non-trainable convolution with 24 oriented filters — extracts double-edge resonance"],
    ["Fusion Head", "Gates tokens by sigmoid(OFCV), concatenates BRF, decodes pixel-level mask"],
], colWidths=[4*cm, 12*cm])
arch_tbl.setStyle(TableStyle([
    ("FONT", (0,0), (-1,-1), "Helvetica", 9.5),
    ("FONT", (0,0), (0,-1), "Helvetica-Bold", 9.5),
    ("TEXTCOLOR", (0,0), (0,-1), ACCENT_DARK),
    ("BACKGROUND", (0,0), (0,-1), PANEL),
    ("VALIGN", (0,0), (-1,-1), "TOP"),
    ("LEFTPADDING", (0,0), (-1,-1), 6),
    ("RIGHTPADDING", (0,0), (-1,-1), 6),
    ("TOPPADDING", (0,0), (-1,-1), 5),
    ("BOTTOMPADDING", (0,0), (-1,-1), 5),
    ("GRID", (0,0), (-1,-1), 0.4, LINE),
]))
story.append(arch_tbl)

story.append(PageBreak())

# -------- 4. KEY RESULTS --------
story.append(Paragraph("4.&nbsp; Headline Results", h1))

story.append(Paragraph("Trans10K validation — Epoch 10", h2))
res_tbl = Table([
    ["Metric", "Value", "What it means"],
    ["IoU", "0.9217", "Intersection-over-Union — the gold-standard mask accuracy"],
    ["F-measure", "0.9560", "Harmonic mean of precision & recall"],
    ["MAE", "0.0293", "Mean absolute pixel error"],
    ["Test mean IoU", "0.9237", "Across all 4,428 test images — generalises"],
], colWidths=[3.5*cm, 2.5*cm, 10*cm])
res_tbl.setStyle(TableStyle([
    ("FONT", (0,0), (-1,0), "Helvetica-Bold", 10),
    ("BACKGROUND", (0,0), (-1,0), ACCENT),
    ("TEXTCOLOR", (0,0), (-1,0), colors.white),
    ("FONT", (0,1), (-1,-1), "Helvetica", 10),
    ("FONT", (1,1), (1,-1), "Helvetica-Bold", 10),
    ("TEXTCOLOR", (1,1), (1,-1), GOOD),
    ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, PANEL]),
    ("GRID", (0,0), (-1,-1), 0.4, LINE),
    ("LEFTPADDING", (0,0), (-1,-1), 6),
    ("RIGHTPADDING", (0,0), (-1,-1), 6),
    ("TOPPADDING", (0,0), (-1,-1), 5),
    ("BOTTOMPADDING", (0,0), (-1,-1), 5),
]))
story.append(res_tbl)
story.append(Spacer(1, 0.4*cm))

story.append(Paragraph("Matched-condition benchmark", h2))
story.append(Paragraph(
    "All four baselines trained on the <i>identical</i> Trans10K splits, "
    "<i>identical</i> 10-epoch schedule, <i>identical</i> metric pipeline.", body))
bench_tbl = Table([
    ["Model", "Params (M)", "Val IoU", "MAE"],
    ["SAM ViT-B (zero-shot)", "91 (frozen)", "0.1341", "0.292"],
    ["SegFormer-B0", "3.8", "0.8680", "0.057"],
    ["U-Net (ResNet-34)", "24.4", "0.8799", "0.051"],
    ["DeepLabV3+ (ResNet-50)", "39.6", "0.8856", "0.047"],
    ["SPECTRA (ours)", "26.6", "0.9217", "0.029"],
], colWidths=[6*cm, 3*cm, 3*cm, 3*cm])
bench_tbl.setStyle(TableStyle([
    ("FONT", (0,0), (-1,0), "Helvetica-Bold", 10),
    ("BACKGROUND", (0,0), (-1,0), ACCENT),
    ("TEXTCOLOR", (0,0), (-1,0), colors.white),
    ("FONT", (0,1), (-1,-1), "Helvetica", 10),
    ("BACKGROUND", (0,-1), (-1,-1), colors.HexColor("#DDEBFF")),
    ("FONT", (0,-1), (-1,-1), "Helvetica-Bold", 10),
    ("TEXTCOLOR", (0,-1), (-1,-1), ACCENT_DARK),
    ("ALIGN", (1,0), (-1,-1), "RIGHT"),
    ("GRID", (0,0), (-1,-1), 0.4, LINE),
    ("LEFTPADDING", (0,0), (-1,-1), 6),
    ("RIGHTPADDING", (0,0), (-1,-1), 6),
    ("TOPPADDING", (0,0), (-1,-1), 5),
    ("BOTTOMPADDING", (0,0), (-1,-1), 5),
]))
story.append(bench_tbl)
story.append(Spacer(1, 0.2*cm))
story.append(Paragraph(
    "<b>One-liner to say:</b> <i>“+0.036 IoU over the strongest baseline, "
    "–38% MAE, and 0.79 IoU over zero-shot SAM — meaning foundation models "
    "without task-specific training do not solve transparent-object segmentation.”</i>",
    body))

story.append(PageBreak())

# -------- 5. FILE WALKTHROUGH --------
story.append(Paragraph("5.&nbsp; File Walkthrough — “Open this, Say this”", h1))
story.append(Paragraph(
    "If the professor asks you to <i>show the code</i>, open these files in order. "
    "For each one, the right-hand column is the sentence to say out loud.",
    body))
story.append(Spacer(1, 0.2*cm))

walk = [
    ["#", "Open this file", "What to say"],
    ["1",
     "README.md",
     "“This is the project overview — it has the headline numbers, the architecture diagram, and instructions to run the web app on two terminals.”"],
    ["2",
     "paper/PAPER.md",
     "“This is the 6-page academic write-up, following standard sections: abstract, method, results, ablation, limitations.”"],
    ["3",
     "models/spectra_model.py",
     "“This is the top-level model. It wires together the DINOv2 backbone, the OFCV detector, the BRF module, and the fusion head. Forward pass starts here.”"],
    ["4",
     "models/backbone.py",
     "“This loads DINOv2 ViT-S/14 from torch hub and freezes it. We don’t retrain the backbone — we condition its features using physics.”"],
    ["5",
     "modules/ofcv_detector.py",
     "“This is the novel module. It takes two consecutive frames, runs RAFT to get optical flow, computes the photometric residual and the forward-backward flow inconsistency, and outputs a per-patch violation score. That score is the gate.”"],
    ["6",
     "modules/brf.py",
     "“This builds a Gabor filter bank — 8 orientations × 3 scales — and convolves it with the input. Locations where two parallel responses fire within a small window get summed; that is the double-edge signature of glass.”"],
    ["7",
     "modules/signal_fusion.py",
     "“The fusion head. It multiplies the DINOv2 tokens by sigmoid(OFCV) — the gate — then concatenates the BRF map after a learned projection, and decodes to a pixel-level mask.”"],
    ["8",
     "train/train_baseline.py",
     "“The training loop. Ten epochs, AdamW, mixed precision. The important detail is the variance regulariser on the OFCV output — without it, the gate collapses to a constant.”"],
    ["9",
     "eval/eval_checkpoints.py",
     "“Computes IoU, F-measure, MAE, BER on the validation set. This is the file that produced the 0.9217 number.”"],
    ["10",
     "api/server.py",
     "“FastAPI server, port 8080. It loads the trained checkpoint once and exposes /predict and /predict_full endpoints for the frontend.”"],
    ["11",
     "web/app/(app)/demo/page.tsx",
     "“The Next.js frontend at /demo. User uploads an image, the page POSTs it to /predict_full, and renders the segmentation overlay plus the OFCV, BRF, flow-residual, entropy, and material confidence heatmaps.”"],
]
walk_tbl = Table(walk, colWidths=[0.7*cm, 5*cm, 10.3*cm])
walk_tbl.setStyle(TableStyle([
    ("FONT", (0,0), (-1,0), "Helvetica-Bold", 10),
    ("BACKGROUND", (0,0), (-1,0), ACCENT),
    ("TEXTCOLOR", (0,0), (-1,0), colors.white),
    ("FONT", (0,1), (-1,-1), "Helvetica", 9.2),
    ("FONT", (1,1), (1,-1), "Courier-Bold", 8.8),
    ("TEXTCOLOR", (1,1), (1,-1), ACCENT_DARK),
    ("VALIGN", (0,0), (-1,-1), "TOP"),
    ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, PANEL]),
    ("GRID", (0,0), (-1,-1), 0.3, LINE),
    ("LEFTPADDING", (0,0), (-1,-1), 5),
    ("RIGHTPADDING", (0,0), (-1,-1), 5),
    ("TOPPADDING", (0,0), (-1,-1), 5),
    ("BOTTOMPADDING", (0,0), (-1,-1), 5),
]))
story.append(walk_tbl)

story.append(PageBreak())

# -------- 6. PREDICTED Q&A --------
story.append(Paragraph("6.&nbsp; Predicted Questions &amp; Answers", h1))
story.append(Paragraph(
    "Below are the twelve questions a professor is most likely to ask. "
    "Each answer is calibrated to <i>~30 seconds spoken</i> — long enough to "
    "show depth, short enough not to ramble.", body))

qa = [
    ("Q1.  How is your model exactly detecting transparent objects?",
     "“The model takes two consecutive frames. First, RAFT computes the optical "
     "flow between them. On opaque surfaces the flow matches rigid-body camera "
     "motion, but on transparent surfaces the flow is <b>biased by the refracted "
     "background</b> — this disagreement is the OFCV signal. My OFCV detector "
     "outputs a per-patch violation score, which is then used as a <b>gate</b> on "
     "the DINOv2 patch features. So the network only emphasises features in "
     "regions where physics says ‘something refractive is happening here’ — and "
     "that’s how it isolates glass.”"),

    ("Q2.  Which algorithm did you use, and where is it applied? In which file?",
     "“Three algorithms work together. <b>DINOv2 ViT-S/14</b> is the backbone "
     "(<i>models/backbone.py</i>) — it converts the image to patch tokens. "
     "<b>RAFT</b> is the optical-flow estimator (<i>flow/</i>) — it computes "
     "frame-to-frame motion. The novel core is <b>OFCV gating + BRF priors</b>, "
     "implemented in <i>modules/ofcv_detector.py</i> and <i>modules/brf.py</i>, "
     "fused in <i>modules/signal_fusion.py</i>. The whole pipeline is trained "
     "with cross-entropy + Dice + boundary loss in <i>train/train_baseline.py</i>.”"),

    ("Q3.  Why DINOv2 and not a CNN like ResNet?",
     "“DINOv2 is a vision transformer pretrained <b>self-supervised</b> on 142 "
     "million natural images. Its patch features already encode strong "
     "geometric and semantic priors without any labelled data. That makes it an "
     "ideal substrate to <b>condition</b> with physics signals — I don’t need to "
     "retrain it from scratch; I just gate it. A ResNet would also work, but "
     "DINOv2 gives a stronger starting point.”"),

    ("Q4.  Why is RAFT frozen? Why don’t you fine-tune it?",
     "“RAFT is already state-of-the-art for optical flow and was trained on "
     "millions of synthetic motion pairs. Fine-tuning it on Trans10K — which has "
     "no flow ground truth — would only degrade it. Freezing RAFT also keeps "
     "the OFCV signal <b>physically meaningful</b> rather than letting it drift "
     "into a learned shortcut.”"),

    ("Q5.  Your ablation shows OFCV/BRF don’t improve final accuracy. So why include them?",
     "“Three reasons. First, they cut convergence time — the model reaches the "
     "same IoU in <b>one epoch</b> instead of five. Second, they give "
     "<b>interpretability</b>: I can visualise the OFCV heatmap and see exactly "
     "where the model thinks glass is. Third, they improve <b>hard-case "
     "robustness</b> — under motion blur and glare the model degrades more "
     "gracefully. I state this openly in the paper because the honest framing "
     "is stronger than overclaiming accuracy.”"),

    ("Q6.  How do you handle a single still image at inference time? There’s no second frame.",
     "“It’s a real limitation. At deployment we pass the same image as both "
     "frames, which produces zero flow and degrades the OFCV signal. The "
     "<b>DINOv2 backbone and the BRF prior still work</b>, so accuracy is "
     "acceptable, but the full physics gate only fires on video. Future work "
     "would synthesise a plausible second frame from a single still using a "
     "small learned displacement.”"),

    ("Q7.  How do you know OFCV is genuinely learning physics and not just memorising the dataset?",
     "“Two pieces of evidence. First, the <b>OFCV variance grows during training "
     "and stays grown</b> — it doesn’t collapse to a constant, which is the "
     "usual sign of a degenerate gate. Second, in scenes containing both glass "
     "and opaque objects — like a glass bottle with a wooden roller inside — "
     "the OFCV map <b>fires on the glass and goes dark inside the opaque "
     "object</b>. No segmentation supervision distinguishes those two; the only "
     "way the model can tell them apart is by using the physical signal.”"),

    ("Q8.  Why is training 5× slower than U-Net? Is that worth it?",
     "“Two costs: DINOv2 is a transformer, so its forward pass is heavier than "
     "a ResNet; and RAFT runs once per training step to produce the flow. At "
     "<b>inference</b> the difference is smaller — a single forward pass, "
     "comparable latency. So the cost is only paid once, at training, in "
     "exchange for +0.036 IoU and interpretability.”"),

    ("Q9.  What dataset and why?",
     "“Trans10K — the standard benchmark for transparent-object segmentation, "
     "by Xie <i>et al.</i> 2020. It has 5,000 train, 1,000 val, and 4,428 test "
     "images of glass and clear-plastic objects. All four baselines and my "
     "model use the exact same splits and the exact same training schedule, so "
     "the comparison is fair.”"),

    ("Q10. What about robustness — what happens in bad lighting or motion blur?",
     "“I evaluated 8 corruption families × 5 severities. Clean IoU is 0.969 on "
     "the subset. At the worst severity, lighting, fog, JPEG compression, and "
     "colour jitter barely move it — all stay above 0.96. The two named hard "
     "cases — <b>glare</b> and <b>motion blur</b> — stay above <b>0.82</b>. "
     "Gaussian noise is the only real weakness at 0.76.”"),

    ("Q11. How is this different from prior glass-segmentation work like GDNet or TransLab?",
     "“Those models inject various <b>boundary</b> cues — my BRF module is in "
     "the same family. What is new in SPECTRA is the <b>optical-flow "
     "consistency gate</b>, OFCV. To my knowledge, using flow-consistency-as-"
     "gating is novel for transparent-object segmentation. Prior work also "
     "tends to be black-box; my system produces per-image transparency maps a "
     "human can read.”"),

    ("Q12. What are the main limitations and what would you do next?",
     "“Four. (1) Single-dataset evaluation — only Trans10K, no GSD or "
     "ClearPose yet. (2) Two-frame requirement at inference. (3) BRF behaves "
     "more like a static prior than a per-image signal. (4) Ablation is only "
     "5 epochs. Next steps: cross-dataset evaluation, learned synthetic second "
     "frame, a richer material classification head, and a full 30-epoch "
     "ablation to nail down late-epoch behaviour.”"),
]

for q, a in qa:
    block = KeepTogether([
        Paragraph(q, q_style),
        Paragraph(a, a_style),
    ])
    story.append(block)

story.append(Spacer(1, 0.5*cm))
story.append(rule())
story.append(Paragraph(
    "<b>Final tip:</b> if a question stumps you, say "
    "<i>“That’s a good question — let me think about it.”</i> Then anchor your "
    "answer to one of: the <b>OFCV gate</b>, the <b>physics motivation</b>, the "
    "<b>matched-condition benchmark</b>, or the <b>interpretability result</b>. "
    "Every honest answer in this project lives near one of those four pillars.",
    body))

story.append(Spacer(1, 0.5*cm))
story.append(Paragraph(
    "Lalith Gona &nbsp;·&nbsp; SPECTRA &nbsp;·&nbsp; LPU 2026",
    small_center))

# ============================================================
# Build PDF
# ============================================================
doc = SimpleDocTemplate(
    OUT, pagesize=A4,
    leftMargin=2.2*cm, rightMargin=2.2*cm,
    topMargin=2*cm, bottomMargin=2*cm,
    title="SPECTRA — Viva Voce Prep Pack",
    author="Lalith Gona",
)

def _on_page(canvas, doc_):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(MUTED)
    # Footer rule
    canvas.setStrokeColor(LINE)
    canvas.setLineWidth(0.4)
    canvas.line(2.2*cm, 1.5*cm, A4[0] - 2.2*cm, 1.5*cm)
    # Footer text
    if doc_.page > 1:
        canvas.drawString(2.2*cm, 1.0*cm, "SPECTRA — Viva Voce Prep")
        canvas.drawRightString(A4[0] - 2.2*cm, 1.0*cm, f"Page {doc_.page}")
    canvas.restoreState()

doc.build(story, onFirstPage=_on_page, onLaterPages=_on_page)
print(f"OK -> {OUT}")
