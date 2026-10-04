# Cover letter / Comments to the Action Editor — TMLR submission

**Title:** SPECTRA: Representation over Refraction — A Controlled Study of Physics-Motivated Cues for Transparent-Object Segmentation

**Submission type:** Empirical analysis / controlled negative result.

Dear Action Editor,

We submit this paper to TMLR because its contribution is *correctness and rigor*,
not novelty or state-of-the-art performance — which we understand to be exactly
what TMLR's acceptance criteria reward. We address the two criteria directly.

## Criterion 1 — Are the claims supported by accurate, convincing, clear evidence?

The paper makes three claims, each tied to a specific experiment, and we have
been careful not to claim anything beyond what the data shows.

1. **Two representative physics-motivated cues (an optical-flow consistency gate
   and a Gabor boundary prior) add no accuracy to a strong self-supervised
   backbone on this task.** Evidence: a controlled five-variant × 30-epoch ×
   three-seed ablation in which the backbone is held fixed and only the cues
   vary. The entire between-variant spread is Δ = 0.0033 IoU, while the seed
   standard deviation is σ = 0.0008 — i.e., the effect is an order of magnitude
   below noise (Sec. "Ablation Studies", Tables IV–V).

2. **The learned flow cue is inert even given genuine motion.** Evidence: a
   real-video control on the VSGD benchmark. Supplying the true next frame
   instead of a duplicated frame changes the OFCV map by < 0.05% and the
   prediction by < 0.001%; the two settings are identical to three decimals on
   every metric (Sec. "Cross-Dataset…", Table VI). This converts a theoretical
   degeneracy argument (Sec. "Theoretical Analysis") into a direct empirical
   result.

3. **The accuracy that exists is a representation effect, not a method effect.**
   A DINOv2 backbone with a plain decoder ("neither" variant) already reaches
   0.932 IoU, and the modules leave this unchanged. We explicitly flag that our
   matched-budget comparison to ImageNet-pretrained CNN baselines is *confounded*
   (self-supervised ViT vs. ImageNet CNN encoders) and report it only as context
   for the operating point — never as a result. The central claim rests on the
   backbone-controlled ablation, which is not affected by that confound.

We have deliberately avoided three things that would weaken the evidential
standard: (i) we do not compare against published Trans10K leaderboard numbers,
because they use a different (multi-class mIoU) protocol than our binary
foreground IoU and are not comparable; (ii) we do not claim our 0.92 IoU as a
win; and (iii) we scope the refutation to the two cues and backbone we tested,
rather than to "all physical priors."

## Criterion 2 — Would some of TMLR's audience be interested?

Physics-motivated and "causal" inductive biases are increasingly proposed for
ill-posed perception problems, and are frequently validated only by a single
headline metric on one dataset against weaker backbones. We show, on a task
where such a prior is especially intuitive (refraction for transparency), that a
cue can be simultaneously interpretable, non-collapsing, and entirely inert —
and that an unexamined backbone advantage can be mistaken for a method effect.
We believe researchers working on physics-guided learning, self-supervised
representations, and transparent-object/glass perception will find both the
result and the controlled methodology (including the real-video falsification and
the explicit confound disclosure) useful as a template for testing rather than
assuming that a prior contributes.

## Reproducibility

All experiments use public data (Trans10K, ClearPose, VSGD), fixed seeds
{7, 13, 42}, deterministic cuDNN, and a single consumer GPU. All checkpoints,
configs, per-epoch logs, and metric files are released, and every number in the
paper is traceable to a released artifact.

## Honest disclosure of limitations

We state plainly (Sec. "Limitations") that the headline comparison is confounded
and would need a converged-CNN baseline and a fixed-decoder backbone-swap to be
turned into a clean backbone result; that the refutation covers two cues on one
backbone; and that cross-dataset transfer to ClearPose is weak (0.535 zero-shot).
None of these affect the central, backbone-controlled claim.

We would welcome reviewer guidance on additional controls that would further
strengthen the attribution analysis.

Sincerely,
Lalith Gona
Lovely Professional University
gonalalith2005@gmail.com
