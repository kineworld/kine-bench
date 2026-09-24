# KineWorld Technical Radar — 2026 Q3

Snapshot: 2026-09-24. Scope: anti-collapse objectives, physically-grounded latent
dynamics, and world-model evaluation methodology for compact action-conditioned
world models on consumer hardware.

Reading rules (unchanged from the `spatial-0` radar):

- A paper result is a **discovery signal**, not an adoption decision.
- Adoption requires an identical-input run on this laptop plus a separate
  code / weight / license audit.
- Anything whose anti-collapse pressure or evaluation protocol we adopt must be
  attributed, and the adopted claim must be reproducible from a script, not
  hand-typed.

## 1. Why this radar exists

`kine-jepa` trains an online encoder, an EMA target encoder and a transformer
predictor with a single L1 latent-prediction loss and **no explicit
anti-collapse term**. `STATUS.md` already records FUT improving while
TEMP / MOT / EVT / EMB fall across 10k steps. That pattern is the textbook
signature of partial representation collapse: the predictor gets cheaper to fit
because the target geometry loses information, and the pooled-feature probes
lose discriminability with it.

Three independent 2026 lines of work say the same thing from different angles:
**the anti-collapse pressure does not have to come from a distributional prior
(EMA + stop-gradient + SIGReg); it can come from the transition data itself.**
This radar records the candidates, their licenses and their adoption status.

## 2. Anti-collapse / action-sensitivity objectives

| Candidate | What it contributes | License / weight terms | Decision |
|---|---|---|---|
| **AC-MTM** — Action-Contrastive Masked Transition Modeling (arXiv:2608.17542) | Training-only inverse-dynamics head trained with Action-NCE: each latent transition must identify the action that produced it among batch negatives. A collapsed encoder provably fails this discrimination. Inverse branch is discarded at test time, so encoding / planning / compute are unchanged. | Code MIT (github.com/jackboyla/action-contrastive-jepa); paper CC-BY | **ADOPT as primary anti-collapse mechanism** |
| **Delta-JEPA** — Latent Difference Action Decoder (arXiv:2606.31232) | Reconstruct the action from the *latent displacement* between consecutive observations rather than from concatenated endpoint embeddings. Displacement-level supervision regularises transition geometry directly. Ablations: displacement decoding beats endpoint concatenation consistently. | Code: none released at v1 | **ADOPT the loss idea; reimplement in-house** |
| **PhyLatent** (arXiv:2608.05720) | Names three failure modes beyond global collapse: *physical invariance collapse*, *physical identifiability collapse*, *counterfactual dynamics collapse*. Reports OGBench-Cube failure rates 15.60/6.71/8.41% → 7.53/0.95/4.62%; MPC success 70.0% → 78.1%; TwoRooms 81.0% → 98.0%. | Code: none released | **ADOPT as diagnostic taxonomy** |
| **VISReg** — Variance-Invariance-Sketching Regularization (arXiv, 2026-07) | Decouples the anti-collapse regulariser into independent *scale* and *shape* objectives. Reports matching DINOv2 OOD with ~1/10 training data. | Code MIT + weights public (github.com/HaiyuWu/visreg) | WATCH — heavier than needed for ViT-S scale |
| SIGReg (LeWM) | Forces latent distribution toward isotropic Gaussian. The incumbent prior-based mechanism we are moving away from. | permissive upstream | RETAIN as baseline arm, not as the main line |

**Council position.** AC-MTM and Delta-JEPA are complementary, not competing:
AC-MTM supplies the *discriminative* pressure (transitions must be identifiable),
Delta-JEPA supplies the *geometric* pressure (displacements must be correct). A
single inverse head can satisfy both — predict the action from the displacement
under a contrastive objective. PhyLatent supplies the *diagnostic vocabulary*
used to check the result. See `kine-jepa` private training recipe for the
combined implementation.

## 3. Physical grounding for goal-conditioned planning

| Candidate | What it contributes | License | Decision |
|---|---|---|---|
| **State Alignment + IDM** (arXiv:2609.03565v2, GENISOM AI) | IDM discourages collapse; state alignment grounds consecutive latents in physical configuration. TwoRoom 100%, PushT 98%, OGBench-Cube 87%. Transition-subspace analysis: state-aligned models have higher effective transition dimension than the baseline despite lower "straightening". | Code: none released | **ADOPT the effective-transition-dimension instrument** |
| **Delta-JEPA action-sensitivity analysis** | Action-conditioned latent response clarity as a reported diagnostic | — | ADOPT as a reported metric |
| Point-cloud JEPA (arXiv:2608.29434) | Lifts three canonical JEPA designs to point clouds; shows object positions almost linearly decodable, attention lands on moving points; goal latent constructible from target + current at no success cost | code permissive (check at adoption) | WATCH — geometric observation is `spatial-0` domain, not `kine-jepa` |

**Note on "straightening" vs effective dimension.** arXiv:2609.03565 reports a
case where the *higher* straightening baseline has transition energy concentrated
in a **lower**-dimensional subspace, and the state-aligned model that plans
better has higher effective transition dimension. Practical consequence: we
report effective dimension alongside any single scalar "straightening" number,
because straightening alone can reward a collapsed transition geometry.

## 4. Evaluation methodology

| Candidate | What it contributes | License | Decision |
|---|---|---|---|
| **CrashTwin** (arXiv:2606.28757) | Calibration-free reconstruction pipeline recovering metric-scale kinematics from uncalibrated monocular rollouts; diagnostic suite over spatio-temporal consistency, momentum/energy conservation, world-dynamics integrity. Key finding: perceptual quality frequently masks severe physical violations. | paper CC-BY; dataset terms check at use | **ADOPT the protocol idea** (calibration-free physical attribute recovery), not the collision dataset |
| **PlayWorld** (arXiv:2608.13552v2) | Closed-loop agent-player benchmark over long-horizon objectives; VQA rubric verifier with trajectory-validity gating. Best of nine models scores 2.12/5. Key point: identical controls produce different movement magnitudes across models, so fixed action scripts conflate trajectory failure with geometric inconsistency. | paper; check code terms | ADOPT the *trajectory-validity gating* idea: do not score an interaction that never happened |
| **WorldRoamBench** (arXiv:2606.31672v4) | Per-frame action metric that bypasses cross-model semantic-scale disparity; sliding-window drift metric catching non-monotonic mid-sequence collapse missed by start-vs-end comparison. Asserts **not monotonic** degradation, so end-point comparison is insufficient. | AMAP CV Lab; check terms | **ADOPT the sliding-window drift metric** |
| **HappyWorld-Bench** (arXiv:2609.24308) | Six-capability hierarchy (W1–W6) across video / spatial / embodied tracks; 14 video + 9 spatial + 8 embodied systems; reports spatial models at best 70.14% placement accuracy, embodied models failing to preserve state across multi-step actions | check terms | WATCH — overlaps our spatial track |

**Common thread across all four benchmarks.** End-point or fixed-script
evaluation systematically overstates capability. Every one of them independently
arrived at a *windowed or gated* protocol. This is the strongest methodological
signal in the whole radar and drives the representational-health probes below.

## 5. What lands where (open / closed boundary)

This radar does not change the boundary in `OPEN_SOURCE_BOUNDARY.md`; it applies it.

- **Public (`kine-bench`, this repo):** the *measurability* of representational
  health — effective rank, dimension utilisation, displacement identifiability,
  sliding-window drift. Every metric here is a *diagnostic*; publishing it tells
  competitors what to measure, not how to fix it.
- **Private (`kine-jepa` internal training recipe):** the *combined objective* —
  how the contrastive inverse head is weighted, warmup-scheduled and coupled to
  the forward loss, and how the EMA momentum is retuned around it. This is the
  post-training recipe class already listed as a barrier asset.
- **Unchanged:** weights, action annotations, data mixtures, serving graphs.

## 6. Attribution obligations

Any code adopted from AC-MTM / VISReg / upstream stable-worldmodel must keep the
upstream `NOTICE` and cite the paper. Delta-JEPA, PhyLatent and arXiv:2609.03565
released no code, so their contributions are reimplemented from the paper; the
implementation is original KineWorld code and the paper is cited as the source
of the idea, not of the code.

## 7. Open questions this radar does not answer

1. Whether the contrastive inverse head stays stable at ViT-S / 16-frame /
   224px scale, or whether batch negative count is too small at batch-size 8 to
   give a usable discrimination signal. **Must be measured, not assumed.**
2. Whether displacement decoding and contrastive decoding double-count the same
   gradient or genuinely compose.
3. Whether the sliding-window drift metric is meaningful on the synthetic smoke
   distribution or only on real motion — synthetic clips are deterministic ramps
   and may have a degenerate drift profile.

Unresolved items are recorded as open questions, not as adopted claims.
