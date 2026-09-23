# KINE-Bench

Evidence-first evaluation tools for compact world models on consumer hardware.

KINE-Bench is a KineWorld-authored research harness, **not** an official third-party leaderboard. It publishes protocols, baselines, machine-readable results and negative findings so that claims can be inspected rather than inferred from demos.

## Current public evidence

The current headline artifact is `KW-RISK-CARD-0001`, derived from an internal reproduction of a public JEPA-WM Push-T checkpoint:

- 96 episodes and 576 horizon-error observations;
- task success: 44/96 (45.83%);
- high object-displacement spike rate: 7.50%, versus 1.39% in the lower-displacement slice;
- episode-cluster robustness audit: +6.11 percentage points, permutation p=0.00320, bootstrap 95% CI [+1.67, +10.73] percentage points;
- the proposed physical-risk bands were 8–22× wider than the pooled baseline and therefore failed the product-utility gate.

This is **E1 internal evidence**. It does not establish causality, independent validation, deployment utility or model leadership. KineWorld did not train the evaluated public checkpoint.

## Earlier probes

The repository also contains historical KINE-EXP-001 frozen-representation probes. Their published results are retained for reproducibility, including results at or below baseline. They are exploratory diagnostics, not a claim that KineWorld has trained a frontier model.

## Run the harness

```bash
git clone https://github.com/kineworld/kine-bench.git
git clone https://github.com/kineworld/kine-jepa.git
cd kine-bench
python -m pip install -r requirements.txt
python -m kinebench run --smoke --max-clips 8 --device cpu
```

Adapters report unsupported capabilities as `n/a`; they do not manufacture comparable scores for missing interfaces.

For offline V-JEPA 2 evaluation, install `requirements-vjepa2.txt` and point
`KINE_VJEPA2_LOCAL` at the local Hugging Face checkpoint directory. The adapter
loads all safetensors shards through Transformers and rejects missing, extra or
incompatible parameters before scoring. The optional CI job checks this with
small synthetic local weights; it does not download or redistribute Meta weights.

## Independent verification

Passing KineWorld's integrity checks only proves artifact consistency. Evidence becomes independent only when an external evaluator obtains the named upstream checkpoint, reruns the frozen protocol, retains raw logs and signs the supplied attestation with all deviations disclosed.

Website: [kineworld.com](https://kineworld.com)

## License

KineWorld-authored code is MIT licensed. Model weights and datasets follow their respective upstream licenses.
