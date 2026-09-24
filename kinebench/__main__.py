"""KINE-Bench CLI: python -m kinebench run [--model M] [--ckpt P] [--data-dir D] [--smoke]"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch

from .adapters import PROBE_CAPABILITY, build_adapter, describe_all, list_adapters
from .adapters.base import OPTIONAL_PARTS
from .cau import intervention_auc
from .emb import embodied_imagination
from .events import event_shift
from .metrics import ALL_TASKS
from .synth import synthetic_clips

VERSION = "0.5.0"


def build_parser():
    ap = argparse.ArgumentParser(prog="kinebench", description=f"KINE-Bench v{VERSION} evaluation harness")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("run", help="run all tasks on a model")
    p.add_argument("--model", type=str, default="kineone-wm",
                   help="adapter alias (see --list-models) or kineone-wm:<ckpt>")
    p.add_argument("--ckpt", type=str, default=None,
                   help="checkpoint path for the kineone-wm adapter")
    p.add_argument("--data-dir", type=str, default=None)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--max-clips", type=int, default=48)
    p.add_argument("--num-frames", type=int, default=16)
    p.add_argument("--img-size", type=int, default=None)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--raw-dir", type=str, default=None)
    p.add_argument("--events-json", type=str, default=None)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--out", type=str, default=None)

    sub.add_parser("models", help="list benchmarkable models and their capabilities")

    r = sub.add_parser(
        "rep",
        help="KINE-REP-1: representational-health probes (collapse diagnostics)",
    )
    r.add_argument("--model", type=str, default="kineone-wm",
                   help="adapter alias or kineone-wm:<ckpt>")
    r.add_argument("--ckpt", type=str, default=None)
    r.add_argument("--data-dir", type=str, default=None)
    r.add_argument("--smoke", action="store_true")
    r.add_argument("--max-clips", type=int, default=48)
    r.add_argument("--num-frames", type=int, default=16)
    r.add_argument("--img-size", type=int, default=None)
    r.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    r.add_argument("--batch-size", type=int, default=2)
    r.add_argument("--out", type=str, default=None)
    return ap


def resolve_event_inputs(args):
    if args.raw_dir and args.events_json:
        return Path(args.raw_dir), Path(args.events_json)
    if args.data_dir:
        base = Path(args.data_dir)
        for c in (base, base.parent):
            if (c / "events.json").is_file() and (c / "raw").is_dir():
                return c / "raw", c / "events.json"
    return None


def load_clips(args):
    if args.smoke or not args.data_dir:
        n = args.max_clips if not args.smoke else min(8, args.max_clips)
        clips = synthetic_clips(n, num_frames=args.num_frames, size=args.img_size or 64, seed=0)
        return clips, f"synthetic({n})"
    from .load import ensure_jepa_path

    ensure_jepa_path()
    from kineworld_jepa.dataset import VideoClipDataset

    ds = VideoClipDataset(args.data_dir, num_frames=args.num_frames, size=args.img_size or 224)
    n = min(len(ds), args.max_clips)
    clips = [ds[i] for i in range(n)]
    return clips, f"{args.data_dir} ({n}/{len(ds)} clips)"


def record(results, adapter, name, payload):
    """Store a probe result, annotating optional parts the model cannot serve."""
    if payload is None:
        return
    if isinstance(payload, dict):
        payload = dict(payload)
        payload.setdefault("status", "ok")
        missing = OPTIONAL_PARTS.get(name)
        if missing and missing not in adapter.capabilities:
            payload["degraded"] = (
                f"{name} '{missing}' branch unavailable: auc_do not measured"
            )
    results["tasks"][name] = payload


def resolve_device(choice):
    if choice == "cpu":
        return torch.device("cpu")
    if choice == "cuda":
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_rep(args):
    """KINE-REP-1 CLI: report representational health for one adapter.

    Deliberately standalone from `run`: the diagnostics only need an encoder and
    a set of clips, so they can be applied to any model regardless of whether it
    exposes the full planning interface.
    """
    from .rephealth import representational_health

    device = resolve_device(args.device)
    img_size = args.img_size or (64 if (args.smoke or not args.data_dir) else 224)
    args.img_size = img_size

    clips, source = load_clips(args)
    adapter = build_adapter(
        args.model, ckpt=args.ckpt, img_size=img_size,
        num_frames=args.num_frames, batch_size=args.batch_size,
    )
    print(f"[rep] data: {source} | device: {device}")
    print(f"[rep] model: {adapter.info.name} <{adapter.info.source}> license={adapter.info.license}")
    model = adapter.build(str(device))

    result = representational_health(model, clips, device=str(device))
    payload = {
        "harness": "kinebench",
        "version": VERSION,
        "probe": "KINE-REP-1",
        "model": adapter.describe(),
        "checkpoint": args.ckpt,
        "data": source,
        "num_frames": args.num_frames,
        "img_size": img_size,
        "device": str(device),
        "result": result,
    }
    print(f"[rep] KINE-REP-1: {json.dumps(result, ensure_ascii=False)}")
    if result.get("displacement", {}).get("status") == "unavailable":
        print("[rep] note: the displacement probe needs >= 8 displacements per clip; "
              "this run did not meet that, so that field is None rather than a score.")
    if args.out:
        Path(args.out).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[rep] results -> {args.out}")
    return 0


def main(argv=None):
    args = build_parser().parse_args(argv)

    if args.cmd == "models":
        print(json.dumps(describe_all(), indent=2, ensure_ascii=False))
        return 0

    if args.cmd == "rep":
        return run_rep(args)

    if args.cmd != "run":
        return 1

    if args.device == "cpu":
        device = torch.device("cpu")
    elif args.device == "cuda":
        device = torch.device("cuda")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    img_size = args.img_size or (64 if (args.smoke or not args.data_dir) else 224)
    args.img_size = img_size

    t0 = time.time()
    clips, source = load_clips(args)

    adapter = build_adapter(
        args.model, ckpt=args.ckpt, img_size=img_size,
        num_frames=args.num_frames, batch_size=args.batch_size,
    )
    print(f"[bench] data: {source} | device: {device}")
    print(f"[bench] model: {adapter.info.name} <{adapter.info.source}> "
          f"license={adapter.info.license} caps={sorted(adapter.capabilities)}")
    model = adapter.build(str(device))

    results = {
        "harness": "kinebench",
        "version": VERSION,
        "model": adapter.describe(),
        "checkpoint": args.ckpt,
        "data": source,
        "num_clips": len(clips),
        "num_frames": args.num_frames,
        "img_size": img_size,
        "device": str(device),
        "tasks": {},
        "not_applicable": {},
    }

    grid = getattr(model, "grid", None)
    if grid is not None:
        print(f"[bench] grid {tuple(grid)}")

    for name, fn in ALL_TASKS:
        if not adapter.supports(name):
            results["not_applicable"][name] = adapter.unavailable(name)
            print(f"[bench] {name}: n/a ({PROBE_CAPABILITY[name]} not exposed)")
            continue
        r = fn(model, clips, device)
        record(results, adapter, name, r)
        print(f"[bench] {name}: {r}")

    evt = resolve_event_inputs(args)
    if "KINE-EVT-1" in results["not_applicable"]:
        pass
    elif evt is not None:
        raw_dir, events_json = evt
        print(f"[bench] KINE-EVT-1 inputs: {raw_dir} + {events_json.name}")
        r = event_shift(model, raw_dir, events_json, device,
                        num_frames=args.num_frames, img_size=img_size)
        record(results, adapter, "KINE-EVT-1", r)
        print(f"[bench] KINE-EVT-1: {r}")
    else:
        print("[bench] KINE-EVT-1 skipped (no raw videos + events.json found)")

    if adapter.supports("KINE-EMB-1"):
        r = embodied_imagination(model, device, num_frames=args.num_frames, img_size=img_size)
        if r.get("error"):
            print(f"[bench] KINE-EMB-1 skipped ({r['error']})")
        else:
            record(results, adapter, "KINE-EMB-1", r)
            print(f"[bench] KINE-EMB-1: {r}")
    else:
        results["not_applicable"]["KINE-EMB-1"] = adapter.unavailable("KINE-EMB-1")
        print("[bench] KINE-EMB-1: n/a (predict not exposed)")

    if adapter.supports("KINE-CAU-1"):
        cau_size = 64 if args.smoke or img_size < 128 else min(img_size, 128)
        r = intervention_auc(model, device, n_pairs=16,
                             frames=min(args.num_frames, 16), size=cau_size)
        record(results, adapter, "KINE-CAU-1", r)
        print(f"[bench] KINE-CAU-1: {r}")

    results["wall_s"] = round(time.time() - t0, 1)

    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[bench] results -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
