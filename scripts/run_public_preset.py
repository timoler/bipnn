"""MIS diagnostic using the public notebook's Max-Cut network/training preset.

Not a reproduction of the original paper's MIS experiments (none are claimed).
The objective and feasible decoder are adapted to the explicitly chosen MIS.
"""
import argparse
import hashlib
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from dhg import Hypergraph

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mis.problem import HypergraphProblem, repair_and_extend
from mis.neural import polynomial_tensors
from mis.verify_report import verify
from mis.benchmark import write_json
from src.models import Layer, LayerType, Net
from src.train import train
from src.loss import loss_pubo


def solve(problem, mode, seed, epochs):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    edges, h, c = polynomial_tensors(problem, mode, 2.0, "cuda")
    graph = Hypergraph(problem.n, edges, device=torch.device("cuda"))
    layers = [Layer(LayerType.HGNNPCONV, 1024, 1024),
              Layer(LayerType.HGNNPCONV, 1024, 512, last_conv=True),
              Layer(LayerType.LINEAR, 512, 1, drop_rate=0.5)]
    net = Net(layers).cuda()
    features = torch.rand(problem.n, 1024, device="cuda")
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-4, weight_decay=5e-8)
    trace = []
    for epoch in range(1, epochs+1):
        loss, outs = train(net, features, graph, optimizer, loss_pubo,
                           H=h, C=c.unsqueeze(0), epoch=epoch, acl=lambda e: -1+e/1000)
        if not math.isfinite(loss):
            raise FloatingPointError("Nonfinite public-preset training loss")
        if epoch == 1 or epoch % 200 == 0:
            trace.append({"epoch": epoch, "loss_before_update": loss})
    # Original run() returns the last training forward pass, before its update.
    scores = outs.detach().flatten().cpu().tolist()
    raw = [i for i, p in enumerate(scores) if p >= 0.5]
    selected = repair_and_extend(problem, scores, mode)
    if not problem.feasible(selected, mode):
        raise AssertionError("Decoded solution failed verification")
    torch.cuda.synchronize()
    return {"method": "public_notebook_preset_mis", "seed": seed,
            "selected": selected, "size": len(selected), "feasible": True,
            "raw_selected": raw, "raw_size": len(raw), "raw_feasible": problem.feasible(raw, mode),
            "scores": scores, "total_seconds": time.perf_counter()-start,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(), "trace": trace,
            "settings": {"epochs": epochs, "feature_dim": 1024, "hidden": [1024,512],
                         "lr": 1e-4, "weight_decay": 5e-8, "penalty": 2.0,
                         "annealing": "-1 + epoch/1000", "gradient_clipping": False,
                         "network_wrapper": "unmodified src.models", "training_step": "unmodified src.train.train",
                         "loss": "unmodified src.loss.loss_pubo", "output": "last training forward pass before update"}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("suite", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sizes", nargs="+", type=int, default=[50,200,500])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0,1,2])
    parser.add_argument("--epochs", type=int, default=1600)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this diagnostic")
    manifest = {"preset_source": "public example/desc.ipynb Max-Cut configuration",
                "upstream_commit": "708c10a95a50fb2b06dd35bf1604e65a6ec5ebe5",
                "selection_rule": "one k=3,m=2n,seed=20261018 instance per scale, both definitions, all three seeds",
                "limits": "different training budget from first suite; configuration diagnostic, not equal-time superiority evidence",
                "torch": torch.__version__, "gpu": torch.cuda.get_device_name(),
                "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "public_source_sha256": {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest()
                                          for p in ['src/models.py','src/train.py','src/loss.py']}}
    write_json(args.output / "manifest.json", manifest)
    for n in args.sizes:
        name = f"n{n}_k3_d2_s20261018"
        for mode in ["weak", "strong"]:
            output = args.output / "cases" / f"{name}_{mode}.json"
            if output.exists():
                verify(output)
                continue
            original = json.loads((args.suite/"cases"/f"{name}_{mode}.json").read_text(encoding="utf8"))
            instance = original["instance"]
            problem = HypergraphProblem(instance["n"], tuple(tuple(e) for e in instance["edges"]))
            records = []
            for seed in args.seeds:
                result = solve(problem, mode, seed, args.epochs)
                records.append(result)
                print(f"PUBLIC_PRESET n={n} {mode} seed={seed} size={result['size']} "
                      f"raw={result['raw_size']} feasible={result['raw_feasible']} t={result['total_seconds']:.2f}", flush=True)
            report = {"purpose": "public-code preset diagnostic with MIS objective adaptation",
                      "definition": mode, "instance": instance, "environment": original["environment"],
                      "public_preset_manifest": manifest, "records": records}
            write_json(output, report)
            verify(output)


if __name__ == "__main__":
    main()
