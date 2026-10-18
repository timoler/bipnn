"""GPU microbenchmark of mathematically equivalent dense and indexed losses.

This measures loss forward/backward only. It does not measure training, solution
quality, or identical optimization trajectories. Every requested case is saved.
"""
import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mis.benchmark import generate_instance, write_json
from mis.indexed_loss import indexed_polynomial_loss, indexed_tensors
from mis.neural import polynomial_loss, polynomial_tensors
from mis.problem import HypergraphProblem


def benchmark_kernel(loss_fn, values, tensors, *, warmup, blocks, iterations, device):
    p = values.detach().clone().requires_grad_()

    def step():
        p.grad = None
        loss_fn(p, *tensors).backward()

    for _ in range(warmup):
        step()
    torch.cuda.synchronize(device)
    p.grad = None
    gc.collect()
    baseline = torch.cuda.memory_allocated(device)
    torch.cuda.reset_peak_memory_stats(device)
    seconds = []
    for _ in range(blocks):
        torch.cuda.synchronize(device)
        start = time.perf_counter()
        for _ in range(iterations):
            step()
        torch.cuda.synchronize(device)
        seconds.append((time.perf_counter() - start) / iterations)
    peak = torch.cuda.max_memory_allocated(device)
    return {"forward_backward_seconds_per_iteration": seconds,
            "median_seconds": statistics.median(seconds),
            "mean_seconds": statistics.mean(seconds),
            "baseline_allocated_bytes": baseline,
            "peak_allocated_bytes": peak,
            "incremental_peak_allocated_bytes": peak - baseline}


def run_case(n, k, density, seed, definition, args):
    instance = generate_instance(n, k, density, seed)
    problem = HypergraphProblem(instance["n"], tuple(tuple(e) for e in instance["edges"]))
    generator = torch.Generator().manual_seed(seed)
    cpu_values = 0.05 + 0.9 * torch.rand(n, generator=generator)
    values = cpu_values.to(args.device)
    edges, incidence, dense_coefs = polynomial_tensors(
        problem, definition, 2.0, args.device)
    indexed_edges, indices, indexed_coefs = indexed_tensors(
        problem, definition, 2.0, args.device)
    if edges != indexed_edges:
        raise AssertionError("Term ordering changed")
    check_p = values.detach().clone().requires_grad_()
    dense_loss = polynomial_loss(check_p, incidence, dense_coefs)
    dense_grad = torch.autograd.grad(dense_loss, check_p)[0]
    indexed_loss = indexed_polynomial_loss(check_p, indices, indexed_coefs)
    indexed_grad = torch.autograd.grad(indexed_loss, check_p)[0]
    torch.testing.assert_close(indexed_loss, dense_loss, rtol=5e-5, atol=1e-3)
    torch.testing.assert_close(indexed_grad, dense_grad, rtol=5e-4, atol=5e-5)
    agreement = {"dense_loss": float(dense_loss.detach()),
                 "indexed_loss": float(indexed_loss.detach()),
                 "absolute_loss_difference": float((dense_loss-indexed_loss).abs().detach()),
                 "maximum_absolute_gradient_difference": float((dense_grad-indexed_grad).abs().max()),
                 "float32_tolerances": {"loss_rtol": 5e-5, "loss_atol": 1e-3,
                                        "gradient_rtol": 5e-4, "gradient_atol": 5e-5}}
    del dense_loss, indexed_loss, dense_grad, indexed_grad, check_p
    common = {"warmup": args.warmup, "blocks": args.blocks,
              "iterations": args.iterations, "device": args.device}
    # Both static representations remain resident for both measurements.
    # incremental_peak subtracts this shared resident allocation.
    dense = benchmark_kernel(polynomial_loss, values, (incidence, dense_coefs), **common)
    indexed = benchmark_kernel(indexed_polynomial_loss, values, (indices, indexed_coefs), **common)
    dense_storage = incidence.numel() * incidence.element_size()
    indexed_storage = indices.numel() * indices.element_size()
    record = {"n": n, "k": k, "density_m_over_n": density,
              "instance_seed": seed, "definition": definition,
              "number_hyperedges": len(problem.edges), "number_terms": len(edges),
              "probabilities_sha256": hashlib.sha256(cpu_values.numpy().tobytes()).hexdigest(),
              "term_order_sha256": hashlib.sha256(json.dumps(edges).encode()).hexdigest(),
              "dense_matrix_shape": list(incidence.shape), "indexed_shape": list(indices.shape),
              "dense_matrix_storage_bytes": dense_storage,
              "indexed_storage_bytes": indexed_storage,
              "coefficient_storage_bytes_per_kernel": dense_coefs.numel() * dense_coefs.element_size(),
              "agreement": agreement, "dense": dense, "indexed": indexed,
              "median_speedup_dense_over_indexed": dense["median_seconds"] / indexed["median_seconds"],
              "matrix_to_index_storage_ratio": dense_storage / indexed_storage}
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "results/loss-kernel-20261018")
    parser.add_argument("--sizes", type=int, nargs="+", default=[200, 500, 1000])
    parser.add_argument("--orders", type=int, nargs="+", default=[3, 5])
    parser.add_argument("--density", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20261018)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--blocks", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if min(args.warmup, args.blocks, args.iterations) < 1:
        raise ValueError("Warmup, blocks and iterations must be positive")
    if not torch.cuda.is_available() or torch.device(args.device).type != "cuda":
        raise RuntimeError("This benchmark requires an available CUDA device")
    torch.set_num_threads(1)
    torch.cuda.init()
    settings = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    sources = ["mis/indexed_loss.py", "mis/neural.py", "mis/problem.py", "scripts/benchmark_loss.py"]
    manifest = {"created_at_utc": datetime.now(timezone.utc).isoformat(),
                "purpose": "Loss-kernel forward/backward microbenchmark; not end-to-end training",
                "settings": settings,
                "git_commit": subprocess.check_output(
                    ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
                "source_sha256": {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in sources},
                "environment": {"python": platform.python_version(), "torch": torch.__version__,
                                "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(args.device),
                                "torch_cpu_threads": torch.get_num_threads()},
                "timing": "5 warmups and 5 blocks x 5 forward+backward iterations by default; synchronized before/after each block; both representations resident; allocation peak subtracts resident baseline",
                "limitations": "Float32 reduction roundoff can differ; no claim of identical training trajectories. Dense measured before indexed in each case; no temperature or power controls.",
                "cases": []}
    manifest_path = args.output / "manifest.json"
    if manifest_path.exists():
        raise ValueError("Output already contains a manifest; use a new directory to retain all runs")
    write_json(manifest_path, manifest)
    records = []
    for n in args.sizes:
        for k in args.orders:
            for definition in ("weak", "strong"):
                name = f"n{n}_k{k}_d{args.density}_{definition}"
                try:
                    record = run_case(n, k, args.density, args.seed, definition, args)
                    record["status"] = "ok"
                    print(f"{name} speedup={record['median_speedup_dense_over_indexed']:.2f} "
                          f"storage={record['matrix_to_index_storage_ratio']:.1f}", flush=True)
                except Exception as error:
                    record = {"n": n, "k": k, "definition": definition,
                              "density_m_over_n": args.density, "status": "error",
                              "error_type": type(error).__name__, "error": str(error)}
                    print(f"{name} ERROR {record['error_type']}: {error}", flush=True)
                write_json(args.output / "cases" / (name + ".json"), record)
                records.append(record)
                manifest["cases"].append(name)
                write_json(manifest_path, manifest)
                gc.collect()
                torch.cuda.empty_cache()
    successful = [r for r in records if r["status"] == "ok"]
    summary = {"number_requested": len(records), "number_successful": len(successful),
               "number_failed": len(records)-len(successful), "records": records}
    write_json(args.output / "summary.json", summary)
    if len(successful) != len(records):
        raise SystemExit("Some cases failed; every case remains saved")


if __name__ == "__main__":
    main()
