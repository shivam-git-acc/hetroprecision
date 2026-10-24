"""Step 1: measure the GPU's real memory bandwidth and record device properties.

Usage: python platform_check.py --out ../results/platform.json
"""
import argparse
import json
import statistics

import torch


def time_op(fn, warmup=10, iters=50):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) / 1e3)  # seconds
    return statistics.median(times), times


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="platform.json")
    ap.add_argument("--gib", type=float, default=4.0, help="buffer size in GiB")
    args = ap.parse_args()

    dev = torch.cuda.get_device_properties(0)
    n = int(args.gib * 2**30) // 2  # number of fp16 elements
    nbytes = n * 2
    x = torch.empty(n, dtype=torch.float16, device="cuda").uniform_()
    y = torch.empty_like(x)

    # copy: reads nbytes and writes nbytes
    t_copy, ts_copy = time_op(lambda: y.copy_(x))
    # read-mostly: a reduction reads nbytes and writes almost nothing
    t_read, ts_read = time_op(lambda: x.sum(dtype=torch.float32))

    q = lambda ts: statistics.quantiles(ts, n=4)
    res = {
        "device": dev.name,
        "compute_capability": f"{dev.major}.{dev.minor}",
        "sm_count": dev.multi_processor_count,
        "total_memory_gb": dev.total_memory / 1e9,
        "l2_cache_mb": getattr(dev, "L2_cache_size", 0) / 2**20,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "buffer_gib": args.gib,
        "copy_bw_GBps_median": 2 * nbytes / t_copy / 1e9,
        "copy_bw_GBps_iqr": [2 * nbytes / t / 1e9 for t in reversed(q(ts_copy)[::2])],
        "read_bw_GBps_median": nbytes / t_read / 1e9,
        "read_bw_GBps_iqr": [nbytes / t / 1e9 for t in reversed(q(ts_read)[::2])],
    }
    print(json.dumps(res, indent=2))
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()
