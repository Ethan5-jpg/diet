"""Read-only CPU/memory snapshot for choosing independent terminal workloads."""

import os
from pathlib import Path


def read(path):
    try:
        return Path(path).read_text().strip()
    except (OSError, ValueError):
        return None


def positive(value):
    try:
        number = int(value)
        return number if 0 < number < 2**60 else None
    except (TypeError, ValueError):
        return None


def snapshot():
    affinity = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
    quota = None
    current = read("/sys/fs/cgroup/cpu.max")
    if current:
        rate, period = current.split()
        if positive(rate) and positive(period):
            quota = int(rate) / int(period)
    else:
        rate, period = (
            read("/sys/fs/cgroup/cpu/cpu.cfs_quota_us"),
            read("/sys/fs/cgroup/cpu/cpu.cfs_period_us"),
        )
        if positive(rate) and positive(period):
            quota = int(rate) / int(period)
    memory = {}
    for line in (read("/proc/meminfo") or "").splitlines():
        fields = line.split()
        if fields[0] in ("MemTotal:", "MemAvailable:"):
            memory[fields[0][:-1]] = int(fields[1]) * 1024
    limit = positive(read("/sys/fs/cgroup/memory.max")) or positive(
        read("/sys/fs/cgroup/memory/memory.limit_in_bytes")
    )
    used = positive(read("/sys/fs/cgroup/memory.current")) or positive(
        read("/sys/fs/cgroup/memory/memory.usage_in_bytes")
    )
    available = memory.get("MemAvailable")
    if limit and used:
        available = min(available, max(limit - used, 0)) if available is not None else max(limit - used, 0)
    return dict(
        affinity_cpus=affinity,
        quota_cpus=quota,
        effective_cpus=min(affinity, quota or affinity),
        available_gib=available / 2**30 if available is not None else None,
    )


if __name__ == "__main__":
    data = snapshot()
    print("RESOURCE_SNAPSHOT read-only; other users/workloads may share these resources.")
    print(
        "CPU affinity=%d quota=%s effective<=%.2f"
        % (
            data["affinity_cpus"],
            "unknown/unlimited" if data["quota_cpus"] is None else "%.2f" % data["quota_cpus"],
            data["effective_cpus"],
        )
    )
    print(
        "AVAILABLE_MEMORY_GiB=%s"
        % ("unknown" if data["available_gib"] is None else "%.2f" % data["available_gib"])
    )
    print("DEFAULT_V13_THREADS=2. Separate V12/V13 runs also duplicate data/model memory.")
    print("If CPU quota is below 4, run sequentially or use --threads 1 for each job.")
