"""Deterministic planning logic for the Student Group Project Coordinator.

Pure functions, no network, no I/O: this is the "second real tool" for Milestone II.
Everything the agent reports about workloads and dates can be checked against these.
"""
from __future__ import annotations

from datetime import date, timedelta


def assign_tasks(tasks: list[dict], members: list[dict]) -> dict:
    """Split tasks across members as evenly as possible.

    tasks:   [{"name": str, "hours": number}]
    members: [{"name": str, "capacity_hours": number}]

    Rule: longest task first; each task goes to the member with the lowest
    load-to-capacity ratio (ties broken alphabetically). Deterministic.
    """
    if not members:
        raise ValueError("At least one team member is required.")
    for t in tasks:
        if t["hours"] <= 0:
            raise ValueError(f"Task '{t['name']}' must have positive hours.")
    for m in members:
        if m["capacity_hours"] <= 0:
            raise ValueError(f"Member '{m['name']}' must have positive capacity.")

    load = {m["name"]: 0.0 for m in members}
    cap = {m["name"]: float(m["capacity_hours"]) for m in members}
    assigned: dict[str, list[str]] = {m["name"]: [] for m in members}

    for t in sorted(tasks, key=lambda t: (-t["hours"], t["name"])):
        pick = min(load, key=lambda n: (load[n] / cap[n], n))
        load[pick] += t["hours"]
        assigned[pick].append(t["name"])

    over = {n: round(load[n] - cap[n], 2) for n in load if load[n] > cap[n]}
    return {
        "assignments": assigned,
        "load_hours": {n: round(v, 2) for n, v in load.items()},
        "spread_hours": round(max(load.values()) - min(load.values()), 2),
        "over_capacity": over,  # member -> hours over; empty means feasible
        "feasible": not over,
    }


def build_schedule(tasks: list[dict], start: str, deadline: str | None = None) -> dict:
    """Compute earliest dates from dependencies.

    tasks: [{"name": str, "days": int, "depends_on": [names]}]
    start/deadline: ISO dates (YYYY-MM-DD).
    A task starts when all its dependencies finish and finishes `days` later
    (finish = start + days). Tasks with no dependency start on `start`.
    """
    by_name = {t["name"]: t for t in tasks}
    for t in tasks:
        for d in t.get("depends_on", []):
            if d not in by_name:
                raise ValueError(f"Task '{t['name']}' depends on unknown task '{d}'.")

    # Kahn's algorithm: also detects cycles.
    indeg = {n: len(t.get("depends_on", [])) for n, t in by_name.items()}
    children: dict[str, list[str]] = {n: [] for n in by_name}
    for n, t in by_name.items():
        for d in t.get("depends_on", []):
            children[d].append(n)
    ready = sorted(n for n, k in indeg.items() if k == 0)
    order: list[str] = []
    while ready:
        n = ready.pop(0)
        order.append(n)
        for c in sorted(children[n]):
            indeg[c] -= 1
            if indeg[c] == 0:
                ready.append(c)
    if len(order) != len(by_name):
        stuck = sorted(set(by_name) - set(order))
        return {"ok": False, "error": "circular_dependency", "tasks_in_cycle": stuck}

    day0 = date.fromisoformat(start)
    s: dict[str, date] = {}
    f: dict[str, date] = {}
    for n in order:
        deps = by_name[n].get("depends_on", [])
        s[n] = max([f[d] for d in deps], default=day0)
        f[n] = s[n] + timedelta(days=int(by_name[n]["days"]))

    end = max(f.values(), default=day0)
    # critical path: walk back from the latest-finishing task
    crit: list[str] = []
    cur = max(f, key=lambda n: (f[n], n)) if f else None
    while cur:
        crit.append(cur)
        deps = by_name[cur].get("depends_on", [])
        cur = max(deps, key=lambda d: (f[d], d)) if deps else None
    crit.reverse()

    result = {
        "ok": True,
        "schedule": {n: {"start": s[n].isoformat(), "finish": f[n].isoformat()} for n in order},
        "project_finish": end.isoformat(),
        "critical_path": crit,
    }
    if deadline:
        dl = date.fromisoformat(deadline)
        slack = (dl - end).days
        result.update(
            deadline=dl.isoformat(),
            meets_deadline=slack >= 0,
            slack_days=slack if slack >= 0 else 0,
            days_late=-slack if slack < 0 else 0,
        )
    return result
