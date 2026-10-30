"""Resource use of a run, from its parsl monitoring database.

Usage: ``python -m pollux.pipeline.report {save_dir_root}/monitoring.db [--run name] [--min-seconds 5]``

Memory and CPU are sampled every minute: they are unreliable for shorter tasks, and CPU time
counts all the threads of a task.
"""
from __future__ import annotations

import argparse
import collections
import datetime
import sqlite3

import numpy as np


def _timestamp(value: str | None) -> float | None:
    """Return the time in seconds of a database date, if any."""
    return datetime.datetime.fromisoformat(value).timestamp() if value else None


def task_table(db: sqlite3.Connection, runs: list[str], min_seconds: float = 5.) -> list[dict]:
    """Return the time, memory and CPU use of each kind of task.

    Parameters
    ----------
    db : sqlite3.Connection
        Monitoring database.
    runs : list of str
        Run identifiers.
    min_seconds : float, optional
        Tries shorter than this are ignored (e.g. outputs already on disk).

    Returns
    -------
    list of dict
        Per task name and executor, sorted by total time: number of tries ``n``, median and
        maximum time in seconds, median and maximum memory in GB, mean number of cores
        used (CPU time over wall time), and total time ``hours``.
    """
    marks = ','.join('?' * len(runs))
    samples = {}
    for run, task, attempt, memory, user, system in db.execute(
            f"""select run_id, task_id, try_id, max(psutil_process_memory_resident), max(psutil_process_time_user), max(psutil_process_time_system)
            from resource where run_id in ({marks}) group by run_id, task_id, try_id""", runs):
        samples[run, task, attempt] = ((memory or 0.) / 1e9, (user or 0.) + (system or 0.))

    tries = collections.defaultdict(list)
    for run, task, attempt, name, executor, start, end in db.execute(
            f"""select t.run_id, t.task_id, t.try_id, k.task_func_name, t.task_executor, t.task_try_time_running, t.task_try_time_returned
            from try t join task k on t.run_id = k.run_id and t.task_id = k.task_id where t.run_id in ({marks})""", runs):
        if start and end and _timestamp(end) - _timestamp(start) >= min_seconds:
            tries[name, executor].append((_timestamp(end) - _timestamp(start), *samples.get((run, task, attempt), (np.nan, np.nan))))

    table = []
    for (name, executor), values in tries.items():
        seconds, memory, cpu = np.array(values).T
        sampled = np.isfinite(memory)
        table.append(
            dict(
                task=name,
                executor=executor,
                n=len(seconds),
                median_s=np.median(seconds),
                max_s=seconds.max(),
                median_gb=np.median(memory[sampled]) if sampled.any() else np.nan,
                max_gb=memory[sampled].max() if sampled.any() else np.nan,
                cores=cpu[sampled].sum() / seconds[sampled].sum() if sampled.any() else np.nan,
                hours=seconds.sum() / 3600.))
    return sorted(table, key=lambda row: -row['hours'])


def failures(db: sqlite3.Connection, runs: list[str]) -> collections.Counter:
    """Return the number of failed tries per task name and kind of error.

    Parameters
    ----------
    db : sqlite3.Connection
        Monitoring database.
    runs : list of str
        Run identifiers.

    Returns
    -------
    collections.Counter
        Keyed by ``(task name, error)``.
    """
    marks = ','.join('?' * len(runs))
    counts = collections.Counter()
    for name, history in db.execute(f"""select k.task_func_name, t.task_fail_history from try t join task k on t.run_id = k.run_id and t.task_id = k.task_id
            where t.run_id in ({marks}) and t.task_fail_history != ''""", runs):
        for error in history.split(','):
            error = error.strip().split('(')[0]
            if error.isidentifier():
                counts[name, error] += 1
    return counts


def main() -> None:
    """Print the report of a run."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('database', help='monitoring database of the run')
    parser.add_argument('--run', default='%', help='workflow name (SQL pattern); default: all runs in the database')
    parser.add_argument('--min-seconds', type=float, default=5., help='ignore shorter tries (default: 5)')
    args = parser.parse_args()

    db = sqlite3.connect(f'file:{args.database}?mode=ro', uri=True, timeout=30)
    runs = [row[0] for row in db.execute('select run_id from workflow where workflow_name like ?', (args.run,))]

    print(f"{'task':34s}{'executor':>9s}{'tries':>7s}{'med s':>8s}{'max s':>8s}{'med GB':>8s}{'max GB':>8s}{'cores':>7s}{'hours':>8s}")
    for row in task_table(db, runs, args.min_seconds):
        print(
            f"{row['task']:34s}{row['executor']:>9s}{row['n']:7d}{row['median_s']:8.0f}{row['max_s']:8.0f}"
            f"{row['median_gb']:8.1f}{row['max_gb']:8.1f}{row['cores']:7.1f}{row['hours']:8.1f}")

    errors = failures(db, runs)
    if errors:
        print('\nFailed tries:')
        for (name, error), count in errors.most_common():
            print(f"{count:7d}  {name}: {error}")


if __name__ == '__main__':
    main()
