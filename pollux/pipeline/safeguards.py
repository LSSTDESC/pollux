"""Safeguards of the runs, so that tasks that do not work do not use the allocation: no retries that cannot succeed
(:func:`retry_handler`), a canary of each code and configuration before a run (:class:`Canary`), and a guard during
the run (:class:`RunGuard`).

A canary runs one task of each kind on the debug QOS, on a mirror of an existing realization (symbolic links to its
files: the tasks of the canary remake their products, whose atomic writes replace the links without touching the
files of the realization; the other tasks are not run), and records their time and memory.
It is compared to the last accepted canary of the run: a change beyond :data:`TOLERANCE` needs to be accepted
(``--accept-canary``) before a run, which also needs a canary of its code and configuration. During the run, the
guard stops it when a kind of task takes much longer than in its canary, or keeps failing.

Usage, with the run scripts::

    python run_validation.py config.yaml --canary                 # one debug job, then a comparison
    python run_validation.py config.yaml --accept-canary          # accept the last canary of this code
    python run_validation.py config.yaml                          # the run, refused without an accepted canary
    python run_validation.py config.yaml --no-canary              # the run, without canary (guard on failures only)
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import shutil
import sqlite3
import threading
import time

import numpy as np

from pollux.pipeline.report import task_table

# a kind of task differs from the last accepted canary beyond this ratio and difference, in time or memory
TOLERANCE = dict(ratio=1.5, seconds=60., gb=2.)
# the guard stops a run when a kind of task takes this many times its time in the canary (median of 3 tries or more,
# longer by 5 minutes at least), or when its failed tries exceed twice its tasks (plus 5), or 3 of its tasks fail
GUARD = dict(ratio=3., seconds=300., min_tries=3, retries_per_task=2., final_failures=3, interval_s=300.)
# options of a run configuration that do not change its tasks
NOT_IN_KEY = ('seeds', 'seed', 'save_dir_root', 'debug', 'monitoring_db', 'allow_uncommitted', 'canary')
# tasks the canary does not run: plots (written in place, not atomically) and the combinations of the results of
# other tasks, passed as inputs
NOT_IN_CANARY = ('compute_shell_cls', 'compute_kappa_cls', 'compute_catalog_2pt', 'compute_catalog_cls')

# ======================================================================================================================
# region Retries


def retry_handler(drain_margins: dict, walltimes: dict, retries: int, monitoring_db: str | None = None):
    """Return the retry handler of a run (``Config.retry_handler``): the cost of a failed try, out of ``retries``.

    - lost worker (memory): a quarter, so 4 retries;
    - lost manager, by how long the try ran (tasks start no later than the drain margin of their executor before the
      end of their block, half its walltime without drain):

      - less than the drain margin: the block was lost (node failure, job cancelled): 1;
      - more: the block reached its walltime. If the try ran almost a whole block (walltime minus 5 minutes), the task
        does not fit in a block: no retry. Otherwise, it may fit in a block if it starts earlier: one retry;
    - ``OSError``: half (file systems), so 2 retries;
    - other errors: no retry (they would happen again).

    Parameters
    ----------
    drain_margins, walltimes : dict
        Drain margin (None for none) and walltime of the executors, in seconds, by label.
    retries : int
        Retries of the configuration (total cost allowed).
    monitoring_db : str or None, optional
        Monitoring database, for the time at which the try started running (else, at which it was launched).

    Returns
    -------
    callable
    """

    def handler(exception, task_record):
        name = type(exception).__name__
        if name == 'WorkerLost':
            return retries / 4
        if name == 'ManagerLost':
            label = task_record['executor']
            walltime = walltimes.get(label, 3600)
            margin = drain_margins.get(label) or walltime / 2
            ran = _try_seconds(task_record, monitoring_db)
            if ran is None or ran < margin:
                return 1
            task = f"{task_record['func_name']} (task {task_record['id']}) was lost at the end of its block after {ran / 60:.0f} min"
            if ran > walltime - 300:
                print(f"Not retried: {task}, about a whole block of {label}: it does not fit in one.", flush=True)
                return retries + 1
            print(f"Retried once: {task}; it may fit in a block if it starts earlier.", flush=True)
            return retries / 2 + 1  # a second such loss exceeds the retries
        if isinstance(exception, OSError):
            return retries / 2
        return retries + 1

    return handler


def _try_seconds(task_record: dict, monitoring_db: str | None) -> float | None:
    """Return the seconds since the current try of a task started running.

    Parameters
    ----------
    task_record : dict
        Parsl record of the task.
    monitoring_db : str or None
        Monitoring database, for the time the try started running; else, the time it was launched (sent to its
        executor) is used.

    Returns
    -------
    float or None
        None if the try has no start time.
    """
    start = None
    if monitoring_db is not None and os.path.exists(monitoring_db):
        try:
            with sqlite3.connect(f'file:{monitoring_db}?mode=ro', uri=True, timeout=10) as db:
                row = db.execute(
                    'select task_try_time_running from try where run_id = ? and task_id = ? and try_id = ?',
                    (task_record['dfk'].run_id, task_record['id'], task_record['try_id'])).fetchone()
            start = row and row[0] and datetime.datetime.fromisoformat(row[0])
        except sqlite3.Error:
            pass
    start = start or task_record.get('try_time_launched')
    return None if start is None else (datetime.datetime.now() - start).total_seconds()


# endregion

# ======================================================================================================================
# region Canary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the arguments of the run scripts: their configuration, and the options of the canary.

    Parameters
    ----------
    argv : list of str or None, optional
        Arguments, those of the command line by default.

    Returns
    -------
    argparse.Namespace
        ``config``, ``canary``, ``no_canary`` and ``accept_canary``.
    """
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('config', help='YAML configuration of the run')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--canary', action='store_true', help='run the canary of this code and configuration (debug QOS)')
    group.add_argument('--no-canary', action='store_true', help='run without canary (the guard then only checks failures)')
    parser.add_argument('--accept-canary', action='store_true', help='accept the last canary of this code and configuration')
    return parser.parse_args(argv)


class Canary:
    """The canaries of a run configuration: their key (code and configuration), mirror, tasks and records.

    The key of a canary is the git tree of the package (so that changes of the docs or tests keep it), its uncommitted
    changes, and the options of the configuration that change its tasks. Records are in ``{save_dir_root}/canary/{name}.json``, the mirror and the monitoring database of the last canary
    in ``{save_dir_root}/canary/{name}/``. The mirrored realization is the seed ``canary: {seed: ...}`` of the
    configuration (default 0), in ``canary: {save_dir_root: ...}`` (default that of the run); its products must be
    complete.

    Parameters
    ----------
    config : dict
        Configuration of the run (see :func:`~pollux.pipeline.config.read_run_config`).
    config_path : str
        Its file (``name`` is its base name).
    """

    def __init__(self, config: dict, config_path: str):
        from pollux.io.code_version import code_version, _git
        self.config = config
        self.name = os.path.splitext(os.path.basename(config_path))[0]
        self.dir = os.path.join(config['save_dir_root'], 'canary')
        self.records_file = os.path.join(self.dir, f'{self.name}.json')
        # the code of the package (not the docs or tests): its git tree, and its uncommitted changes
        version = code_version()
        tree = (_git('rev-parse', 'HEAD:pollux') or 'nogit').strip()
        diff = _git('diff', 'HEAD', '--', ':/pollux') or ''
        options = {k: v for k, v in config.items() if k not in NOT_IN_KEY}
        self.version = version
        self.config_hash = _hash(options)
        self.key = '-'.join([tree[:12], _hash(diff) if diff else 'clean', self.config_hash])
        self.calls = {}
        self._patched = []

    # Records

    def records(self) -> list[dict]:
        """Return the records of the canaries of this configuration, oldest first.

        Returns
        -------
        list of dict
            ``key``, ``commit``, ``dirty``, ``config_hash``, ``date``, ``tasks`` (per kind of task: ``executor``,
            ``seconds``, ``gb``, ``count`` in a run, ``node_hours``), ``failures``, ``node_hours``, ``changes`` and
            ``accepted``.
        """
        if not os.path.exists(self.records_file):
            return []
        with open(self.records_file) as f:
            return json.load(f)

    def _write_records(self, records: list[dict]):
        """Write the records of the canaries (atomically)."""
        os.makedirs(self.dir, exist_ok=True)
        tmp = f'{self.records_file}.{os.getpid()}.tmp'
        with open(tmp, 'w') as f:
            json.dump(records, f, indent=1)
        os.replace(tmp, self.records_file)

    def last_accepted(self, other_than: str | None = None) -> dict | None:
        """Return the record of the last accepted canary.

        Parameters
        ----------
        other_than : str or None, optional
            Key to leave out (that of the canary compared).

        Returns
        -------
        dict or None
            None if no canary is accepted.
        """
        accepted = [r for r in self.records() if r['accepted'] and r['key'] != other_than]
        return accepted[-1] if accepted else None

    def check(self) -> dict:
        """Return the tasks of the accepted canary of this code and configuration, for the guard of the run.

        Returns
        -------
        dict
            Per kind of task, as in :meth:`records`.

        Raises
        ------
        SystemExit
            Without canary of this key, or if it is not accepted.
        """
        mine = [r for r in self.records() if r['key'] == self.key]
        if not mine:
            raise SystemExit(f"No canary of this code and configuration (key {self.key}): run with --canary first "
                             "(one debug job), or with --no-canary.")
        if not mine[-1]['accepted']:
            raise SystemExit(
                f"The canary of this code and configuration (key {self.key}, {mine[-1]['date']}) is not accepted: "
                f"review its comparison ({self.records_file}), then run with --accept-canary.")
        return mine[-1]['tasks']

    def accept(self):
        """Accept the last canary of this code and configuration.

        Raises
        ------
        SystemExit
            Without canary of this key.
        """
        records = self.records()
        mine = [r for r in records if r['key'] == self.key]
        if not mine:
            raise SystemExit(f"No canary of this code and configuration (key {self.key}) to accept: run with --canary first.")
        mine[-1]['accepted'] = True
        self._write_records(records)
        print(f"Accepted the canary of {mine[-1]['date']} (key {self.key}).")

    # Running

    def configure(self) -> dict:
        """Return the configuration of the canary, and make its mirror of the realization of reference.

        The mirror (its directories, with links to its files) replaces that of the previous canary. Workers see
        ``POLLUX_CANARY``, so that their tasks remake their products (see :func:`pollux.pipeline.tasks._run`).

        Returns
        -------
        dict
            The configuration of the run on the mirror, with the seed of reference, on the debug QOS, with its own
            monitoring database.

        Raises
        ------
        SystemExit
            Without realization of reference.
        """
        options = self.config.get('canary', {}) or {}
        seed = int(options.get('seed', 0))
        source_root = os.path.expandvars(options.get('save_dir_root', self.config['save_dir_root']))
        nside_dir = os.path.join(self.config['sim_relative_path'], f"nside{int(self.config['nside'])}")
        source = os.path.join(source_root, nside_dir)
        root = os.path.join(self.dir, self.name)
        if not os.path.isdir(os.path.join(source, f'seed{seed}')):
            raise SystemExit(f"No realization of reference for the canary in {source}/seed{seed}: set canary: {{seed, save_dir_root}}.")
        # the mirror is made by the run script only: the processes of Parsl that import it again inherit POLLUX_CANARY
        if not os.environ.get('POLLUX_CANARY'):
            if os.path.exists(root):
                shutil.rmtree(root)  # links and the products of the previous canary only
            for sub in ('footprint', f'seed{seed}'):
                if os.path.isdir(os.path.join(source, sub)):
                    _mirror(os.path.join(source, sub), os.path.join(root, nside_dir, sub))
        os.environ['POLLUX_CANARY'] = '1'
        config = {
            **self.config, 'seeds': [seed],
            'save_dir_root': root,
            'debug': True,
            'monitoring_db': os.path.join(root, 'monitoring.db'),
            'allow_uncommitted': True
        }
        if 'seed' in config:  # scripts of one realization
            config['seed'] = seed
        return config

    def patch(self, *namespaces: dict):
        """Replace the apps of the namespaces, and of :mod:`pollux.pipeline.tasks`, by recorders of their calls that
        return done futures: the run script then creates its tasks without running them (see :meth:`submit`).

        Parameters
        ----------
        *namespaces : dict
            Namespaces of the run script (its ``globals()``).
        """
        from concurrent.futures import Future
        from parsl.app.python import PythonApp
        import pollux.pipeline.tasks as tasks

        def recorder(app):

            def record(*args, **kwargs):
                self.calls.setdefault(app.func.__name__, (app, []))[1].append((args, kwargs))
                future = Future()
                future.set_result(None)
                return future

            return record

        for namespace in (*namespaces, vars(tasks)):
            for name, app in list(namespace.items()):
                if isinstance(app, PythonApp) and not name.startswith('_'):
                    self._patched.append((namespace, name, app))
                    namespace[name] = recorder(app)

    def submit(self) -> list:
        """Restore the apps, and submit the canary: the middle call of each kind of task (in the order of the run
        script; among those on the dark-matter-only shells, if any), without its inputs (on disk), except the plots and
        the combinations of results (``NOT_IN_CANARY``).

        Returns
        -------
        list
            Futures of the tasks of the canary.
        """
        for namespace, name, app in self._patched:
            namespace[name] = app
        self.counts = {name: len(calls) for name, (_, calls) in self.calls.items()}
        futures = []
        for name, (app, calls) in self.calls.items():
            if name.startswith('plot_') or name in NOT_IN_CANARY:
                continue
            calls = [c for c in calls if c[1].get('baryon_dict') is None] or calls
            args, kwargs = calls[len(calls) // 2]
            futures.append(app(*args, **{**kwargs, 'inputs': []}))
        print(f"Canary: {len(futures)} tasks, one of each kind:", ', '.join(sorted(f.task_record['func_name'] for f in futures)), flush=True)
        return futures

    def record(self, monitoring_db: str, run_id: str, failures: list[str], accept: bool = False) -> dict:
        """Record the canary of this run, compare it to the last accepted one, and print the comparison.

        It is accepted with ``accept``, or if it matches the last accepted canary (:data:`TOLERANCE`) without failure.

        Parameters
        ----------
        monitoring_db : str
            Monitoring database of the canary.
        run_id : str
            Its run identifier.
        failures : list of str
            Its failed tasks.
        accept : bool, optional
            Accept it whatever the comparison.

        Returns
        -------
        dict
            The record.
        """
        from pollux.pipeline.config import EXECUTORS
        workers = {label: e['workers'] for label, e in EXECUTORS.items()}  # tasks per node, for the node-hours
        with sqlite3.connect(f'file:{monitoring_db}?mode=ro', uri=True) as db:
            rows = task_table(db, [run_id], min_seconds=0.)
        tasks = {}
        for row in rows:
            tasks[row['task']] = dict(executor=row['executor'], seconds=float(row['median_s']), gb=float(row['max_gb']), count=self.counts.get(row['task'], 1))
        for t in tasks.values():
            t['node_hours'] = t['seconds'] * t['count'] / workers.get(t['executor'], 1) / 3600.
        reference = self.last_accepted(other_than=self.key)
        changes = _compare(tasks, reference['tasks'] if reference else {})
        record = dict(
            key=self.key,
            commit=self.version['commit'],
            dirty=self.version['dirty'],
            config_hash=self.config_hash,
            date=datetime.datetime.now().isoformat(timespec='seconds'),
            tasks=tasks,
            failures=failures,
            node_hours=sum(t['node_hours'] for t in tasks.values()),
            changes=changes,
            accepted=bool(accept or (reference is not None and not changes and not failures)))
        records = self.records() + [record]
        self._write_records(records)
        _print_comparison(record, reference)
        return record


def _mirror(source: str, target: str):
    """Make the directories of ``source`` in ``target``, with links to its files (but not to temporary files)."""
    for folder, _, files in os.walk(source):
        out = os.path.join(target, os.path.relpath(folder, source))
        os.makedirs(out, exist_ok=True)
        for fname in files:
            if not fname.endswith('.tmp') and '.tmp.' not in fname:
                os.symlink(os.path.join(folder, fname), os.path.join(out, fname))


def _compare(tasks: dict, reference: dict) -> dict:
    """Return the changes of the tasks of a canary from those of the reference one beyond :data:`TOLERANCE`.

    Parameters
    ----------
    tasks, reference : dict
        Tasks of the canaries, as in :meth:`Canary.records`.

    Returns
    -------
    dict
        Description of the change, by kind of task (``'new'`` for kinds absent from the reference).
    """
    changes = {}
    for name, t in tasks.items():
        r = reference.get(name)
        if r is None:
            changes[name] = 'new'
            continue
        what = []
        if t['seconds'] > TOLERANCE['ratio'] * r['seconds'] and t['seconds'] - r['seconds'] > TOLERANCE['seconds']:
            what.append(f"time x{t['seconds'] / r['seconds']:.1f}")
        if np.isfinite(t['gb']) and np.isfinite(r['gb']) and t['gb'] > TOLERANCE['ratio'] * r['gb'] and t['gb'] - r['gb'] > TOLERANCE['gb']:
            what.append(f"memory x{t['gb'] / r['gb']:.1f}")
        if what:
            changes[name] = ', '.join(what)
    return changes


def _print_comparison(record: dict, reference: dict | None):
    """Print the tasks of a canary next to those of the reference one (None for none), and whether it is accepted."""
    ref = reference['tasks'] if reference else {}
    print(f"\nCanary {record['key']} ({record['date']})" + (f", compared to {reference['key']} ({reference['date']})" if reference else ", first canary"))
    print(f"{'task':34s} {'executor':>8s} {'s':>7s} {'ref s':>7s} {'GB':>6s} {'ref GB':>6s} {'node-h':>7s}  change")
    for name, t in sorted(record['tasks'].items(), key=lambda kv: -kv[1]['node_hours']):
        r = ref.get(name, {})
        print(
            f"{name:34s} {t['executor']:>8s} {t['seconds']:7.0f} {r.get('seconds', np.nan):7.0f} {t['gb']:6.1f} {r.get('gb', np.nan):6.1f} "
            f"{t['node_hours']:7.1f}  {record['changes'].get(name, '')}")
    print(f"Projected node-hours of a realization: {record['node_hours']:.0f}" + (f" (reference: {reference['node_hours']:.0f})" if reference else ''))
    for failure in record['failures']:
        print('Failed', failure)
    if record['accepted']:
        print("Accepted.")
    else:
        print(
            "Not accepted: " + ('failed tasks. ' if record['failures'] else '') + ('first canary. ' if reference is None else '') +
            (f"changes beyond the tolerance ({TOLERANCE}). " if record['changes'] else '') + "Run with --accept-canary to accept it, if intended.")


# endregion

# ======================================================================================================================
# region Guard


class RunGuard(threading.Thread):
    """Thread stopping a run (its blocks, then its process) when a kind of task takes much longer than in its canary,
    or keeps failing (see :data:`GUARD`).

    Parameters
    ----------
    monitoring_db : str
        Monitoring database of the run.
    reference : dict or None
        Tasks of the canary of the run (:meth:`Canary.check`), None to check the failures only.
    retries : int, optional
        Retries of the configuration: a task whose failed tries cost more has failed.
    """

    def __init__(self, monitoring_db: str, reference: dict | None, retries: int = 8):
        super().__init__(daemon=True, name='pollux-guard')
        self.monitoring_db = monitoring_db
        self.reference = reference or {}
        self.retries = retries

    def run(self):
        """Check the run every ``GUARD['interval_s']`` seconds, and stop it if needed."""
        import parsl
        run_id = parsl.dfk().run_id
        while True:
            time.sleep(GUARD['interval_s'])
            try:
                reason = self.check(run_id)
            except sqlite3.Error:
                continue
            if reason:
                self.stop(reason)

    def check(self, run_id: str) -> str | None:
        """Return why a run should stop.

        Parameters
        ----------
        run_id : str
            Identifier of the run in the monitoring database.

        Returns
        -------
        str or None
            None if it should not stop.
        """
        with sqlite3.connect(f'file:{self.monitoring_db}?mode=ro', uri=True, timeout=30) as db:
            rows = task_table(db, [run_id], min_seconds=5.)  # not the tasks skipped (products saved)
            counts = db.execute(
                """select k.task_func_name, count(distinct t.task_id), sum(t.task_fail_history is not null and t.task_fail_history != '')
                                   from try t join task k on t.run_id = k.run_id and t.task_id = k.task_id where t.run_id = ? group by k.task_func_name""",
                (run_id,)).fetchall()
            final = dict(
                db.execute(
                    """select task_func_name, count(*) from task where run_id = ? and task_fail_cost > ? group by task_func_name""",
                    (run_id, self.retries)).fetchall())
        for row in rows:
            r = self.reference.get(row['task'])
            if r and row['n'] >= GUARD['min_tries'] and row['median_s'] > GUARD['ratio'] * r['seconds'] and row['median_s'] - r['seconds'] > GUARD['seconds']:
                return (
                    f"{row['task']} takes {row['median_s'] / 60:.0f} min (median of {row['n']} tries), "
                    f"{row['median_s'] / r['seconds']:.1f} times its {r['seconds'] / 60:.0f} min in the canary.")
        for name, ntasks, failed in counts:
            if failed and failed > GUARD['retries_per_task'] * ntasks + 5:
                return f"{name}: {failed} failed tries for {ntasks} tasks."
        for name, n in final.items():
            if n >= GUARD['final_failures']:
                return f"{name}: {n} tasks failed after their retries."
        return None

    @staticmethod
    def stop(reason: str):
        """Cancel the blocks of the run, and exit (status 3).

        Parameters
        ----------
        reason : str
            Printed.
        """
        import parsl
        print(f"\nRun stopped by its guard: {reason}", flush=True)
        dfk = parsl.dfk()
        for executor in dfk.executors.values():
            jobs = list(getattr(executor, 'blocks_to_job_id', {}).values())
            if jobs and getattr(executor, 'provider', None) is not None:
                try:
                    executor.provider.cancel(jobs)
                except Exception as e:  # the exit below matters more
                    print(f"Could not cancel the jobs of {executor.label}: {e}", flush=True)
        os._exit(3)


def run_tasks(futures: list, monitoring_db: str, reference: dict | None = None, guard: bool = True) -> list[str]:
    """Wait for the tasks of a run, with its guard: a failed task does not stop the run (the tasks depending on it fail
    too, the others go on), the guard does (see :class:`RunGuard`).

    Parameters
    ----------
    futures : list
        Futures of the tasks.
    monitoring_db : str
        Monitoring database of the run.
    reference : dict or None, optional
        Tasks of the canary of the run, for the guard (None: failures only).
    guard : bool, optional
        Start the guard. Default is True.

    Returns
    -------
    list of str
        The failed tasks.
    """
    from tqdm.auto import tqdm
    if guard:
        import parsl
        RunGuard(monitoring_db, reference, retries=parsl.dfk().config.retries).start()
    failures = []
    for future in tqdm(futures):
        try:
            future.result()
        except Exception as e:
            failures.append(f"task {future.tid} ({future.task_record['func_name']}): {type(e).__name__}: {e}")
            print(f"Failed {failures[-1]}", flush=True)
    return failures


# endregion


def _hash(obj) -> str:
    """Return an identifier (8 hexadecimal characters) of a JSON-serializable object."""
    return hashlib.sha1(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:8]
