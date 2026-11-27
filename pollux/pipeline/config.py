"""Parsl configurations of the pipeline, and the YAML configurations of runs.

The tasks of :mod:`pollux.pipeline.tasks` run on executors named by the memory and CPU use of their tasks,
:data:`EXECUTOR_LABELS`: any Parsl configuration of a run must define executors with these labels.

- :func:`perlmutter_config`: Slurm jobs on the CPU nodes of NERSC's Perlmutter (512 GB, 256 threads per node), sized
  for full-sky maps at nside 8192 (:data:`EXECUTORS`). Site-specific: the account, queues, node size and Slurm options
  are those of NERSC.
- :func:`local_config`: all executors on the local machine (a laptop, a login or interactive node), for tests and
  small runs.

Elsewhere, write a :class:`parsl.config.Config` with executors of these labels and the provider of the site's
scheduler (see the `Parsl documentation <https://parsl.readthedocs.io/en/stable/userguide/configuring.html>`_), sized
from :data:`EXECUTORS`, and pass it to :func:`check_labels`.
"""

import os
import time
from functools import partial

import yaml

from parsl.addresses import address_by_hostname
from parsl.config import Config
from parsl.executors import HighThroughputExecutor
from parsl.jobs.error_handlers import windowed_error_handler
from parsl.jobs.states import JobState, JobStatus
from parsl.launchers import SrunLauncher
from parsl.monitoring.monitoring import MonitoringHub
from parsl.providers import LocalProvider, SlurmProvider

from pollux.pipeline.safeguards import retry_handler

#: Executors of the tasks (``@python_app(executors=[...])`` in :mod:`pollux.pipeline.tasks`), by the memory and CPU use of
#: their tasks: every Parsl configuration of a run must define them.
EXECUTOR_LABELS = ('default', 'highmem', 'heavy', 'twopt')

nodes_per_block = 1
RETRIES = 8  # cost of the failed tries of a task before it fails (see pollux.pipeline.safeguards.retry_handler)

# Per-executor resources on Perlmutter CPU nodes (512 GB, 128 cores, 256 threads), for perlmutter_config.
# Executors differ by the number of tasks sharing a node, chosen so their memory fits
# in the node's 512 GB. Threads are capped so that the tasks of a node share its 256 threads, except for 'heavy',
# whose single task uses them all. Sized for nside 8192
# full sky / 2048^3 particles:
#
# - 'default': light tasks: sub-volume shells, full-shell assembly, kappa sums.
# - 'highmem': up to ~120 GB: Cls, halo catalogs, baryonification, catalog shells.
# - 'heavy': full-sky healpy transforms, which already use all cores, so one at a
#   time: alms, shear and tidal maps, NaMaster workspaces.
# - 'twopt': catalog 2pt functions, two groups of bin pairs per node (up to ~190 GB each with 25% of the galaxies,
#   ~90 GB with 10%), each with half of the threads.
#
# workers:      tasks per node
# env:          extra environment variables for the workers (optional)
# max_blocks:   max simultaneous 1-node Slurm jobs
# walltime:     Slurm walltime
# drain_margin: stop sending new tasks to a block this long before its walltime ends,
#               so in-flight tasks finish instead of being killed (ManagerLost);
#               must exceed the executor's longest task. None: no draining (tasks
#               still running at the walltime are killed and retried), for short tasks.
#: Resources of each executor on Perlmutter CPU nodes (see the comments above), used by :func:`perlmutter_config`.
EXECUTORS = {
    # 16 threads per task, so that the 32 tasks share the 256 threads of a node
    'default':
        dict(
            workers=32,
            max_blocks=16,
            walltime='1:00:00',
            drain_margin=10 * 60,
            env={
                'OMP_WAIT_POLICY': 'PASSIVE',
                'OMP_NUM_THREADS': 16,
                'NUMBA_NUM_THREADS': 16
            }),  # idle OpenMP threads sleep instead of spinning
    # 64 threads per task: uncapped, the thread pools of four workers exceed what a node can create
    'highmem':
        dict(
            workers=4,
            max_blocks=4,
            walltime='1:00:00',
            drain_margin=20 * 60,
            env={
                'OMP_WAIT_POLICY': 'PASSIVE',
                'OMP_NUM_THREADS': 64,
                'NUMBA_NUM_THREADS': 64
            }),
    'heavy':
        dict(workers=1, max_blocks=8, walltime='1:00:00', drain_margin=15 * 60),
    'twopt':
        dict(
            workers=2,
            max_blocks=8,
            walltime='2:00:00',
            drain_margin=50 * 60,
            env={
                'OMP_WAIT_POLICY': 'PASSIVE',
                'OMP_NUM_THREADS': 128,
                'NUMBA_NUM_THREADS': 128
            }),
}

# Debug QOS: 30 min walltime, at most 5 submitted and 2 running jobs per user, so the
# debug max_blocks sum to 5 (extra jobs wait in the queue). No draining in debug:
# blocks run until the walltime and tasks still running then are retried.
DEBUG = {
    'default': dict(max_blocks=2, drain_margin=None),
    'highmem': dict(max_blocks=1, drain_margin=None),
    'heavy': dict(max_blocks=1, drain_margin=None),
    'twopt': dict(max_blocks=1, drain_margin=None),
}


class GuardedHighThroughputExecutor(HighThroughputExecutor):
    """HighThroughputExecutor whose scale-in by zero blocks does nothing.

    With ``max_idletime``, HighThroughputExecutor.scale_in only stops selecting blocks
    once it has ``blocks`` of them, which never happens for ``blocks=0``: it then
    cancels every block without tasks, including pending ones (idle forever). The
    htex_auto_scale strategy requests this whenever there are more slots than tasks
    (e.g. one task left on a multi-worker executor), so it would keep cancelling the
    blocks submitted for that task.
    """

    def scale_in(self, blocks, max_idletime=None):
        """Scale in by ``blocks`` blocks, as :class:`HighThroughputExecutor`, but do nothing for zero blocks.

        Parameters
        ----------
        blocks : int
            Number of blocks to stop.
        max_idletime : float or None, optional
            Minimum idle time of the blocks to stop.

        Returns
        -------
        list
            Identifiers of the blocks stopped.
        """
        if blocks == 0:
            return []
        return super().scale_in(blocks, max_idletime=max_idletime)


class GracePeriodSlurmProvider(SlurmProvider):
    """SlurmProvider that tolerates sacct's accounting-database lag.

    NERSC's slurmdbd can take over a minute to ingest a just-submitted job,
    during which `sacct` returns zero rows for it. The base SlurmProvider
    treats any job missing from sacct output as COMPLETED (see its _status),
    which HighThroughputExecutor then rewrites to MISSING and crashes the
    whole run via TooManyJobFailuresError — even though the job is still
    starting up fine. This subclass ignores that COMPLETED verdict for jobs
    submitted less than `grace_period_s` seconds ago.
    """

    grace_period_s = 180

    def submit(self, *args, **kwargs):
        """Submit a job, as :class:`SlurmProvider`, recording its submission time.

        Returns
        -------
        str
            Identifier of the job.
        """
        job_id = super().submit(*args, **kwargs)
        self.resources[job_id]['submit_time'] = time.time()
        return job_id

    def _status(self):
        """Update the status of the jobs, as :class:`SlurmProvider`, but keep those missing from ``sacct`` pending during the grace period."""
        active_jobs = {jid: job for jid, job in self.resources.items() if not job['status'].terminal}
        super()._status()
        now = time.time()
        for jid, job in active_jobs.items():
            submit_time = job.get('submit_time')
            if job['status'].state == JobState.COMPLETED and submit_time is not None and (now - submit_time) < self.grace_period_s:
                job['status'] = JobStatus(JobState.PENDING, stdout_path=job['status'].stdout_path, stderr_path=job['status'].stderr_path)


def read_run_config(path):
    """Read the YAML configuration of a run, expanding environment variables (e.g. ``$PSCRATCH``) in its top-level paths.

    Parameters
    ----------
    path : str
        YAML file of the run.

    Returns
    -------
    dict

    Raises
    ------
    ValueError
        If a top-level path uses an undefined environment variable.
    """
    with open(path) as f:
        config = yaml.safe_load(f)
    for key, value in config.items():
        if isinstance(value, str):
            config[key] = os.path.expandvars(value)
            if '$' in config[key]:
                raise ValueError(f"Undefined environment variable in {key}: {value}")
    return config


def perlmutter_config(run_dir, debug, workflow_name, monitoring_db):
    """Return the Parsl configuration of a run on the CPU nodes of NERSC's Perlmutter.

    Parameters
    ----------
    run_dir : str
        Parsl run directory.
    debug : bool
        Use the debug QOS (see ``DEBUG``) instead of the regular one.
    workflow_name : str
        Name of the workflow in the monitoring database.
    monitoring_db : str
        Path of the monitoring database, in the user's output directory.

    Returns
    -------
    parsl.config.Config

    Raises
    ------
    ValueError
        If ``$POLLUX_ACCOUNT`` is not set.

    Notes
    -----
    Site-specific: one-node Slurm jobs on Perlmutter's CPU nodes (``-C cpu``, QOS ``regular`` or ``debug``), sized by
    :data:`EXECUTORS` (or :data:`DEBUG`). The NERSC account is ``$POLLUX_ACCOUNT`` (required: no default, so that
    no project is charged by mistake), and Slurm emails go to ``$POLLUX_MAIL`` if set. Workers inherit the
    environment of the run script.
    """
    qos = 'debug' if debug else 'regular'
    account = os.environ.get('POLLUX_ACCOUNT')
    if not account:
        raise ValueError("Set the NERSC account of the run in $POLLUX_ACCOUNT (e.g. export POLLUX_ACCOUNT=m1727)")
    mail = os.environ.get('POLLUX_MAIL')
    mail_options = f' \n#SBATCH --mail-user={mail} \n#SBATCH --mail-type=ALL' if mail else ''

    executors, drain_margins, walltimes = [], {}, {}
    for label, e in EXECUTORS.items():
        if debug:
            e = {**e, **DEBUG[label], 'walltime': '00:30:00'}
        walltime_s = _seconds(e['walltime'])
        drain_period = None if e['drain_margin'] is None else walltime_s - e['drain_margin']
        drain_margins[label], walltimes[label] = e['drain_margin'], walltime_s
        provider = GracePeriodSlurmProvider(
            account=account,
            nodes_per_block=nodes_per_block,
            init_blocks=0,
            min_blocks=0,
            max_blocks=e['max_blocks'],
            parallelism=1.0,  # request blocks for all queued tasks, up to max_blocks
            scheduler_options=f'#SBATCH -C cpu \n#SBATCH --qos={qos}{mail_options}',
            walltime=e['walltime'],
            cmd_timeout=120,
            launcher=SrunLauncher(overrides='-c 256'),
            worker_init=''.join(f'export {k}={v}\n' for k, v in e.get('env', {}).items()),
        )
        executors.append(
            GuardedHighThroughputExecutor(
                label=label,
                max_workers_per_node=e['workers'],
                provider=provider,
                heartbeat_threshold=600,  # to avoid network interruption (default to 120)
                # only give up on an executor after 5 consecutive failed blocks: the default handler gives up
                # after the first one (with init_blocks=0), e.g. a submission refused by the debug QOS job limit
                block_error_handler=partial(windowed_error_handler, threshold=5),
                drain_period=drain_period,
            ))

    config = Config(
        executors=executors,
        run_dir=run_dir,
        monitoring=MonitoringHub(
            workflow_name=workflow_name,
            hub_address=address_by_hostname(),
            monitoring_debug=False,
            resource_monitoring_interval=60,  # 600,
            logging_endpoint=f'sqlite:///{monitoring_db}',
        ),
        retries=RETRIES,
        retry_handler=retry_handler(drain_margins, walltimes, RETRIES, monitoring_db),  # no retries that cannot succeed
        strategy='htex_auto_scale',  # also scale in idle blocks while other blocks still have tasks
        max_idletime=60,
    )
    return check_labels(config)


get_config = perlmutter_config  # earlier name, kept for the run scripts


def local_config(run_dir, workers=None, workflow_name=None, monitoring_db=None):
    """Return a Parsl configuration running all the executors on the local machine.

    For tests and small runs (e.g. low nside or a small footprint) on a laptop, a login node or an interactive node:
    no scheduler, no retries.

    Parameters
    ----------
    run_dir : str
        Parsl run directory.
    workers : dict or int or None, optional
        Workers of each executor (by label), or of all of them. Default is one each: memory, not cores, usually
        limits how many tasks run at once (see :data:`EXECUTORS` for the memory of tasks at nside 8192).
    workflow_name : str or None, optional
        Name of the workflow in the monitoring database (with ``monitoring_db``).
    monitoring_db : str or None, optional
        Path of a monitoring database, to record the tasks (as on Perlmutter, for
        ``python -m pollux.pipeline.report`` and the guard of :func:`~pollux.pipeline.safeguards.run_tasks`). Default
        is None: no monitoring (run the tasks with ``run_tasks(futures, None, guard=False)``).

    Returns
    -------
    parsl.config.Config
    """
    if workers is None or isinstance(workers, int):
        workers = dict.fromkeys(EXECUTOR_LABELS, workers or 1)
    executors = [
        HighThroughputExecutor(label=label, max_workers_per_node=workers[label], provider=LocalProvider(init_blocks=1, max_blocks=1))
        for label in EXECUTOR_LABELS
    ]
    monitoring = None
    if monitoring_db is not None:
        monitoring = MonitoringHub(
            workflow_name=workflow_name, hub_address='127.0.0.1', resource_monitoring_interval=60, logging_endpoint=f'sqlite:///{monitoring_db}')
    return check_labels(Config(executors=executors, run_dir=run_dir, monitoring=monitoring, retries=0))


def check_labels(config):
    """Check that a Parsl configuration has the executors of the tasks (:data:`EXECUTOR_LABELS`).

    Parameters
    ----------
    config : parsl.config.Config

    Returns
    -------
    parsl.config.Config
        ``config`` itself.

    Raises
    ------
    ValueError
        If an executor of the tasks is missing.
    """
    missing = set(EXECUTOR_LABELS) - {e.label for e in config.executors}
    if missing:
        raise ValueError(f"The Parsl configuration has no executor {sorted(missing)}: the tasks need {list(EXECUTOR_LABELS)}")
    return config


def _seconds(walltime):
    """Return the seconds of a Slurm walltime ``'HH:MM:SS'``.

    Parameters
    ----------
    walltime : str

    Returns
    -------
    int
    """
    h, m, sec = (int(x) for x in walltime.split(':'))
    return 3600 * h + 60 * m + sec
