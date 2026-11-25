"""Parsl configuration of the pipeline: executors by memory and CPU use on Slurm at NERSC, and the YAML configurations of runs."""

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
from parsl.providers import SlurmProvider

from pollux.pipeline.safeguards import retry_handler

nodes_per_block = 1
RETRIES = 8  # cost of the failed tries of a task before it fails (see pollux.pipeline.safeguards.retry_handler)

# Per-executor resources (tasks pick one via @python_app(executors=[...])).
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


def get_config(run_dir, debug, workflow_name, monitoring_db):
    """Return the parsl configuration of a run.

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

    Notes
    -----
    The NERSC account is ``$POLLUX_ACCOUNT`` (default ``m1727``), and Slurm emails go to
    ``$POLLUX_MAIL`` if set. Workers inherit the environment of the run script.
    """
    qos = 'debug' if debug else 'regular'
    account = os.environ.get('POLLUX_ACCOUNT', 'm1727')
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
