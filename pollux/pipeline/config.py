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

nodes_per_block = 1

# Per-executor resources (tasks pick one via @python_app(executors=[...])).
# Executors differ by the number of tasks sharing a node, chosen so their memory fits
# in the node's 512 GB; threads are not capped, numba / OpenMP use all cores and
# oversubscription fills the cores left idle by serial steps. Sized for nside 8192
# full sky / 2048^3 particles (run_12 measurements, and run_11 at nside 4096 scaled
# up for map-level tasks):
#
# - 'default': tasks up to ~10 GB that take minutes: sub-volume shells (<10 GB, up
#   to ~5 min), full-shell assembly (~5 GB), kappa sums (~8 GB).
# - 'highmem': up to ~120 GB: Cls (from ~50 GB alms), halo catalogs (116 GB at
#   nside 4096 already) and baryonification.
# - 'heavy': full-sky healpy transforms (~50-100 GB), which already use all cores, so
#   one at a time (one nside 8192 map2alm takes ~1.7 min on a full node): alms
#   (~3 min), shear (~5 min) and tidal (~8 min) maps, NaMaster workspace (~12 min);
#   and source catalogs (memory unknown at full sky / nside 8192).
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
    'default': dict(workers=32, max_blocks=16, walltime='1:00:00', drain_margin=None,
                    env={'OMP_WAIT_POLICY': 'PASSIVE'}),  # idle OpenMP threads sleep instead of spinning
    'highmem': dict(workers=4, max_blocks=4, walltime='1:00:00', drain_margin=20 * 60),
    'heavy': dict(workers=1, max_blocks=8, walltime='1:00:00', drain_margin=15 * 60),
}

# Debug QOS: 30 min walltime, at most 5 submitted and 2 running jobs per user, so the
# debug max_blocks sum to 5 (extra jobs wait in the queue). No draining in debug:
# blocks run until the walltime and tasks still running then are retried.
DEBUG = {
    'default': dict(max_blocks=2, drain_margin=None),
    'highmem': dict(max_blocks=1, drain_margin=None),
    'heavy': dict(max_blocks=2, drain_margin=None),
}


def _seconds(walltime):
    h, m, sec = (int(x) for x in walltime.split(':'))
    return 3600 * h + 60 * m + sec


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
        job_id = super().submit(*args, **kwargs)
        self.resources[job_id]['submit_time'] = time.time()
        return job_id

    def _status(self):
        active_jobs = {jid: job for jid, job in self.resources.items() if not job['status'].terminal}
        super()._status()
        now = time.time()
        for jid, job in active_jobs.items():
            submit_time = job.get('submit_time')
            if job['status'].state == JobState.COMPLETED and submit_time is not None and (now - submit_time) < self.grace_period_s:
                job['status'] = JobStatus(JobState.PENDING, stdout_path=job['status'].stdout_path, stderr_path=job['status'].stderr_path)


def load_run_config(path):
    """Read the YAML configuration of a run, expanding environment variables (e.g. ``$PSCRATCH``) in its top-level paths."""
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

    Notes
    -----
    The NERSC account is ``$POLLUX_ACCOUNT`` (default ``m1727``), and Slurm emails go to
    ``$POLLUX_MAIL`` if set. Workers inherit the environment of the run script.
    """
    qos = 'debug' if debug else 'regular'
    account = os.environ.get('POLLUX_ACCOUNT', 'm1727')
    mail = os.environ.get('POLLUX_MAIL')
    mail_options = f' \n#SBATCH --mail-user={mail} \n#SBATCH --mail-type=ALL' if mail else ''

    executors = []
    for label, e in EXECUTORS.items():
        if debug:
            e = {**e, **DEBUG[label], 'walltime': '00:30:00'}
        walltime_s = _seconds(e['walltime'])
        drain_period = None if e['drain_margin'] is None else walltime_s - e['drain_margin']
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
        retries=8,
        strategy='htex_auto_scale',  # also scale in idle blocks while other blocks still have tasks
        max_idletime=60,
    )
    return config
