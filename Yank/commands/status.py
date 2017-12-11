#!/usr/local/bin/env python

# =============================================================================================
# MODULE DOCSTRING
# =============================================================================================

"""
Query output files for quick status.

"""

# =============================================================================================
# MODULE IMPORTS
# =============================================================================================

import operator
import itertools
import collections

from .. import experiment

# =============================================================================================
# COMMAND-LINE INTERFACE
# =============================================================================================

usage = """
YANK status

Usage:
  yank status (-y FILEPATH | --yaml=FILEPATH) [--status=STRING] [--njobs=INTEGER] [-v | --verbose]

Description:
  Print the current status of the experiments.

Required Arguments:
  -y, --yaml=FILEPATH           Path to the YAML script specifying options and/or how to
                                set up and run the experiment.

General Options:
  --status=STRING               Print only the jobs in a particular status. Accepted values
                                are "completed", "ongoing", and "pending". This works only
                                if verbose is set.
  --njobs=INTEGER               Print the job id associated to each experiment assuming
                                njobs to be the one specified here.
  -v, --verbose                 Print status of each experiment individually. If this is
                                not set, only a summary of the status of all experiments
                                is printed.

"""


# =============================================================================================
# COMMAND DISPATCH
# =============================================================================================

def find_contiguous_ids(job_ids):
    """Return the contiguous job ids in the given list.

    Returns
    -------
    contiguous_job_ids : str
        The job ids organized in contiguous sets.

    """
    contiguous_job_ids = []
    for k, g in itertools.groupby(enumerate(job_ids), lambda x:x[0]-x[1]):
        group = list(map(operator.itemgetter(1), g))
        if len(group) == 1:
            contiguous_job_ids.append(str(group[0]))
        else:
            contiguous_job_ids.append('{}-{}'.format(group[0], group[-1]))
    return ','.join(contiguous_job_ids)


def dispatch(args):
    # Handle optional arguments.
    if args['--njobs']:
        n_jobs = int(args['--njobs'])
    else:
        n_jobs = None

    exp_builder = experiment.ExperimentBuilder(args['--yaml'], n_jobs=n_jobs)

    # Count all experiment status.
    job_ids_by_status = {
        'completed': [],
        'ongoing': [],
        'pending': []
    }
    counter = collections.Counter()
    for exp_status in exp_builder.status():
        counter[exp_status.status] += 1

        # Store job id.
        job_ids_by_status[exp_status.status].append(exp_status.job_id)

        # Print experiment and phases details.
        if args['--verbose']:
            # Filter by status if requested.
            if args['--status'] and args['--status'] != exp_status.status:
                continue

            # Print experiment information.
            exp_description = '{}: status={}'.format(exp_status.name,
                                                     exp_status.status)
            if n_jobs is not None:
                exp_description += ', job_id={}'.format(exp_status.job_id)
            print(exp_description)

            # Print phases information.
            for phase_name, phase_status in exp_status.phases.items():
                if phase_status.iteration is not None:
                    iteration = ', iteration={}/{}'.format(phase_status.iteration,
                                                           exp_status.number_of_iterations)
                else:
                    iteration = ''
                print('\t{}: status={}{}'.format(phase_name, phase_status.status, iteration))

    # Print summary.
    tot_n_experiments = sum(count for count in counter.values())
    summary_msg = ', '.join('{} {}'.format(count, status) for status, count in counter.items())
    print('Total number of experiments: {} ({})'.format(tot_n_experiments, summary_msg))

    if n_jobs is not None:
        for status, job_ids in job_ids_by_status.items():
            print('{} job ids: {}'.format(status, find_contiguous_ids(job_ids)))

    return True
