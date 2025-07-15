from argparse import ArgumentParser
from contextlib import ExitStack
from glob import glob
import json
import os
import sys

from . import definitions as defs
from .interface import run_checks, PrecheckSeverity
from ..file_format_updating import v2020C
from ..common_utils import setup_logging
from ..constants import LOG_LEVEL_CHOICES

from typing import Dict, IO, Optional, Sequence, Union


def main():
    p = ArgumentParser(description='Run pre-QA/QC checks on a private netCDF file')
    p.add_argument('target_file', help='The private netCDF file to check.')
    p.add_argument('-s', '--summary', default='STDOUT',
                   help='Where to write the summary. A value of "-" or "STDOUT" writes to standard out, "STDERR" to standard error, '
                        'and any other value is assumed to be a path to a file to write. Default is %(default)s.')
    p.add_argument('-d', '--details', default='precheck_details.txt',
                   help='Where to write the details. Accepts the same values as --summary. Default is %(default)s.')
    p.add_argument('--log-level', default='WARNING', type=lambda x: x.upper(), choices=LOG_LEVEL_CHOICES,
                   help="Log level for the screen (no log file will be written). Default is %(default)s",)
    p.add_argument('--pdb', action='store_true', help='Launch the python debugger')

    subp = p.add_subparsers()
    filep = subp.add_parser('files', help='Specify existing file to check against for duplicate times as arguments.')
    filep.add_argument('files', nargs='*', help='Individual existing private netCDF files to check')
    filep.set_defaults(ex_files_fxn=_get_files_list)

    globp = subp.add_parser('globs', help='Specify existing files (and their labels) to check against for duplicate times with a glob pattern.')
    globp.add_argument('labels_and_globs', nargs='*',
                       help='Alternate file labels and glob patterns, e.g.: "published" "/data/pub/*.nc" "preliminary" "/data/prelim/*.nc". '
                            'For most shells, the globs must be quoted to avoid expanding then in the shell.')
    globp.set_defaults(ex_files_fxn=_get_files_glob)

    jsonp = subp.add_parser('json', help='Specify existing files (optionally with labels) in a JSON file')
    jsonp.add_argument('json_file', help='A JSON file containing either (a) a list of paths to existing netCDF files or '
                                         '(b) a map of netCDF files to labels for them to check against for duplicate times.')
    jsonp.set_defaults(ex_files_fxn=_get_files_json)

    p.epilog = (
        'If you do not specify any of the "files", "globs", or "json", then there will be no check for duplicate times. '
        'The exit codes are as follows: a 0 indicates that the file passed all checks (and the report has no information), '
        'a 2 indicates that the file can be moved to QA/QC, but there is information in the report, and '
        'a 4 indicates that the file has issues that must be addressed before moving to QA/QC. '
        'A 1 indicates a Python error.'
    )

    clargs = vars(p.parse_args())
    if clargs.pop('pdb'):
        import pdb
        pdb.set_trace()
    setup_logging(clargs['log_level'], log_file=None)

    get_files_fxn = clargs.get('ex_files_fxn', _get_no_files)

    existing_files = get_files_fxn(clargs)

    with ExitStack() as stack:
        summary_writer = _open_writer(clargs['summary'], stack)
        details_writer = _open_writer(clargs['details'], stack)
        severity = driver(
            target_file=clargs['target_file'],
            existing_files=existing_files,
            summary_writer=summary_writer,
            details_writer=details_writer
        )

    if severity is None:
        sys.exit(0)
    elif severity.proceed_automatically:
        sys.exit(2)
    else:
        sys.exit(4)


def _get_files_list(args):
    return args['files']

def _get_files_glob(args):
    n = len(args['labels_and_globs'])
    if n % 2 != 0:
        raise ValueError(f'Must be an even number of labels and globs (got {n})')

    files = dict()
    for i in range(0, n, 2):
        label = args['labels_and_globs'][i]
        pattern = args['labels_and_globs'][i+1]
        for f in glob(pattern):
            files[f] = label
    return files

def _get_files_json(args):
    with open(args['json_file']) as f:
        return json.load(f)

def _get_no_files(args):
    return []


def _open_writer(path: str, stack: ExitStack) -> IO:
    if path in {'-', 'STDOUT'}:
        return sys.stdout
    if path == 'STDERR':
        return sys.stderr

    return stack.enter_context(open(path, 'w'))



def driver(target_file: os.PathLike, existing_files: Union[Sequence[os.PathLike], Dict[os.PathLike, str]], summary_writer: IO, details_writer: IO) -> Optional[PrecheckSeverity]:
    checks = [
        defs.FileFormatCheck(v2020C),
        defs.DupTimeCheck(existing_files),
        defs.UnitScalingCheck(),
        defs.PriorMismatchCheck(),
        defs.FileNameDateCheck(),
        defs.ChronologicalOrderCheck()
    ]

    return run_checks(target_file, checks, summary_writer=summary_writer, details_writer=details_writer)


if __name__ == '__main__':
    main()
