"""This module defines the interface for the prechecks and the common function to run those checks.
"""
from abc import ABC, abstractmethod
from enum import Enum
from netCDF4 import Dataset
import os
from pathlib import Path

from typing import Optional, Sequence, IO


class PrecheckSeverity(Enum):
    """An enum representing whether a precheck failure is a critical problem or not. Variants:

    - ERROR: failure is a critical problem, file MUST be corrected before moving into QA/QC
    - WARNING: failure may not be critical; file should not be automatically moved to QA/QC
      but admins may elect to override and move it into QA/QC as-is if the problem is acceptable
      in their judgement.
    - INFO: failure is not critical or was automatically corrected. This should be noted in the output,
      but the file should still be moved to QA/QC automatically.
    """
    INFO = 'info'
    WARNING = 'warning'
    ERROR = 'error'

    # Credit to https://stackoverflow.com/a/71839532 for the ordering code
    def __lt__(self, other):
        if self == other:
            return False
        # the following works because the order of elements in the definition is preserved
        for elem in PrecheckSeverity:
            if self == elem:
                return True
            elif other == elem:
                return False
        raise RuntimeError('Bug: we should never arrive here')

    def __gt__(self, other):
        return not (self < other)

    def __ge__(self, other):
        if self == other:
            return True
        return not (self < other)

    @property
    def display_order(self) -> int:
        if self == PrecheckSeverity.INFO:
            return 30
        if self == PrecheckSeverity.WARNING:
            return 20
        if self == PrecheckSeverity.ERROR:
            return 10

        return 0

    def max_severity_message(self, target_basename) -> str:
        if self == PrecheckSeverity.INFO:
            return f'{target_basename} passed the pre-QA/QC checks. However, there are some issues to be aware of.'
        elif self == PrecheckSeverity.WARNING:
            return f'{target_basename} has one or more issues that to reviewed before proceeding to QA/QC.'
        elif self == PrecheckSeverity.ERROR:
            return f'{target_basename} has one or more CRITICAL issues that MUST BE corrected before proceeding to QA/QC.'
        else:
            return f'{target_basename} has at least one {self.value}.'

    @property
    def meaning(self) -> str:
        if self == PrecheckSeverity.INFO:
            return 'For information only. Does not prevent a file from automatically proceeding to QA/QC.'
        if self == PrecheckSeverity.WARNING:
            return 'A issue that may require reprocessing. Discussion with editor/reviewers/alg team required before proceeding to QA/QC.'
        if self == PrecheckSeverity.ERROR:
            return 'A issue that does require reprocessing. If you do not know how to correct this, please ask for help in the QA/QC topic.'

        return 'A meaning has not been defined for this severity! (Please let the alg team know you saw this message.)'

    @property
    def proceed_automatically(self) -> bool:
        return self <= PrecheckSeverity.INFO


class PrecheckResult:
    """A class summarizing a failure of a pre-QA/QC consistency check. Fields:

    - severity: whether this is a critical, moderate, or fixable error.
    - summary_line: a single line summarizing the failure that can be printed to the screen or sent
      in an email body.
    - detailed_lines: a list of lines that describe the failure in detail that will usually be appended to
      a report file.
    """
    def __init__(self, severity: PrecheckSeverity, summary_line: str, possible_fix: Optional[str] = None, detailed_lines: Sequence[str] = ()):
        self.severity = severity
        self.summary_line = summary_line
        self.possible_fix = possible_fix
        self.detailed_lines = detailed_lines


class TcconPrecheck(ABC):
    @abstractmethod
    def check_file(self, netcdf_handle: Dataset) -> Optional[PrecheckResult]:
        """Run this check on the given netCDF file and return a summary and detailed report.

        If there are no issues with this file, this method must return ``None``. Otherwise, it must
        return an instance of ``PrecheckResult`` that describes the failure.
        """
        pass



def run_checks(target_file: os.PathLike, checks: Sequence[TcconPrecheck], summary_writer: IO, details_writer: IO) -> Optional[PrecheckSeverity]:
    """Run a series of pre-QA/QC checks on a target file and report the results.

    Parameters
    ----------
    target_file
        Path to the netCDF file to check.

    checks
        The set of prechecks to apply.

    summary_writer
        A type to which the summary results of the checks and possible fixes will be written -
        usually either an open file or an in-memory buffer like :class:`io.StringIO`.

    details_writer
        A type to which the detailed results of the checks will be written - usually either
        an open file or an in-memory buffer like :class:`io.StringIO`.

    Returns
    -------
    max_severity
        The highest severity of all the check results, which can be used to determine whether
        a file can proceed to QA/QC. If no checks failed, this will be ``None``.
    """
    results = []
    with Dataset(target_file) as ds:
        for check in checks:
            res = check.check_file(ds)
            if res is not None:
                results.append(res)

    target_basename = Path(target_file).name
    if len(results) == 0:
        summary_writer.write(f'{target_basename} passed all pre-QA/QC checks')
        return None

    max_severity = max(r.severity for r in results)
    # sort the results so that the most critical issues are first
    results = sorted(results, key=lambda r: r.severity.display_order)

    summary_writer.write(max_severity.max_severity_message(target_basename))
    summary_writer.write('\n\nBelow is a summary of the issued identified. The meaning of the severity levels is as follows:\n')
    for severity in PrecheckSeverity:
        summary_writer.write(f'- {severity.value}: {severity.meaning}\n')
    summary_writer.write('\n')

    for index, issue in enumerate(results, start=1):
        summary_writer.write(f'Issue {index} ({issue.severity.value}): {issue.summary_line}\n')
        if issue.possible_fix is not None:
            summary_writer.write(f' --> Possible fix: {issue.possible_fix}\n')
        summary_writer.write('\n')

        if len(issue.detailed_lines) == 0:
            details_writer.write(f'Issue {index}: No additional details.\n\n')
        else:
            details_writer.write(f'Issue {index}: ')
            for line in issue.detailed_lines:
                details_writer.write(f'{line}\n')
            details_writer.write('\n')

    return max_severity
