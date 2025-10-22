"""This module defines the concrete implementations of the different prechecks.
"""
from netCDF4 import Dataset
import logging
import numpy as np
import os
import pandas as pd
from pathlib import Path

from .interface import PrecheckResult, PrecheckSeverity, TcconPrecheck, TcconSanityCheck
from ..common_utils import FileFmtVer, grammatical_join, check_prior_index, correct_prior_index, md5sum_file

from typing import Optional, Union, Sequence, Dict


class ReadableFileCheck(TcconSanityCheck):
    def check_file(self, target_file: os.PathLike) -> Optional[PrecheckResult]:
        try:
            ds = Dataset(target_file)
        except Exception:
            result = self._make_result(target_file)
        else:
            ds.close()
            result = None

        return result

    @staticmethod
    def _make_result(target_file: os.PathLike) -> PrecheckResult:
        basename = os.path.basename(target_file)
        md5sum = md5sum_file(target_file)
        summary = f'The file {basename} cannot be read and is most likely corrupted'
        fix = (
            "First, confirm that you can read this file on your computer with ncdump, Panoply, or Python's netCDF4 package. "
            f"Also compare its MD5 checksum to the following: {md5sum}. If you can read the file on your computer and the "
            "checksum differs, it was likely corrupted during the upload to Caltech. Reupload the file and let the Caltech "
            "team know you have reuploaded it. However, if you cannot read your local copy, then write_netcdf likely crashed "
            "while you were running post-processing. Check your write_netcdf.log file for errors, make the necessary fixes, "
            "and rerun the write_netcdf line in post_processing.sh. Then check the new .nc file on your computer and, if you "
            "can read it, upload the new file to Caltech (and let the Caltech team know)."
        )
        return PrecheckResult(
            severity=PrecheckSeverity.ERROR,
            summary_line=summary,
            possible_fix=fix
        )



class FileFormatCheck(TcconPrecheck):
    """Checks that a new file has a file format attribute and it is the expected values.

    Parameters
    ----------
    expected_version
        The file format version (e.g. "2020.1.A") that the file must have.

    or_greater
        If ``True``, then any file format version greater than or equal to ``expected_version``
        is allowed.
    """
    def __init__(self, expected_version: Union[str, FileFmtVer], or_greater: bool = False):
        if isinstance(expected_version, str):
            expected_version = FileFmtVer(expected_version)
        self.expected_version = expected_version
        self.or_greater = or_greater

    def check_file(self, netcdf_handle: Dataset) -> Optional[PrecheckResult]:
        logging.info(f'Checking {netcdf_handle.filepath()} for file format version.')
        try:
            file_vers_str = netcdf_handle.file_format_version
        except AttributeError:
            logging.info(f'{netcdf_handle.filepath()} missing file format version attribute.')
            return PrecheckResult(
                severity=PrecheckSeverity.ERROR,
                summary_line='File does not contain a "file_format_version" attribute',
                possible_fix='Confirm that you are using the GGG2020.1 version of the GGG code.'
            )

        try:
            file_vers = FileFmtVer(file_vers_str)
        except ValueError as err:
            logging.info(f'{netcdf_handle.filepath()} has an invalid file format version attribute.')
            return PrecheckResult(
                severity=PrecheckSeverity.ERROR,
                summary_line=f'{err} (an error occurred while parsing the file format version)',
                possible_fix='Confirm that you are using the GGG2020.1 version of the GGG code.'
            )

        if self.or_greater and file_vers >= self.expected_version:
            logging.info(f'{netcdf_handle.filepath()} has an incorrect file format version.')
            return PrecheckResult(
                severity=PrecheckSeverity.ERROR,
                summary_line=f'Expected the file format version to be greater than or equal to "{self.expected_version}", instead got "{file_vers}"',
                possible_fix='Confirm that you are using the GGG2020.1 version of the GGG code.'
            )

        elif not self.or_greater and file_vers != self.expected_version:
            logging.info(f'{netcdf_handle.filepath()} has an incorrect file format version.')
            return PrecheckResult(
                severity=PrecheckSeverity.ERROR,
                summary_line=f'Expected the file format version to be "{self.expected_version}", instead got "{file_vers}"',
                possible_fix='Confirm that you are using the GGG2020.1 version of the GGG code.'
            )

        logging.info(f'{netcdf_handle.filepath()} has an acceptable file format version.')
        return None


class DupTimeCheck(TcconPrecheck):
    """Checks if the new file has any times within some delta time within a given threshold of existing files

    Parameters
    ----------
    existing_file
        Either a list of paths to existing files to check against or a dictionary mapping paths to a description
        of the file. In the detailed report, files will be referenced by their basename only, so using a description
        for files in different directories can help resolve which file is being referred to.

    max_dt_sec
        Two times will be considered the same if they different by fewer seconds that this value.
    """
    def __init__(self, existing_files: Union[Sequence[os.PathLike], Dict[os.PathLike, str]], max_dt_sec: float = 1.0):
        if not isinstance(existing_files, dict):
            self.existing_files = {f: 'file' for f in existing_files}
        else:
            self.existing_files = existing_files
        self.max_dt_sec = max_dt_sec

    def check_file(self, netcdf_handle: Dataset) -> Optional[PrecheckResult]:
        new_file_timestamps = netcdf_handle['time'][:]
        new_file_start = np.ma.min(new_file_timestamps)
        new_file_stop = np.ma.max(new_file_timestamps)

        dup_time_reports = []

        for filepath, filedesc in self.existing_files.items():
            logging.info(f'Checking {netcdf_handle.filepath()} against {filepath} for duplicate times.')
            with Dataset(filepath) as ds:
                old_file_timestamps = ds['time'][:]

                # This check should be faster than the broadcasted difference we'd have to do to actually compute if
                # any two spectra are close in time to each other
                if new_file_stop < np.ma.min(old_file_timestamps) or new_file_start > np.ma.max(old_file_timestamps):
                    continue

                dup_times = self._report_dup_times(new_file_timestamps, old_file_timestamps)
                if np.size(dup_times) > 0:
                    dup_time_reports.extend(self._make_dup_time_reports(filepath, filedesc, dup_times))

        if len(dup_time_reports) == 0:
            logging.info(f'{netcdf_handle.filepath()} has no duplicate times.')
            return None
        else:
            logging.info(f'{netcdf_handle.filepath()} has some duplicate times.')
            summary = f'Some times in this file were within {self.max_dt_sec} seconds of times in previously uploaded files. Unless this is a new revision, this must be corrected.'
            dup_time_reports.insert(0, f'{len(dup_time_reports)} also exist in other files:')
            return PrecheckResult(
                severity=PrecheckSeverity.WARNING,
                summary_line=summary,
                detailed_lines=dup_time_reports
            )


    def _report_dup_times(self, new_file_times: np.ndarray, old_file_times: np.ndarray) -> np.ndarray:
        dt = new_file_times[:, np.newaxis] - old_file_times[np.newaxis, :]
        is_dup = np.any(np.abs(dt) < self.max_dt_sec, axis=1)
        return new_file_times[is_dup]

    def _make_dup_time_reports(self, filepath: os.PathLike, filedesc: str, dup_times: np.ndarray) -> Sequence[str]:
        filename = Path(filepath).name
        times = pd.Timestamp(1970,1,1) + pd.to_timedelta(np.sort(dup_times), unit='s')
        reports = []
        for t in times:
            reports.append(f'- {t:%Y-%m-%dT%H:%M:%S} also in {filedesc} {filename}')
        return reports


class UnitScalingCheck(TcconPrecheck):
    EXPECTED_SCALINGS = {
        '': 1.0,
        '1': 1.0,
        'ppm': 1e6,
        'ppmv': 1e6,
        'ppb': 1e9,
        'ppbv': 1e9,
        'ppt': 1e12,
        'pptv': 1e12,
        'ppq': 1e15,
        'ppqv': 1e15,
    }

    def check_file(self, netcdf_handle: Dataset) -> Optional[PrecheckResult]:
        logging.info(f'Checking {netcdf_handle.filepath()} for OOF scaling/unit mismatches.')
        mismatches = dict()
        unknown_units = dict()
        for varname, variable in netcdf_handle.variables.items():
            # This will apply only to Xgas variables (not their errors)
            if varname.startswith('x') and 'error' not in varname and hasattr(variable, 'oof_scaling'):
                scaling = variable.oof_scaling
                unit = variable.units
                if unit not in self.EXPECTED_SCALINGS:
                    unknown_units[varname] = f'For variable "{varname}", the unit "{unit}" does not have an expected scaling'
                elif not np.isclose(scaling, self.EXPECTED_SCALINGS[unit]):
                    mismatches[varname] = f'For variable "{varname}", the scaling of {scaling:.1g} is inconsistent with the unit "{unit}" (expected {self.EXPECTED_SCALINGS[unit]:.1g})'


        has_mismatch = len(mismatches) > 0
        has_unknown = len(unknown_units) > 0
        if not has_mismatch and not has_unknown:
            logging.info(f'All OOF scalings and units in {netcdf_handle.filepath()} are consistent')
            return None
        else:
            logging.info(f'{netcdf_handle.filepath()} has at least one unknown OOF unit or OOF scaling/unit mismatch')

        severity = PrecheckSeverity.ERROR if has_mismatch else PrecheckSeverity.WARNING
        summaries = []
        fixes = []
        if has_mismatch:
            variables = grammatical_join(sorted(mismatches.keys()))
            summaries.append(f'{len(mismatches)} variables have OOF scalings inconsistent with their units ({variables})')
            fixes.append('For incorrect scalings, check your qc.dat file and confirm that the scaling for these variables matches their units')
        if has_unknown:
            variables = grammatical_join(sorted(unknown_units.keys()))
            summaries.append(f'{len(unknown_units)} variables that are scaled by the scale value in the qc.dat units that could not be checked for the correct scaling ({variables})')
            fixes.append('For unknown units, double check the spelling of the unit in the qc.dat file. If that is correct, respond in GGGBugs with what scaling that unit represents.')

        return PrecheckResult(
            severity=severity,
            summary_line=' '.join(summaries),
            possible_fix=' '.join(fixes),
            detailed_lines=list(mismatches.values()) + list(unknown_units.values())
        )


class PriorMismatchCheck(TcconPrecheck):
    """Check that the prior times are close enough to the corresponding ZPD times.

    Parameters
    ----------
    max_dt_hr
        How large a time difference (in hours) can be between the ZPD time and prior time before
        it is considered an error. The default value is chosen to be slightly larger than the value
        used in the netCDF writer itself (by the :func:`check_prior_index` function) to avoid false
        positives from numeric issues.
    """
    def __init__(self, max_dt_hr: float = 1.56):
        self.max_dt_hr = max_dt_hr

    def check_file(self, netcdf_handle: Dataset) -> Optional[PrecheckResult]:
        logging.info(f'Checking {netcdf_handle.filepath()} for mismatched prior/ZPD times')
        if check_prior_index(netcdf_handle, max_delta_hours=self.max_dt_hr):
            logging.info(f'All prior/ZPD times in {netcdf_handle.filepath()} are consistent')
            return None
        else:
            logging.info(f'{netcdf_handle.filepath()} has at least one inconsistent prior/ZPD time')
            report = self._make_report(netcdf_handle)
            fix = (
                '(1) Ensure you are running the GGG2020.1 code, as the netCDF write should automatically fix this from that version on, '
                '(2) Check your write_netcdf.log file for errors, in case the writer exited before applying the fix, '
                '(3) Ensure that the runlog contains all spectra referenced in the .mav file "next spectrum" lines and that the spectra '
                'are consistent among the runlog, .col files, and post processing files.'
            )
            return PrecheckResult(
                severity=PrecheckSeverity.ERROR,
                summary_line=f'At least one prior time is more than {self.max_dt_hr} hours away from its corresponding spectrum ZPD time',
                possible_fix=fix,
                detailed_lines=report
            )

    def _make_report(self, netcdf_handle: Dataset) -> Sequence[str]:
        zpd_times = pd.Timestamp(1970,1,1) + pd.to_timedelta(netcdf_handle['time'][:], unit='s')
        prior_times = pd.Timestamp(1970,1,1) + pd.to_timedelta(netcdf_handle['prior_time'][:], unit='s')
        curr_inds = netcdf_handle['prior_index'][:]
        new_inds = correct_prior_index(netcdf_handle, assign=False)
        ii_wrong = np.flatnonzero(curr_inds != new_inds)

        report = ['The following prior index values are incorrect:']
        for i in ii_wrong:
            report.append(
                f'- Spectrum at index {i} ({zpd_times[i]:%Y-%m-%dT%H:%M:%S}): '
                f'current prior index and time = {curr_inds[i]}, {prior_times[curr_inds[i]]:%Y-%m-%dT%H:%M:%S}, '
                f'suggested prior index and time = {new_inds[i]}, {prior_times[new_inds[i]]:%Y-%m-%dT%H:%M:%S}, '
            )
        return report


class FileNameDateCheck(TcconPrecheck):
    """Check that all the times for spectra a netCDF file are within the dates given in its file name.
    """
    def check_file(self, netcdf_handle: Dataset) -> Optional[PrecheckResult]:
        logging.info(f'Checking if all times in {netcdf_handle.filepath()} are inside the filename dates')
        filename = os.path.basename(netcdf_handle.filepath())
        stem = filename.split('.')[0]
        fname_start = pd.to_datetime(stem.split('_')[0][2:])
        # Because the end date is the date of the last .aia entry, we must add a day
        # so that data on that date don't get counted as after the end. For example,
        # if the last spectrum was on 2025-07-14 12:34, the end date in the name would
        # be 2025-07-14 (00:00), so for that 12:34 spectrum to be allowed, we must
        # actually check for data after 2025-07-15.
        fname_end = pd.to_datetime(stem.split('_')[1]) + pd.Timedelta(days=1)

        data_times = pd.Timestamp(1970,1,1) + pd.to_timedelta(netcdf_handle['time'][:], unit='s')
        ii_out = (data_times < fname_start) | (data_times > fname_end)
        if np.any(ii_out):
            logging.info(f'{netcdf_handle.filepath()} has at least one time outside the dates in its filename')
            summary, report = self._summarize_and_report(ii_out, data_times)
            fix = (
                'Confirm that you are using the GGG2020.1 version of the GGG code, as that should use the min and max times to '
                'compute the file name, rather than the first and last times. If that does not help, also ensure that the first '
                'and last spectra in your runlog are the earliest and latest spectra, respectively.'
            )
            return PrecheckResult(
                severity=PrecheckSeverity.ERROR,
                summary_line=summary,
                possible_fix=fix,
                detailed_lines=report
            )
        else:
            logging.info(f'All times in {netcdf_handle.filepath()} are inside the dates in its filename')
            return None

    def _summarize_and_report(self, ii_out, times):
        n_out = np.sum(ii_out)
        if n_out <= 3:
            time_strs = [t.strftime('%Y-%m-%dT%H:%M:%S') for t in times[ii_out]]
        else:
            time_strs = [t.strftime('%Y-%m-%dT%H:%M:%S') for t in times[ii_out][:2]]
            time_strs.append(f'{n_out-2} others')

        time_str = grammatical_join(time_strs)
        summary = f'{n_out} spectra have ZPD times ({time_str}) outside the dates listed in the file name'
        report = ['Out-of-range times are:']
        for i in np.flatnonzero(ii_out):
            report.append(f'- {times[i]:%Y-%m-%dT%H:%M:%S} (index = {i})')
        return summary, report



class ChronologicalOrderCheck(TcconPrecheck):
    """Check that the spectra and associated data are ordered by time
    """
    def check_file(self, netcdf_handle: Dataset) -> Optional[PrecheckResult]:
        logging.info(f'Checking if all spectra in {netcdf_handle.filepath()} are in chronological order')
        times = netcdf_handle['time'][:]
        dt = np.diff(times)
        ii_neg = dt < 0
        if np.any(ii_neg):
            logging.info(f'{netcdf_handle.filepath()} has at least one spectrum out of chronological order')
            report = self._build_report(times, ii_neg)
            return PrecheckResult(
                severity=PrecheckSeverity.INFO,
                summary_line='There is at least one spectrum not in chronological order. Note that the spectra will be put in chronological order in the public files.',
                detailed_lines=report
            )
        else:
            logging.info(f'All spectra in {netcdf_handle.filepath()} are in chronological order')
            return None

    def _build_report(self, times, ii_neg):
        report = ['The following negative time jumps were identified:']
        for i in np.flatnonzero(ii_neg):
            t1 = pd.Timestamp(1970,1,1) + pd.to_timedelta(times[i], unit='s')
            t2 = pd.Timestamp(1970,1,1) + pd.to_timedelta(times[i+1], unit='s')
            report.append(f'- {t1:%Y-%m-%dT%H:%M:%S} (index {i}) to {t2:%Y-%m-%dT%H:%M:%S} (index {i+1})')
        return report
