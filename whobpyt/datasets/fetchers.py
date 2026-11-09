"""Fetchers for the WhoBPyT example datasets.

Each ``fetch_*`` function downloads one example dataset (once) and returns
the folder it was saved to. By default, data are saved under
``~/.whobpyt/data/<dataset name>``.

Modified from code in grifflab/kernel-flow-tools; inspired by nilearn and MNE.
"""

import os
import shutil
import tempfile
import warnings
import zipfile

import requests


WHOBPYT_DATA_FOLDER = '~/.whobpyt/data'
OSF_DOWNLOAD_URL = 'https://osf.io/download'

_CHUNK_SIZE = 1024 * 1024
_TIMEOUT = 60  # seconds, per network read


def get_localdefaultdatapath():
    """Return the default data folder (``~/.whobpyt/data``, expanded)."""
    return os.path.expanduser(WHOBPYT_DATA_FOLDER)


# --------------------------------------------------------------------------
# Low-level download helpers
# --------------------------------------------------------------------------

def _download_url(url, destination):
    """Stream ``url`` to ``destination``.

    The file is written to a temporary name first and only moved into place
    once complete, so an interrupted download never leaves a partial file
    that later calls would mistake for a finished one.
    """
    dest_dir = os.path.dirname(os.path.abspath(destination))
    os.makedirs(dest_dir, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=dest_dir, suffix='.part')
    try:
        with requests.get(url, stream=True, timeout=_TIMEOUT) as response:
            response.raise_for_status()
            with os.fdopen(fd, 'wb') as f:
                for chunk in response.iter_content(_CHUNK_SIZE):
                    if chunk:
                        f.write(chunk)
        os.replace(tmp_path, destination)
    except BaseException:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


def _gdown():
    try:
        import gdown
    except ImportError as err:
        raise ImportError(
            'Downloading this dataset from Google Drive needs the gdown package: '
            'pip install gdown') from err
    return gdown


def pull_file(dlcode, destination, download_method='wget'):
    """Download a single file to ``destination``.

    Parameters
    ----------
    dlcode : str
        A full URL (``download_method='wget'``, the name is kept for backward
        compatibility; the download is done in Python, not with the ``wget``
        program), or a Google Drive file ID (``'gdown'`` or ``'requests'``).
    destination : str
        Path of the file to write.
    download_method : {'wget', 'gdown', 'requests'}
    """
    print('Downloading %s' % os.path.basename(destination))
    if download_method == 'wget':
        _download_url(dlcode, destination)
    elif download_method in ('gdown', 'requests'):
        dest_dir = os.path.dirname(os.path.abspath(destination))
        os.makedirs(dest_dir, exist_ok=True)
        out = _gdown().download(id=dlcode, output=destination, quiet=False)
        if out is None:
            raise RuntimeError('Failed to download Google Drive file %s' % dlcode)
    else:
        raise ValueError("download_method must be 'wget', 'gdown' or 'requests', "
                         "got %r" % (download_method,))


def pull_folder(dlcode, newfolder_name, dest_folder=None, download_method='wget'):
    """Download a whole folder into ``dest_folder/newfolder_name``.

    Skipped if that folder already exists.

    Parameters
    ----------
    dlcode : str
        URL of a zip archive (``download_method='wget'``) or a Google Drive
        folder ID (``'gdown'``).
    newfolder_name : str
        Name of the folder to create inside ``dest_folder``.
    dest_folder : str | None
        Parent folder. Defaults to :func:`get_localdefaultdatapath`.
    download_method : {'wget', 'gdown'}

    Returns
    -------
    str
        Path of the downloaded folder.
    """
    if dest_folder is None:
        dest_folder = get_localdefaultdatapath()
    newfolder = os.path.join(dest_folder, newfolder_name)
    if os.path.isdir(newfolder) and os.listdir(newfolder):
        return newfolder

    # Download into a temporary folder and move it into place at the end, so
    # a failed download does not leave a half-filled folder behind.
    os.makedirs(dest_folder, exist_ok=True)
    tmp_folder = tempfile.mkdtemp(dir=dest_folder, prefix='.%s-' % newfolder_name)
    try:
        if download_method == 'wget':
            zip_path = os.path.join(tmp_folder, 'download.zip')
            _download_url(dlcode, zip_path)
            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(tmp_folder)
            os.remove(zip_path)
        elif download_method == 'gdown':
            url = 'https://drive.google.com/drive/folders/' + dlcode
            out = _gdown().download_folder(url=url, output=tmp_folder, quiet=False)
            if not out:
                raise RuntimeError('Failed to download Google Drive folder %s' % dlcode)
        else:
            raise ValueError("download_method must be 'wget' or 'gdown', got %r"
                             % (download_method,))
        if os.path.isdir(newfolder):
            os.rmdir(newfolder)  # only reached if it exists and is empty
        os.replace(tmp_folder, newfolder)
    except BaseException:
        shutil.rmtree(tmp_folder, ignore_errors=True)
        raise
    print('Downloaded %s to %s' % (dlcode, newfolder))
    return newfolder


def _fetch_osf_files(files_dict, dataset_name, dest_folder=None, redownload=False):
    """Download OSF files ``{osf_id: relative_path}`` into a dataset folder.

    Files that are already present are skipped, so an interrupted download
    can be resumed by calling the fetcher again.
    """
    if dest_folder is None:
        dest_folder = os.path.join(get_localdefaultdatapath(), dataset_name)
    if redownload and os.path.isdir(dest_folder):
        shutil.rmtree(dest_folder)
    os.makedirs(dest_folder, exist_ok=True)

    missing = {osf_id: rel_path for osf_id, rel_path in files_dict.items()
               if not os.path.isfile(os.path.join(dest_folder, rel_path))}
    for i, (osf_id, rel_path) in enumerate(missing.items(), start=1):
        print('[%d/%d] ' % (i, len(missing)), end='')
        pull_file('%s/%s' % (OSF_DOWNLOAD_URL, osf_id),
                  os.path.join(dest_folder, rel_path), download_method='wget')
    return dest_folder


# --------------------------------------------------------------------------
# Example dataset fetchers
# --------------------------------------------------------------------------

def fetch_egtmseeg(dest_folder=None, redownload=False):
    """Fetch the data for the TMS-EEG example (one subject).

    Parameters
    ----------
    dest_folder : str | None
        Where to save the files. Defaults to ``~/.whobpyt/data/eg__tmseeg``.
    redownload : bool
        If True, delete any existing copy and download again.

    Returns
    -------
    str
        The folder containing the data.
    """
    files_dict = {
        '680fc6eb5210d93da17c5a1d': 'Schaefer2018_200Parcels_7Networks_count.csv',
        '680fe69f9ba48ef173568c07': 'Schaefer2018_200Parcels_7Networks_order_FSLMNI152_2mm.Centroid_RAS.txt',
        '680fc6fbab92f2b5627c5a0f': 'stim_weights.npy',
        '680fc6f63068d66fc8568e3e': 'Subject_1_low_voltage.fif',
        '680fc6f80959f56dde568f29': 'Subject_1_low_voltage_fittingresults_stim_exp.pkl',
        '680fc6f99f030183213cbfd3': 'Subject_1_low_voltage_lf.npy',
    }
    return _fetch_osf_files(files_dict, 'eg__tmseeg', dest_folder, redownload)


def fetch_egmomi2023(dest_folder=None, redownload=False):
    """Fetch the data for the Momi et al. (2023) example.

    Parameters
    ----------
    dest_folder : str | None
        Where to save the files. Defaults to ``~/.whobpyt/data/eg__momi2023``.
    redownload : bool
        If True, delete any existing copy and download again.

    Returns
    -------
    str
        The folder containing the data.
    """
    files_dict = {
        'h7nmz':                    'sub_0_fittingresults_stim_exp.pkl',
        '681075a6e72532ce337c5a4a': 'sub_1_fittingresults_stim_exp.pkl',
        '6810c2f5839107c1b17c5a99': 'sub_2_fittingresults_stim_exp.pkl',
        '6810dfaca4dd9539133cc0c2': 'sub_3_fittingresults_stim_exp.pkl',
        '681168b4d9b48f0d6d568bda': 'sub_4_fittingresults_stim_exp.pkl',
        '681168ce093e4604787c5a8d': 'sub_5_fittingresults_stim_exp.pkl',
        '68225ef961764a34f5eadf3c': 'sub_0_leadfield',
        '68225efa143105dcdbc4c2bb': 'sub_1_leadfield',
        '68225ef7143105dcdbc4c2b7': 'sub_2_leadfield',
        '68225ef761764a34f5eadf3a': 'sub_3_leadfield',
        '6822923f01f350cd9a9c41a6': 'sub_4_leadfield',
        '68229241b71e556932ce9dcf': 'sub_5_leadfield',
        '68226048878c115ac49c4483': 'stim_weights.npy',
        '6810759b339396b1eb7c5b28': 'Schaefer2018_200Parcels_7Networks_distance.csv',
        '68107591430f2af683568cb2': 'Schaefer2018_200Parcels_7Networks_count.csv',
        '6810758e11d6cb1d67568d05': 'rh.Schaefer2018_200Parcels_7Networks_order.annot',
        '6810758c78cb4eb6283cc01e': 'real_EEG',
        '6810758752a8beebd55ef7b9': 'only_high_trial.mat',
        '681074df24f63ddb219fddf8': 'lh.Schaefer2018_200Parcels_7Networks_order.annot',
        '68107567fed207be8f3cbf9f': 'all_avg.mat_avg_high_epoched',
        '681074d878cb4eb6283cbffa': 'leadfield',
    }
    return _fetch_osf_files(files_dict, 'eg__momi2023', dest_folder, redownload)


def fetch_egismail2025(dest_folder=None, redownload=False):
    """
    Deprecated alias for :func:`fetch_egismail2026`.

    The Ismail et al. paper was published in 2026, and its example dataset is
    now fetched by ``fetch_egismail2026``. This name is kept so that older
    code keeps working; it will be removed in a future release.
    """
    warnings.warn(
        "fetch_egismail2025 is deprecated; use fetch_egismail2026 instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return fetch_egismail2026(dest_folder=dest_folder, redownload=redownload)


def fetch_egismail2026(dest_folder=None, redownload=False):
    """Fetch the data for the Ismail et al. (2026) example.

    Parameters
    ----------
    dest_folder : str | None
        Where to save the files. Defaults to ``~/.whobpyt/data/eg__ismail2026``.
    redownload : bool
        If True, delete any existing copy and download again.

    Returns
    -------
    str
        The folder containing the data.
    """
    files_dict = {
        '4xw2t': 'distance.txt',
        'dbwp9': 'emp_noise_source.npy',
        'h6gjx': 'emp_verb_source.npy',
        '5e68s': 'info.pkl',
        's6wvn': 'info_nosubjinfo.pkl',
        'rkzjw': 'leadfield_3d.mat',
        'edcg2': 'noise_evoked.npy',
        'gmr72': 'sim_noise_sensor.npy',
        'ud9vf': 'sim_noise_source.npy',
        '5tw97': 'sim_verb_sensor.npy',
        'wgcza': 'sim_verb_source.npy',
        'mp5ut': 'verb_evoked.npy',
        'cz8bj': 'weights.csv',
    }
    return _fetch_osf_files(files_dict, 'eg__ismail2026', dest_folder, redownload)


def fetch_egmomi2025(dest_folder=None, redownload=False):
    """Fetch the data for the Momi et al. (2025) example.

    Parameters
    ----------
    dest_folder : str | None
        Where to save the files. Defaults to ``~/.whobpyt/data/eg__momi2025``.
    redownload : bool
        If True, delete any existing copy and download again.

    Returns
    -------
    str
        The folder containing the data, organised into the sub-folders
        ``anatomical``, ``calculate_distance``, ``empirical_data``,
        ``example_fittingresults`` and ``virtual_dissection``.
    """
    files_dict = {
        '6824a8c71ff4f0e5d9ce9f63': 'anatomical/example-bem',
        '6824a8b613f407a746c4c0f0': 'anatomical/example-src.fif',
        '6824a8a91ff4f0e5d9ce9f59': 'anatomical/example-trans.fif',
        '6824a8ac83462d95da9c415e': 'anatomical/lh.Schaefer2018_200Parcels_7Networks_order.annot',
        '6824a8ae66baea5d447a1b32': 'anatomical/rh.Schaefer2018_200Parcels_7Networks_order.annot',
        '6824a8b193c4fffcc97a1c1e': 'anatomical/sub_dist.pickle',
        '6824a75c66baea5d447a1afc': 'calculate_distance/example_ieeg.fif',
        '6824c8b91fc5936557c4c0a7': 'calculate_distance/example_Schaefer2018_200Parcels_7Networks_rewritten.nii',
        '6824a80bcae9ca1675c4c1fb': 'empirical_data/all_eeg_evoked.npy',
        '6824c909325133be06eae20a': 'empirical_data/all_epo_seeg.pkl',
        '6824c8ef57a8ae682f9c4456': 'empirical_data/example_epoched.fif',
        '6824c8f1f0bdacd73cce9eb7': 'empirical_data/dist_Schaefer_1000parcels_7net.pkl',
        '6824b6d93bf9c82dfbce9fa1': 'example_fittingresults/example-fittingresults.pkl',
        '6824a6f7b8afb1d4399c43bc': 'virtual_dissection/dist_Schaefer_1000parcels_7net.pkl',
        '6824a6f57de85602327a1bc5': 'virtual_dissection/eeg_template.fif',
        '6824a73282323cc35c9c4207': 'virtual_dissection/model_results.npy',
    }
    return _fetch_osf_files(files_dict, 'eg__momi2025', dest_folder, redownload)


def fetch_eggriffiths2022(dest_folder=None, redownload=False):
    """Fetch the data for Griffiths et al. (2022), as one OSF folder.

    Parameters
    ----------
    dest_folder : str | None
        Parent folder to save into. The data go in an ``eg__griffiths2022``
        sub-folder. Defaults to ``~/.whobpyt/data``.
    redownload : bool
        If True, delete any existing copy and download again.

    Returns
    -------
    str
        The folder containing the data.
    """
    osf_folder_url = ('https://files.ca-1.osf.io/v1/resources/u7kgw/providers/'
                      'osfstorage/6837772e428f2b32e585b232/?zip=')
    if dest_folder is None:
        dest_folder = get_localdefaultdatapath()
    newfolder = os.path.join(dest_folder, 'eg__griffiths2022')
    if redownload and os.path.isdir(newfolder):
        shutil.rmtree(newfolder)
    return pull_folder(osf_folder_url, newfolder_name='eg__griffiths2022',
                       dest_folder=dest_folder, download_method='wget')


# --------------------------------------------------------------------------
# Older datasets hosted on Google Drive
# --------------------------------------------------------------------------

def fetch_hcpl2k8(dest_folder=None):
    """Fetch the ``hcpl2k8`` dataset from Google Drive.

    Returns the path of the ``hcpl2k8`` folder inside ``dest_folder``
    (default ``~/.whobpyt/data``).
    """
    return pull_folder('18smy3ElTd4VksoL4Z15dhwT5l3yjk6xS', newfolder_name='hcpl2k8',
                       dest_folder=dest_folder, download_method='gdown')


def fetch_MomiEtAlELife2023(dest_folder=None):
    """Fetch the full Momi et al. (2023) reproduction data from Google Drive.

    Downloads the ``DATA_LITE`` folder and the ``fsaverage`` folder into
    ``dest_folder`` (default ``~/.whobpyt/data/MomiEtAlELife2023``). For the
    example in the documentation, use :func:`fetch_egmomi2023` instead.

    Returns
    -------
    str
        The folder containing the data.
    """
    if dest_folder is None:
        dest_folder = os.path.join(get_localdefaultdatapath(), 'MomiEtAlELife2023')
    pull_folder('1t-9m0E88xUUcGWWH024H32maQUs_Jjgx', newfolder_name='data',
                dest_folder=dest_folder, download_method='gdown')
    pull_folder('1YPyf3h9YKnZi0zRwBwolROQuqDtxEfzF', newfolder_name='fsaverage',
                dest_folder=dest_folder, download_method='gdown')
    return dest_folder
