"""Utility functions

This module contains various helpful functions to accommodate
the cbadc toolbox.
"""
import struct
from typing import Generator, Iterator, Union
import numpy as np
from scipy.signal import welch
from typing import Tuple
from tqdm import tqdm
import requests
import os
import pickle


def number_of_bytes_selector(M: int):
    """A helper function for selecting
    the right bytes settings.

    Parameters
    ----------
    M : int
        number of controls to be turned into bytes.

    Returns
    -------
    { number_of_bytes, format_marker, c_type_name} : {int, str, str}
        the byte type for conversion.
    """
    if M < 9:
        return {
            'number_of_bytes': 1,
            'format_marker': 'B',
            'c_type_name': 'unsigned char'
        }
    if M < 17:
        return {
            'number_of_bytes': 2,
            'format_marker': 'h',
            'c_type_name': 'short'
        }
    if M < 65:
        return {
            'number_of_bytes': 4,
            'format_marker': 'i',
            'c_type_name': 'int'
        }
    if M < 129:
        return {
            'number_of_bytes': 8,
            'format_marker': 'q',
            'c_type_name': 'long long'
        }
    raise BaseException(
        f"M={M} is larger than 128 which is the largest allowed size")


def control_signal_2_byte_stream(control_signal: Iterator[np.ndarray], M: int) -> Generator[bytes, None, None]:
    """Convert a control signal into a byte stream

    Parameters
    ----------
    control_signal : [array_like, shape=(M,)]
        a iterator producing sequences of control signals (binary).
    M : int
        number of control inputs.

    Yields
    ------
    bytes
        a binary string representing the control signal.

    """
    format = number_of_bytes_selector(M)
    format_marker = format['format_marker']
    for s in control_signal:
        sum = 0
        for m in range(M):
            # Construct integer by bit shifting
            if s[m] > 0:
                sum += 1 << m
        yield struct.pack(format_marker, sum)


def byte_stream_2_control_signal(byte_stream: Iterator[bytes], M: int) -> Generator[np.ndarray, None, None]:
    """Convert a byte stream into a control_sequence

    Parameters
    ----------
    byte_stream : binary buffer
        a byte stream iterator
    M : int
        number of control inputs.

    Yields
    -------
    array_like, shape=(M,)
        a control signal sample.

    Example
    -------
    >>> M = 3
    >>> control_signal = np.array([[0, 1, 0], [1, 0, 1],[0, 0, 1]])
    >>> cs =  byte_stream_2_control_signal(control_signal_2_byte_stream(control_signal, M), M)
    >>> next(cs)
    array([0, 1, 0], dtype=int8)
    >>> next(cs)
    array([1, 0, 1], dtype=int8)
    >>> next(cs)
    array([0, 0, 1], dtype=int8)
    """
    format = number_of_bytes_selector(M)
    format_marker = format['format_marker']
    mask = 1
    for bs in byte_stream:
        if (not bs):
            raise StopIteration
        sum = struct.unpack(format_marker, bs)[0]
        s = np.zeros(M, dtype=np.int8)
        for m in range(M):
            # populate s from sum
            s[m] = mask & (sum >> m)
        yield s


def write_byte_stream_to_file(filename: str, iterator: Iterator[bytes]):
    """Write a stream into binary file.

    Parameters
    ----------
    filename : `str`
        filename for output file
    iterator : [bytes]
        an iterator of bytes to write to file
    """
    with open(filename, "wb") as f:
        for word in iterator:
            f.write(word)


def write_byte_stream_to_files(filename: str, iterator: Iterator[bytes], words_per_file: int = 100000):
    """Write a stream into a sequence of binary files of size words_per_file.

    Parameters
    ----------
    filename : `str`
        filename for output file
    iterator : [bytes]
        an iterator of bytes to write to files
    words_per_file: `int`
        number of words to be written per file, defaults to 100000.
        For byte-sized words the default corresponds to 100MB files.
    """
    count = 0
    iteration = 0
    base, ext = os.path.splitext(filename)
    while True:
        name = base + f"_{iteration}" + ext
        with open(name, "wb") as f:
            while (count < words_per_file):
                f.write(next(iterator))
                count += 1
        count = 0
        iteration += 1


def read_byte_stream_from_file(filenames: Union[str, list], M: int) -> Generator[bytes, None, None]:
    """Generate a byte stream iterator from file

    Parameters
    ----------
    filename : `str`
        filename for input file
    M : `int`
        number of controls

    Yields
    ------
    bytes :
        returns bytes
    """
    format = number_of_bytes_selector(M)
    if type(filenames) is str:
        filenames = [filenames]
    for filename in filenames:
        with open(filename, "rb") as f:
            byte = b'0'
            try:
                while byte:
                    byte = f.read(format['number_of_bytes'])
                    yield byte
            except StopIteration:
                pass
    raise StopIteration


def read_byte_stream_from_url(urlstring: Union[str, list], M: int) -> Generator[bytes, None, None]:
    """Generate a byte stream iterator from http request

    Parameters
    ----------
    urlstring : `str`
        url for input file
    M : `int`
        number of controls

    Yields
    ------
    bytes :
        returns bytes
    """
    session = requests.Session()
    format = number_of_bytes_selector(M)
    urls = []
    if type(urlstring) is str:
        urls = [urlstring]
    else:
        urls = urlstring
    for url in urls:
        try:
            request = session.get(url, stream=True)
            for chunk in request.iter_content(chunk_size=format['number_of_bytes'], decode_unicode=False):
                yield chunk
        except StopIteration:
            pass
    raise StopIteration


def random_control_signal(M: int, stop_after_number_of_iterations: int = (1 << 63), random_seed: int = 0) -> Generator[np.ndarray, None, None]:
    """Creates a iterator producing random control signals.

    Parameters
    ----------
    M : `int`
        number of controls
    stop_after_number_of_iterations : `int`, `optional`
        number of iterations until :py:class:`StopIteration` is raised,
        defaults to  :math:`2^{63}`.
    random_seed : `int`, `optional`
        used for setting the random seed, see :py:class:`numpy.random.seed`.

    Yields
    ------
    array_like, shape=(M,)
        a random control signal
    """
    if random_seed:
        np.random.seed(random_seed)
    iteration: int = 0
    while (iteration < stop_after_number_of_iterations):
        iteration += 1
        yield np.random.randint(2, size=M, dtype=np.int8)
    else:
        raise StopIteration


def compute_power_spectral_density(sequence: np.ndarray, nperseg: int = 1 << 14, fs: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """Compute power spectral density of sequence.

    Parameters
    ----------
    sequence : array_like, shape=(L, K)
        determine length of each fft sequence
    fs : `double`, `optional`
        sampling frequency, defaults to 1.0

    Returns
    -------
    ((array_like, shape=(K,)), (array_like, shape=(L, K)))
        frequencies [Hz] and PSD [:math:`V^2/\mathrm{Hz}`] of sequence.
    """
    freq, spectrum = welch(
        sequence,
        window='hanning',
        nperseg=nperseg,
        noverlap=None,
        nfft=None,
        return_onesided=True,
        scaling='density',
        fs=fs
    )
    return (np.asarray(freq), np.asarray(spectrum))


def snr_spectrum_computation(spectrum: np.ndarray,
                             signal_mask: np.ndarray,
                             noise_mask: np.ndarray):
    """Compute snr from spectrum

    Parameters
    ----------
    spectrum: ndarray
        a frequency spectrum
    signal_mask: ndarray
        an array containing the indices corresponding to the inband signal
        components.
    noise_mask: ndarray
        an array containing the indices corresponding to the inband noise.

    Returns
    -------
    float
        the signal-to-noise ratio.
    """
    noise = np.sum(spectrum[noise_mask])
    signal = np.sum(spectrum[signal_mask])
    if (noise > 0):
        return signal/noise
    else:
        return np.inf


def find_sinusoidal(spectrum: np.ndarray, mask_width: np.ndarray):
    """Find the peak in the spectrum and return indexes.

    Parameters
    ----------
    spectrum: ndarray
        a power spectral density or equivalent
    mask_width: `int`
        the width around peak to be considered.

    """
    candidate_peak = np.argmax(np.abs(spectrum))
    return np.arange(candidate_peak - mask_width // 2, candidate_peak + mask_width // 2)


def show_status(iterator, length: int = 1 << 63):
    """Write progress to stdout using :py:mod:`tqdm`.

    Parameters
    ----------
    iterator : a iterator
        a iterator to yield values from.
    length : int
        indicate length of iterator. Also causes a raises a StopIteration
        if iteration exceeds length, defaults to 2^63.
    """
    iterator_with_progress = tqdm(iterator)
    if length < (1 << 63):
        iterator_with_progress.total = length
    for iteration, value in enumerate(iterator_with_progress):
        if not iteration < length:
            raise StopIteration
        yield value


def pickle_dump(object_to_be_pickled, filename: str):
    """Convenience function to pickle an object.

    In principle all class instances in this package
    can be pickled to a file for later use.

    For example this includes:

    - analog signal instances,
    - analog system instances,
    - digital control instance,
    - simulation instances,
    - and digital estimator instances.

    See also
    --------
    :py:func:`cbadc.utilities.pickle_load`

    Parameters
    ----------
    object_to_be_pickled: any
        a pickleable object
    filename: `str`
        the path for it to be stored.
    """
    with open(filename, 'wb') as f:
        pickle.dump(object_to_be_pickled, f, protocol=-1)


def pickle_load(filename: str):
    """Convenience function to load python object from pickled file

    See also
    --------
    :py:func:`cbadc.utilities.pickle_dump`

    Parameters
    ----------
    filename: `str`
        the filename of the pickled file.
    """
    with open(filename, 'rb') as f:
        return pickle.load(f)
