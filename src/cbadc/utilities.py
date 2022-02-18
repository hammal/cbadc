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
import scipy.io.wavfile
import numpy.typing as npt
import scipy.signal
import logging

logger = logging.getLogger(__name__)


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
            "number_of_bytes": 1,
            "format_marker": "B",
            "c_type_name": "unsigned char",
        }
    if M < 17:
        return {"number_of_bytes": 2, "format_marker": "h", "c_type_name": "short"}
    if M < 65:
        return {"number_of_bytes": 4, "format_marker": "i", "c_type_name": "int"}
    if M < 129:
        return {"number_of_bytes": 8, "format_marker": "q", "c_type_name": "long long"}
    raise Exception(f"M={M} is larger than 128 which is the largest allowed size")


def control_signal_2_byte_stream(
    control_signal: Iterator[np.ndarray], M: int
) -> Generator[bytes, None, None]:
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
    format_marker = format["format_marker"]
    for s in control_signal:
        sum = 0
        for m in range(M):
            # Construct integer by bit shifting
            if s[m] > 0:
                sum += 1 << m
        yield struct.pack(format_marker, sum)


def byte_stream_2_control_signal(
    byte_stream: Iterator[bytes], M: int
) -> Generator[np.ndarray, None, None]:
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
    format_marker = format["format_marker"]
    mask = 1
    for bs in byte_stream:
        if not bs:
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


def write_byte_stream_to_files(
    filename: str, iterator: Iterator[bytes], words_per_file: int = 100000
):
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
            while count < words_per_file:
                f.write(next(iterator))
                count += 1
        count = 0
        iteration += 1


def read_byte_stream_from_file(
    filenames: Union[str, list], M: int
) -> Generator[bytes, None, None]:
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
            byte = b"0"
            try:
                while byte:
                    byte = f.read(format["number_of_bytes"])
                    yield byte
            except StopIteration:
                pass
    raise StopIteration


def read_byte_stream_from_url(
    urlstring: Union[str, list], M: int
) -> Generator[bytes, None, None]:
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
            for chunk in request.iter_content(
                chunk_size=format["number_of_bytes"], decode_unicode=False
            ):
                yield chunk
        except StopIteration:
            pass
    raise StopIteration


def csv_2_control_signal(
    filename: str, M: int, msb2lsb: bool = False, separator: str = ','
):
    """Creates an iterator that reads a control sequence from a CSV file.

    Parameters
    ----------
    filename : `str`
        filename for input file
    M : `int`
        number of controls
    msb2lsb : `bool`
        bit order of input file.
        Default is least significant bit (LSB) to most significant bit (MSB) from left to right.
    separator : `str`
        separator used in the input file. Default is ` , ` (comma).

    Yields
    ------
    array_like, shape=(M,)
        a control signal sample

    Example
    -------

    .. code block::

        # create a csv file with four bits
        with open('test.csv','w') as f:
            f.write("0, 0, 0, 0\n0, 0, 0, 1\n0, 0, 1, 1\n0, 1, 1, 1")
        # read control signal from the file just created
        for s in csv_2_control_signal('test.csv', 4, msb2lsb=True):
            print(s)
        [0 0 0 0]
        [1 0 0 0]
        [1 1 0 0]
        [1 1 1 0]

    Raises
    ------
    `RuntimeError`
        for lines with number of entries different from M.

    """
    with open(filename, 'r') as read_obj:
        for line in read_obj:
            s = np.fromstring(line, dtype=np.int8, sep=separator)
            if s.size != M:
                raise RuntimeError(
                    "The number of entries in the current line is not equal to M"
                )
            if msb2lsb:
                s = s[::-1]
            yield s


def random_control_signal(
    M: int, stop_after_number_of_iterations: int = (1 << 63), random_seed: int = 0
) -> Generator[np.ndarray, None, None]:
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
    while iteration < stop_after_number_of_iterations:
        iteration += 1
        yield np.random.randint(2, size=M, dtype=np.int8)
    else:
        raise StopIteration


def compute_power_spectral_density(
    sequence: np.ndarray, nperseg: int = 1 << 14, fs: float = 1.0
) -> Tuple[np.ndarray, np.ndarray]:
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
    logger.debug("Computing power spectral density.")
    nperseg = min(nperseg, sequence.size)
    freq, spectrum = welch(
        sequence,
        # window='hanning',
        window="blackman",
        nperseg=nperseg,
        noverlap=None,
        nfft=None,
        return_onesided=True,
        scaling="density",
        fs=fs,
    )
    return (np.asarray(freq), np.asarray(spectrum))


def snr_spectrum_computation(
    spectrum: np.ndarray, signal_mask: np.ndarray, noise_mask: np.ndarray
):
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
    if noise > 0:
        return signal / noise
    else:
        return np.inf


def snr_spectrum_computation_extended(
    spectrum: np.ndarray,
    signal_mask: np.ndarray,
    noise_mask: np.ndarray,
    harmonics_mask: np.ndarray = np.array([0]),
    fs: float = 1,
):
    """Extended spectrum computations

    Parameters
    ----------
    spectrum: ndarray
        a frequency spectrum
    signal_mask: ndarray
        an array containing the indices corresponding to the inband signal
        components.
    noise_mask: ndarray
        an array containing the indices corresponding to the inband noise.
    fs: `float`
        the sampling frequency of spectrum [Hz].

    Returns
    -------
    {
        noise_rms: `float`
        signal_rms: `float`
        snr: `float`
        window: `str`
        CG: `float`
        NG: `float`
        thd: `float`
        thd_n: `float`
        sinad: `float`
    }
        Python dict containing relevant spectrum information.
    """
    win = "blackman"
    CG = 1.0
    NG = 1.0
    N = spectrum.size
    f_bin = fs / N
    if win == "blackman":
        window = scipy.signal.windows.blackman(N)
        CG = np.mean(window)
        NG = np.sum(window ** 2) / N
    if win == "hanning":
        window = scipy.signal.windows.blackman(N)
        CG = np.mean(window)
        NG = np.sum(window ** 2) / N

    noise = np.sum(spectrum[noise_mask])
    signal = np.sum(spectrum[signal_mask])
    harmonics = np.sum(spectrum[harmonics_mask])

    snr = signal / noise
    sinad = (signal + noise + harmonics) / (noise + harmonics)
    thd = np.sqrt(harmonics / signal)
    thd_n = np.sqrt((harmonics + noise) / signal)
    signal_rms = np.sqrt(signal * NG * f_bin / (CG ** 2))
    noise_rms = np.sqrt(noise * f_bin)

    return {
        "noise_rms": noise_rms,
        "signal_rms": signal_rms,
        "snr": snr,
        "window": window,
        "CG": CG,
        "NG": NG,
        "thd": thd,
        "thd_n": thd_n,
        "sinad": sinad,
    }


def find_n_sinusoidals(
    spectrum: np.ndarray, n: int, mask_width: int, min_peak_dist: int = None
):
    """Find the n largest peaks in the spectrum and return indexes.

    Parameters
    ----------
    spectrum: ndarray
        a power spectral density or equivalent.
    n: `int`
        number of peaks to be considered.
    mask_width: `int`
        the width of the returned mask for each peak considered.
    min_peak_dist: `int`
        the minimum distance between two peak (such that they are still considered separate peaks).
        Defaults to mask_width//2.
    """
    if min_peak_dist is None:
        min_peak_dist = mask_width // 2
    indices = np.array([], dtype=np.int64)
    remaining_spectrum = np.array(spectrum)

    for i in range(n):
        # get peak
        candidate_peak = np.argmax(np.abs(remaining_spectrum))

        # add neighbouring indices to result
        indices_masked = np.arange(
            candidate_peak - mask_width // 2, candidate_peak + mask_width // 2
        )
        indices_masked = np.clip(
            indices_masked, 0, spectrum.size - 1
        )  # clip invalid indices
        indices = np.concatenate((indices, indices_masked))  # append to rest

        # remove peak from spectrum
        remove_from_spectrum = np.arange(
            candidate_peak - min_peak_dist, candidate_peak + mask_width
        )
        remove_from_spectrum = np.clip(
            remove_from_spectrum, 0, spectrum.size - 1
        )  # clip invalid indices
        remaining_spectrum[remove_from_spectrum] = 0

    return np.unique(indices)  # sort and remove duplicates


def find_sinusoidal(spectrum: np.ndarray, mask_width: np.ndarray):
    """Find the peak in the spectrum and return indexes.
    Equivalent to `find_n_sinusoidals(spectrum, 1, mask_width)`.

    Parameters
    ----------
    spectrum: ndarray
        a power spectral density or equivalent
    mask_width: `int`
        the width around peak to be considered.

    """
    return find_n_sinusoidals(spectrum, 1, mask_width)


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
    logger.info(f"Dumping object to file {filename}.")
    with open(filename, "wb") as f:
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
    logger.info(f"Loading object from file {filename}.")
    with open(filename, "rb") as f:
        return pickle.load(f)


def iterator_to_numpy_array(iterator: Iterator[bytes], size: int, L: int = 1):
    """Convert an iterator into a numpy array

    Parameters
    ----------
    iterator:
        a iterator with data points to fill up numpy array.
    size: `int`
        length of numpy array.
    L: `int`
        dimenson of the datapoints.

    Returns
    -------
    array_like, shape=(size, L)

    """
    if size < 1 or L < 1:
        raise Exception("Both size and L must be positive integers.")
    data = np.zeros((size, L), dtype=np.double)
    for index in range(size):
        data[index, :] = next(iterator)
    return data


def write_wave(filename: str, sample_rate: int, data: np.ndarray):
    """Create wave file from data array.

    This is a wrapper function for :py:func:`scipy.io.wavefile.write`.

    Parameters
    ----------
    filename: `str`
        name of the file to be generated
    sample_rate: `int`
        the sample rate in samples/second
    data: `array_like, shape=(number of samples, number of channels)
        the data array to be encoded as wave file

    """
    logger.info(f"Writing data array to wave file: {filename}.")
    scipy.io.wavfile.write(filename, sample_rate, data)


class FixedPoint:
    """Fixed point description class.

    Parameters
    ----------
    number_of_bits: `int`
        number of bits used including sign bit.
    max: `float`
        the largest (or smallest) floating number to be represented.

    """

    def __init__(self, number_of_bits: int, max: float):
        self.__number_of_bits = number_of_bits
        self.__max = max
        self.__int_max = 1 << (self.__number_of_bits - 1)
        self.__scale = self.__int_max / self.__max
        self.__min = self.fixed_to_float(1)

    def float_to_fixed(self, value: float) -> int:
        """Convert floating point to fixed point number.

        Parameters
        ----------
        value: `float`
            number to be converted.

        Returns
        -------
        `int`
            fixed point representation

        """
        if abs(value) > self.__max:
            raise ArithmeticError("abs(Value) exceeds max value.")
        return int(value * self.__scale)

    def fixed_to_float(self, value: int) -> float:
        """Convert fixed point to floating point number.

        Parameters
        ----------
        value: `int`
            number to be converted.

        Returns
        -------
        `float`
            the floating point representation.

        """
        if abs(value) > self.__int_max:
            raise ArithmeticError("abs(Value) exceeds max integer value")
        return float(value / self.__scale)

    def __str__(self):
        return f"""
        number of bits = {self.__number_of_bits} including sign bit,
        max float value = {self.__max},
        and min float value = {self.__min}
        """

    def max(self):
        """Largest floating point.

        Returns
        -------
        `float`
            largest floating point representation.

        """
        return self.__max

    def min(self):
        """Smallest floating point

        Returns
        --------
        `float`
            smallest floating point representation.

        """
        return self.__min

    def max_int(self):
        """max integer value.

        Returns
        -------
        `int`
            largest fixed point integer representation.

        """
        return self.__int_max
