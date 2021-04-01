"""This module contains various helpful functions to accomondate 
the cbadc toolbox.
"""
import struct
import numpy as np
cimport numpy as np
import math
from scipy.signal import welch

def number_of_bytes_selector(int M):
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
        return  {
            'number_of_bytes': 1,
            'format_marker': 'B',
            'c_type_name': 'unsigned char'
        }
    if M < 17:
        return  {
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
    raise f"M={M} is larger than 128 which is the largest allowed size"


def control_signal_2_byte_stream(control_signal, int M):
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

def byte_stream_2_control_signal(byte_stream, int M):
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


def write_byte_stream_to_file(filename, iterator):
    """ Write an stream into binary file.

    Parameters
    ----------
    filename : `str`
        filename for output file
    iterator : [bytes]
        an iterator of bytes to write to stream
    """
    with open(filename, "wb") as f:
        for word in iterator:
            f.write(word)
        
def read_byte_stream_from_file(filename, M):
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
    with open(filename, "rb") as f:
        byte = b'0'
        while byte:
            byte = f.read(format['number_of_bytes'])
            yield byte  

def random_control_signal(int M, int stop_after_number_of_iterations=(1 << 63), int random_seed=0):
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
    iteration = 0
    while (iteration < stop_after_number_of_iterations):
        iteration += 1
        yield np.random.randint(2, size=M, dtype = np.int8)
    else:
        raise StopIteration


def compute_power_spectral_density(sequence, nperseg = 1 << 14):
    """Compute power spectral density of sequence.

    Parameters
    ----------
    sequence : array_like, shape=(L, K)

    Returns
    -------
    ((array_like, shape=(K,)), (array_like, shape=(L, K)))
        frequenceis [Hz] and PSD [:math:`V^2/\mathrm{Hz}`] of sequence.
    """
    freq, spectrum = welch(
        sequence, 
        window='hanning',
        nperseg=nperseg, 
        noverlap=None, 
        nfft=None, 
        return_onesided=True, 
        scaling='density' 
        )
    return (freq, spectrum)
