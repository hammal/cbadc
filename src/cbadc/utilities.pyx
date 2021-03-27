"""This module contains various helpful functions to accomondate 
the cbadc toolbox.
"""
import struct
import numpy as np

def number_of_bytes_selector(int M):
    """A helper function for selecting
    the right bytes settings.

    Parameters
    ----------
    M : int
        number of controls to be turned into bytes.

    Returns
    -------
    { number_of_bytes : int, format_marker : str, c_type_name : str }
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
            sum += (s[m] > 0) << m
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
        sum = struct.unpack(format_marker, bs)[0]
        s = np.zeros(M, dtype=np.int8)
        for m in range(M):
            # populate s from sum
            s[m] = mask & (sum >> m)
        yield s
        
        