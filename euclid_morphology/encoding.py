"""
Functions to encode and decode between Zoobot's vector output and a ASCII-compatible string

Zoobot vector output is N (~40) 4 byte floating point numbers.

Convert to bytes with np.tobytes
https://numpy.org/doc/stable/reference/generated/numpy.ndarray.tobytes.html
Then encode those bytes as ASCII

Base64 will convert from bytes to base64-encoded bytes radix-64 encoding.
# https://developer.mozilla.org/en-US/docs/Glossary/Base64

This can then be decoded to ASCII characters with .decode('utf-8')

"""
import base64  # I love python sometimes...

import numpy as np

def np_to_ascii(x: np.ndarray, precision=np.float16) -> str:
    assert x.ndim == 1
    as_bytes = x.astype(precision).tobytes()  # C-order by default
    as_encoded_bytes = base64.b64encode(as_bytes)
    # as_encoded_bytes = base64.a85encode(as_bytes)
    as_ascii = as_encoded_bytes.decode('utf-8')
    return as_ascii


def ascii_to_np(x: str, precision=np.float16) -> np.ndarray:
    as_bytes = base64.b64decode(x.encode('utf-8'))
    as_np = np.frombuffer(as_bytes, dtype=precision).astype(np.float32)
    assert as_np.ndim == 1
    return as_np


if __name__ == '__main__':

    x = np.random.rand(34) * 100 + 1  # 1-101 range matching Zoobot
    # [42.49838467 28.93460703 54.83796706, ...]
    print(x[:3])

    as_str = np_to_ascii(x)
    # UFE8T9tSFFWsVG1TR1DXTklU2UsMViVUQFJoUX9QcE8FU+1SBlQWVPdVi1BcUFhVwlXWUvRVe1DsQ/tVE0/iUgVWhFQ=
    print(as_str)
    # for 34 numbers, 92 characters = 92 bytes
    # https://mothereff.in/byte-counter

    as_np = ascii_to_np(as_str)
    # [42.5     28.9375  54.84375, ...]
    print(as_np[:3])
    # slight shifts (second decimal place) due to half-precision encoding
    # can get back essentially exactly the same with single precision encoding, depending on space requirement
