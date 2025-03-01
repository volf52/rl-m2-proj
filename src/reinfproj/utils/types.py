from os import PathLike
from typing import Literal
import numpy as np

UIntArr = np.ndarray[tuple[int, ...], np.dtype[np.uint16]]
IntArr = np.ndarray[tuple[int, ...], np.dtype[np.int32]]
FloatArr = np.ndarray[tuple[int, ...], np.dtype[np.float32]]

StrPath = str | PathLike[str]
StrOrBytesPath = str | bytes | PathLike[str] | PathLike[bytes]

RenderMode = Literal["human"] | Literal["rgb_array"] | None
