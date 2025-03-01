from enum import IntEnum
from typing import Protocol, Self


class TAction(Protocol):
    def to_int(self) -> int: ...
    @classmethod
    def from_int(cls, i: int) -> Self: ...


class EnumAction(IntEnum):
    def to_int(self) -> int:
        return self.value

    @classmethod
    def from_int(cls, i: int) -> Self:
        return cls(i)


class TState(Protocol):
    @classmethod
    def get_shape(cls, n_actions: int) -> tuple[int, ...]: ...
    def get_slice(
        self, action: TAction | None
    ) -> tuple[int, ...] | tuple[slice, ...] | tuple[int | slice, ...]: ...
