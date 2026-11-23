"""Counter - wrapped integer with non-mutating arithmetic and optional add hook."""

from typing import Any, Callable, Optional


###############################################################################
class Counter:
    """Counter - wrapped integer"""

    def __init__(
        self,
        name: str,
        value: int = 0,
        on_add: Optional[Callable[[int], int]] = None,
    ) -> None:
        self._name = name
        self._value = value
        self._on_add = on_add

    def get(self) -> int:
        """Get the value"""
        return self._value

    def set(self, value: int) -> int:
        """Set the value"""
        self._value = value
        return self._value

    def add(self, mod: int) -> None:
        """Add to the value (applying optional positive-add interceptor, e.g. -$1 token)"""
        if mod > 0 and self._on_add is not None:
            mod = self._on_add(mod)
        self._value += mod

    @staticmethod
    def _coerce(obj: Any, op_name: str) -> int:
        if isinstance(obj, int):
            return obj
        if hasattr(obj, "get"):
            return int(obj.get())
        raise NotImplementedError(f"Counter {op_name}({obj=}) {type(obj)}")

    def __add__(self, obj: Any) -> "Counter":
        """Non-mutating addition returning a new Counter"""
        delta = self._coerce(obj, "__add__")
        return Counter(self._name, self._value + delta)

    def __iadd__(self, obj: Any) -> "Counter":
        """In-place addition (+=), firing on_add interceptor for positive gains"""
        delta = self._coerce(obj, "__iadd__")
        self.add(delta)
        return self

    def __sub__(self, obj: Any) -> "Counter":
        """Non-mutating subtraction returning a new Counter"""
        delta = self._coerce(obj, "__sub__")
        return Counter(self._name, self._value - delta)

    def __isub__(self, obj: Any) -> "Counter":
        """In-place subtraction (-=)"""
        delta = self._coerce(obj, "__isub__")
        self._value -= delta
        return self

    def __int__(self) -> int:
        return self._value

    def __repr__(self) -> str:
        return f"<{self._name}={self._value}>"

    def __bool__(self) -> bool:
        return self._value != 0

    def __eq__(self, obj: Any) -> bool:
        if isinstance(obj, int):
            return self._value == obj
        if hasattr(obj, "get"):
            return self._value == obj.get()
        return NotImplemented

    def __lt__(self, obj: Any) -> bool:
        return self._value < self._coerce(obj, "__lt__")

    def __le__(self, obj: Any) -> bool:
        return self._value <= self._coerce(obj, "__le__")

    def __gt__(self, obj: Any) -> bool:
        return self._value > self._coerce(obj, "__gt__")

    def __ge__(self, obj: Any) -> bool:
        return self._value >= self._coerce(obj, "__ge__")


# EOF
