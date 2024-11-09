from collections import namedtuple

from rayuela.base.symbol import Sym


class Production(namedtuple("Production", "head, body")):
    def __repr__(self):
        return (
            str(self.head)
            + " → "
            + " ".join(
                map(
                    lambda x: str(repr(x))[1:-1] if isinstance(x, Sym) else str(x),
                    self.body,
                )
            )
        )
