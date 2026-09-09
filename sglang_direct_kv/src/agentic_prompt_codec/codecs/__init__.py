from .dictionary import DictionaryCodec
from .identity import IdentityCodec
from .relations import RelationsCodec

BUILTINS = {c.name: c for c in (IdentityCodec, DictionaryCodec, RelationsCodec)}
