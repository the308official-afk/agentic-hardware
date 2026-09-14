from __future__ import annotations

from .identity import IdentityCodec
from .relations import AgentTraceRelationsCodec

BUILTINS = {codec.name: codec for codec in (IdentityCodec, AgentTraceRelationsCodec)}
