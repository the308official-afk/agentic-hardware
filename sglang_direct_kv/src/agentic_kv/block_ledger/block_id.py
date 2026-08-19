from __future__ import annotations

import hashlib
from dataclasses import dataclass


@dataclass(frozen=True)
class BlockIdentity:
    session_id: str
    token_start: int | None
    token_end: int | None
    token_count: int
    node_id: str = ""
    host_index_signature: str = ""
    device_index_signature: str = ""

    def stable_key(self) -> str:
        node = self.node_id or "no-node"
        host_sig = self.host_index_signature or "no-host"
        device_sig = self.device_index_signature or "no-device"
        start = "na" if self.token_start is None else str(self.token_start)
        end = "na" if self.token_end is None else str(self.token_end)
        payload = f"{self.session_id}|{node}|{host_sig}|{device_sig}|{start}|{end}|{self.token_count}"
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


def make_block_id(identity: BlockIdentity) -> str:
    return f"kvblk_{identity.stable_key()}"


def ranges_overlap(
    a_start: int | None,
    a_end: int | None,
    b_start: int | None,
    b_end: int | None,
) -> int:
    if None in (a_start, a_end, b_start, b_end):
        return 0
    return max(0, min(int(a_end), int(b_end)) - max(int(a_start), int(b_start)) + 1)


def nearby_range_score(
    a_start: int | None,
    a_end: int | None,
    a_count: int,
    b_start: int | None,
    b_end: int | None,
    b_count: int,
) -> float:
    if a_count <= 0 or b_count <= 0:
        return 0.0
    overlap = ranges_overlap(a_start, a_end, b_start, b_end)
    if overlap:
        return overlap / max(a_count, b_count)
    if None in (a_start, a_end, b_start, b_end):
        return 0.0
    start_delta = abs(int(a_start) - int(b_start))
    end_delta = abs(int(a_end) - int(b_end))
    count_delta = abs(int(a_count) - int(b_count))
    if count_delta <= 2 and start_delta <= 4 and end_delta <= 4:
        return 0.95
    if count_delta <= 8 and start_delta <= 16 and end_delta <= 16:
        return 0.80
    return 0.0
