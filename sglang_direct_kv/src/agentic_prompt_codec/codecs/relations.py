"""An intentionally narrow, lossless relation grammar; no prose inference."""
import re
from ..models import EncodedSegment
from ..validation import editable_parts

FACT = re.compile(r"^(?P<a>(?:[Tt]he )?[A-Za-z]+) is on (?P<b>(?:the )?[A-Za-z]+)\.$")


class RelationsCodec:
    name = "relations_v1"

    def encode(self, text, config, counter, check_budget):
        lines = [(line, editable) for part, editable in editable_parts(text) for line in part.splitlines(keepends=True)]
        mapping = {}
        out = []
        for line, editable in lines:
            check_budget()
            match = FACT.fullmatch(line.rstrip("\r\n"))
            if match and editable:
                original = match.group()
                alias = f"{match['a']} @ {match['b']}."
                if alias not in text:
                    mapping[alias] = original
                    line = line.replace(original, alias, 1)
            out.append(line)
        if not mapping:
            return EncodedSegment(text, text)
        body = "".join(out)
        legend = 'Local shorthand: "A @ B." means "A is on B." This convention applies only to the following text.\nText:\n'
        return EncodedSegment(legend + body, body, tuple(mapping.items()), legend)
