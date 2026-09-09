from ..models import EncodedSegment


class IdentityCodec:
    name = "identity"

    def encode(self, text, config, counter, check_budget):
        return EncodedSegment(text, text)
