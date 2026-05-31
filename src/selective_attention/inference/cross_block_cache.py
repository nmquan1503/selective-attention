from .ssm_cache import SSMCache
from .selective_attn_cache import SelectiveAttnCache
from .cross_selective_attn_cache import CrossSelectiveAttnCache

class CrossBlockCache:
    def __init__(self):
        self.ssm_cache = SSMCache()
        self.attn_cache = SelectiveAttnCache()
        self.cross_attn_cache = CrossSelectiveAttnCache()