from .ssm_cache import SSMCache
from .min_attn_cache import MinAttnCache

class CausalBlockCache:
    def __init__(self):
        self.ssm_cache = SSMCache()
        self.attn_cache = MinAttnCache()