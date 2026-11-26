from .ssm_cache import SSMCache
from .min_attn_cache import MinAttnCache
from .cross_min_attn_cache import CrossMinAttnCache
from .attn_cache import AttnCache
from .cross_attn_cache import CrossAttnCache

class CrossBlockCache:
    def __init__(self):
        self.ssm_cache = SSMCache()
        self.attn_cache = AttnCache()
        self.cross_attn_cache = CrossAttnCache()