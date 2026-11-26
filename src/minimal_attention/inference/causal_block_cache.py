from .ssm_cache import SSMCache
from .attn_cache import AttnCache

class CausalBlockCache:
    def __init__(self):
        self.ssm_cache = SSMCache()
        self.attn_cache = AttnCache()