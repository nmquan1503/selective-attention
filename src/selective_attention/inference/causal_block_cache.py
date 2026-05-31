from .ssm_cache import SSMCache
from .selective_attn_cache import SelectiveAttnCache

class CausalBlockCache:
    def __init__(self):
        self.ssm_cache = SSMCache()
        self.attn_cache = SelectiveAttnCache()