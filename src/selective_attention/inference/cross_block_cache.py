from .ssm_cache import SSMCache
from .multilevel_conv1d_cache import MultiLevelConv1DCache
from .selective_attn_cache import SelectiveAttnCache
from .cross_selective_attn_cache import CrossSelectiveAttnCache

class CrossBlockCache:
    def __init__(self):
        self.ssm_cache = SSMCache()
        self.mlconv_cache = MultiLevelConv1DCache()
        self.attn_cache = SelectiveAttnCache()
        self.cross_attn_cache = CrossSelectiveAttnCache()