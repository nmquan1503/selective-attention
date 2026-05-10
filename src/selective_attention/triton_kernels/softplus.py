import triton
import triton.language as tl
from packaging import version

if version.parse(triton.__version__) >= version.parse("3.0.0"):
    @triton.jit
    def softplus(x):
        return tl.math.log(tl.math.exp(x) + 1)
else:
    @triton.jit
    def softplus(x):
        return tl.math.log1p(tl.exp(x))