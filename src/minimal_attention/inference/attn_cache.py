import torch

class AttnCache:
    def __init__(self):
        self.k_rot = None          # (B, num_heads, L, head_dim)
        self.v = None              # (B, num_heads, L, head_dim)
        self.mask = None           # (B, L) bool

    @property
    def seq_len(self):
        return self.k_rot.shape[2] if self.k_rot is not None else 0

    def build(self, k_rot, v, mask):
        self.k_rot = k_rot
        self.v = v
        self.mask = mask

    def update(self, k_new, v_new):
        k_new = k_new.unsqueeze(2)   # (B, nh, 1, hd)
        v_new = v_new.unsqueeze(2)
        m_new = torch.ones(k_new.shape[0], 1, dtype=torch.bool, device=k_new.device)
        if self.k_rot is None:
            self.k_rot, self.v, self.mask = k_new, v_new, m_new
        else:
            self.k_rot = torch.cat([self.k_rot, k_new], dim=2)
            self.v = torch.cat([self.v, v_new], dim=2)
            self.mask = torch.cat([self.mask, m_new], dim=1)