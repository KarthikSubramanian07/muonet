"""Reconstruction: scattering data in, a 3D density volume out.

Two classical baselines and one learned model, all producing the same thing:
a voxel grid of scattering density on a shared [0, 1] scale.

  - poca: Point of Closest Approach, the fast geometric baseline every result
    is measured against (Schultz et al., NIM A 519:687, 2004).
  - mlem: Maximum Likelihood Expectation Maximization, the stronger classical
    competitor (Schultz et al., IEEE TIP 16:1985, 2007).
  - learned: a trajectory MLP followed by a compact 3D U-Net (architecture
    lineage: mu-Net, arXiv:2312.17265). Imported lazily so classical
    reconstruction never requires Torch.
"""

from .poca import poca_points, poca_reconstruct

__all__ = ["poca_reconstruct", "poca_points"]
