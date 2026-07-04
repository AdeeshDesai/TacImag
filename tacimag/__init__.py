"""TacImag: Imagining the Sense of Touch for Robotic Manipulation.

Two-stage pipeline on top of ManiFeel (TacSL / IsaacGym):
  Stage 1 (tacimag.imagination) — a conditional diffusion model imagines the
      tactile signal (force field or tactile RGB) from vision + proprioception.
  Stage 2 (tacimag.policy)      — a Diffusion Policy conditions on the imagined
      tactile, so no tactile sensor is needed at deployment.
"""

__version__ = "0.1.0"
