"""
RLBench policy transforms for RICL.

Maps RLBench dataset format (from RiclRLBenchDataset) to the model input format
expected by Pi0-FAST-RICL. Similar to RiclDroidInputs but adapted for RLBench's
camera naming and action dimensions.

Camera mapping:
  top_image (front_rgb)     → base_0_rgb
  right_image (overhead_rgb) → base_1_rgb
  wrist_image (wrist_rgb)   → left_wrist_0_rgb
"""

import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class RiclRLBenchInputs(transforms.DataTransformFn):
    """Convert RLBench dataset format → RICL model input format.

    Handles both retrieved observations and query observation.
    Image keys: top_image→base_0_rgb, right_image→base_1_rgb, wrist→left_wrist_0_rgb.
    """

    action_dim: int  # 7 for RLBench
    num_retrieved_observations: int  # default 4

    def __call__(self, data: dict) -> dict:
        all_prefix = [f"retrieved_{i}_" for i in range(self.num_retrieved_observations)] + ["query_"]

        inputs_dicts = [
            {
                f"{prefix}state": data[f"{prefix}state"],
                f"{prefix}image": {
                    "base_0_rgb": _parse_image(data[f"{prefix}top_image"]),
                    "base_1_rgb": _parse_image(data[f"{prefix}right_image"]),
                    "left_wrist_0_rgb": _parse_image(data[f"{prefix}wrist_image"]),
                },
                f"{prefix}image_mask": {
                    "base_0_rgb": np.True_,
                    "base_1_rgb": np.True_,
                    "left_wrist_0_rgb": np.True_,
                },
            }
            for prefix in all_prefix
        ]

        # Collapse to single dict
        inputs = {k: v for d in inputs_dicts for k, v in d.items()}

        # Include retrieved actions and query actions
        for prefix in all_prefix[:-1]:
            inputs[f"{prefix}actions"] = data[f"{prefix}actions"]
        if "query_actions" in data:
            inputs["query_actions"] = data["query_actions"]

        # Prompts
        for prefix in all_prefix:
            inputs[f"{prefix}prompt"] = data[f"{prefix}prompt"]

        # Action interpolation distances
        if "exp_lamda_distances" in data:
            inputs["exp_lamda_distances"] = data["exp_lamda_distances"]

        # Inference-time flag
        if "inference_time" in data:
            inputs["inference_time"] = data["inference_time"]

        return inputs


@dataclasses.dataclass(frozen=True)
class RiclRLBenchOutputs(transforms.DataTransformFn):
    """Convert model output back to RLBench action format.

    Crops padded actions to 7 dims (dx, dy, dz, drx, dry, drz, gripper).
    """

    def __call__(self, data: dict) -> dict:
        return {"query_actions": np.asarray(data["query_actions"][:, :7])}
