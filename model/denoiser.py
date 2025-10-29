import torch
from flow_matching.utils import ModelWrapper

class ProbabilityDenoiser(ModelWrapper):
    def forward(self, x: torch.Tensor, t: torch.Tensor, **kwargs) -> torch.Tensor:
        label = None
        if "label" in kwargs:
            label = kwargs["label"]
        elif "extras" in kwargs and isinstance(kwargs["extras"], dict):
            label = kwargs["extras"].get("label")
        if label is None:
            raise RuntimeError("No `label` provided to ProbabilityDenoiser")

        logits = self.model(x, t, label)
        return torch.softmax(logits.float(), dim=-1)

class CFGProbabilityDenoiser(ProbabilityDenoiser):
    def __init__(self, model, guidance_scale: float, null_label: int = 2):
        super().__init__(model)
        self.guidance_scale = guidance_scale
        self.null_label = null_label

    def forward(self, x: torch.Tensor, t: torch.Tensor, **kwargs) -> torch.Tensor:
        label = kwargs.get("label")
        if label is None and isinstance(kwargs.get("extras"), dict):
            label = kwargs["extras"].get("label")
        if label is None:
            raise RuntimeError("No label provided for CFG denoiser")

        logits_cond   = self.model(x, t, label)
        null_label    = torch.full_like(label, self.null_label)
        logits_uncond = self.model(x, t, null_label)

        w = self.guidance_scale
        logits_cfg = logits_uncond + w * (logits_cond - logits_uncond)

        return torch.softmax(logits_cfg.float(), dim=-1)
