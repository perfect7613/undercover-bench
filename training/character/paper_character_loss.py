"""Open Character Training DPO + chosen NLL + sampled squared-log-ratio penalty.

Matches the reductions in maiush/OpenRLHF at
eaf40e10e0471a9e50d33697bcef15f7b0a32b05, dpo_trainer.py::_get_batch_kl.
The squared penalty is the released implementation's KL approximation, not an
exact vocabulary KL. DPO core comes from Tinker's cookbook.
"""
import torch
from tinker_cookbook.preference.train_dpo import compute_dpo_loss


def character_loss(policy, reference, masks, beta=.1, nll_coef=.1, kl_coef=.001):
    if len(policy) == 0 or len(policy) % 2 or not len(policy) == len(reference) == len(masks):
        raise ValueError("expected interleaved chosen/rejected pairs")
    sums, ref_sums, means, squared = [], [], [], []
    for lp, ref, mask in zip(policy, reference, masks, strict=True):
        lp = lp.float()
        ref, mask = ref.to(lp).detach(), mask.to(lp)
        if lp.shape != ref.shape or lp.shape != mask.shape or mask.sum() <= 0:
            raise ValueError("unaligned log probabilities or empty response mask")
        if not torch.all((mask == 0) | (mask == 1)):
            raise ValueError("DPO requires binary token masks, not normalized SFT weights")
        # Select rather than multiply so unused/prompt positions cannot leak NaNs.
        active = mask.bool()
        lp, ref = lp[active], ref[active]
        if not torch.isfinite(lp).all() or not torch.isfinite(ref).all():
            raise ValueError("non-finite response log probabilities")
        sums.append(lp.sum())
        ref_sums.append(ref.sum())
        means.append(lp.mean())
        squared.append(((lp-ref)**2).mean())
    loss, metrics = compute_dpo_loss(sums[::2], sums[1::2], ref_sums[::2], ref_sums[1::2], beta)
    nll = -torch.stack(means[::2]).mean()
    kl = torch.stack(squared).mean()
    total = loss + nll_coef*nll + kl_coef*kl
    metrics.update(chosen_nll=nll.item(), squared_logratio_penalty=kl.item(), total_loss=total.item())
    return total, metrics
