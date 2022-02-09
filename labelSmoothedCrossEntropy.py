from torch import nn


class LabelSmoothedCrossEntropyCriterion(nn.Module):
    """Implement label smoothing."""

    def __init__(self, eps=0.1, reduction=True):
        super(LabelSmoothedCrossEntropyCriterion, self).__init__()
        self.eps = eps
        self.reduction = reduction

    def forward(self, lprobs, target, pad_mask):
        lprobs = lprobs.view(-1, lprobs.size(-1))
        target = target.view(-1, 1)
        pad_mask = pad_mask.view(-1, 1)
        if self.reduction:
            nll_loss = -lprobs.gather(dim=-1, index=target).masked_fill_(pad_mask == 0, 0.0)
            nll_loss = nll_loss.sum()
            smooth_loss = -lprobs.sum(dim=-1, keepdim=True).masked_fill_(pad_mask == 0, 0.0)
            smooth_loss = smooth_loss.sum()
        else:
            nll_loss = -lprobs.gather(dim=-1, index=target)[pad_mask]
            smooth_loss = -lprobs.sum(dim=-1, keepdim=True)[pad_mask]
        eps_i = self.eps / lprobs.size(-1)
        loss = (1. - self.eps) * nll_loss + eps_i * smooth_loss
        return loss, pad_mask.sum()
