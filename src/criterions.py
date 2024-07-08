import numpy as np
import torch


class L1AdaptiveThresholdCriterion(torch.nn.Module):
    def forward(self, x, y, adaptive_threshold):
        return torch.clip(torch.abs(x-y) - adaptive_threshold, min=0)


def get_criterion(type: str, beta: float = 0.1):
    if type == "l1":
        return torch.nn.L1Loss(reduction="none")
    elif type == "l2":
        return torch.nn.MSELoss(reduction="none")
    elif type == "huber":
        return torch.nn.SmoothL1Loss(reduction="none", beta=beta)
    elif type == "adaptive":
        return L1AdaptiveThresholdCriterion()
    elif type == "wloss":
        return lambda x, y, mask, border: PatchWiseWasserSteinSinkhorn(x, y, mask, border, 2, 2)
    else:
        raise ValueError(f"Loss criterion '{type}' unknown")


def psnr(x, y, mask=None):
    diff = x-y
    diff = diff.flatten(1)
    y = y.flatten(1)
    if mask is None:
        v_max = torch.max(y, 1)[0]
        N = diff.shape[1]
    else:
        mask = mask.flatten(1)
        diff *= mask
        v_max = torch.max(mask*y, 1)[0]
        N = torch.sum(mask, dim=1)  # account for empty masks

    psnr = 20*torch.log10(v_max / torch.sqrt(torch.sum(diff**2, 1)/N))
    # sort out infinite values
    return torch.mean(psnr[torch.isfinite(psnr)])


def psnr_np(x, y, mask=None):
    """
        Same behavior as psnr_criterion, just in numpy to evalute batches of volumes
        of the form: NxCxHxW.
        returns:
            psnr: np.float32
    """
    diff = x-y
    diff = diff.reshape((diff.shape[0], -1))
    y = y.reshape((y.shape[0], -1))
    if mask is None:
        v_max = np.max(y, 1)[0]
        N = diff.shape[1]
    else:
        mask = mask.reshape((mask.shape[0], -1))
        diff *= mask
        v_max = np.max(mask*y, axis=1)[0]
        N = np.sum(mask, axis=1)  # account for empty masks

    psnr = 20*np.log10(v_max / np.sqrt(np.sum(diff**2, axis=1)/N))
    # sort out infinite values
    return np.mean(psnr[np.isfinite(psnr)])


def PatchWiseWasserSteinSinkhorn(img1: torch.Tensor,
                                 img2: torch.Tensor,
                                 mask: torch.Tensor,
                                 border: int,
                                 window_size: int,
                                 stride: int,
                                 num_iterations: int = 100,
                                 eps: float = 1.,
                                 shift: bool = True):
    if shift:
        roll1 = np.random.choice((window_size // 2) + 1)
        roll2 = np.random.choice((window_size // 2) + 1)
        roll3 = np.random.choice((window_size // 2) + 1)

        d, h, w = img1.shape[2:]

        img1 = torch.roll(img1[:, :, border:d-border, border:h-border, border:w - border],
                          shifts=(roll1, roll2, roll3), dims=(2, 3, 4))
        img2 = torch.roll(img2[:, :, border:d-border, border:h-border, border:w - border],
                          shifts=(roll1, roll2, roll3), dims=(2, 3, 4))
        mask = torch.roll(mask[:, :, border:d-border, border:h-border, border:w - border],
                          shifts=(roll1, roll2, roll3), dims=(2, 3, 4))
        border // stride

    x = img1.unfold(2, window_size, stride)
    x = x.unfold(3, window_size, stride)
    x = x.unfold(4, window_size, stride)

    y = img2.unfold(2, window_size, stride)
    y = y.unfold(3, window_size, stride)
    y = y.unfold(4, window_size, stride)

    m = mask.unfold(2, window_size, stride)
    m = m.unfold(3, window_size, stride)
    m = m.unfold(4, window_size, stride)

    x = x.reshape(-1, window_size ** 3)
    y = y.reshape(-1, window_size ** 3)
    m = m.reshape(-1, window_size ** 3, 1)

    C_res = torch.abs(x[:, None, :] - y[:, :, None])
    with torch.no_grad():
        Q = C_res.detach()
        eps = torch.max(torch.max(Q, -1, keepdim=True)[0], -2, keepdim=True)[0] * 10
        Q = torch.exp(-Q / eps)
        b = torch.ones((Q.shape[0], window_size ** 3, 1)).type_as(Q)
        T = 1
        for _ in range(num_iterations):
            K = T * Q
            a = 1 / window_size**3 / torch.matmul(K, b)
            b = 1 / window_size**3 / torch.matmul(K.transpose(2, 1), a)
            T = a * K * b.transpose(2, 1)
        # For numerical stability reasons.
        T = torch.nan_to_num(T)

    return (T * C_res * m).sum((1, 2))
