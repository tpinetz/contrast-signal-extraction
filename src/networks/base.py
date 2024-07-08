import abc
import logging
import torch


class Model(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

    @abc.abstractmethod
    def forward(self, x):
        raise NotImplementedError

    def to_tensor(x, device):
        return torch.from_numpy(x).permute(3, 0, 1, 2).contiguous().to(device)[None, ...]

    def inference(self, input, device, base_index=1):
        logging.info('Converting numpy arrays to torch.')
        x = Model.to_tensor(input['data'], device)
        condition = torch.from_numpy(input['condition']).to(device)[None, ...]
        mean = Model.to_tensor(input['mean'], device)
        std = Model.to_tensor(input['std'], device)

        logging.info('Computing prediction.')
        with torch.no_grad():
            diff = self.forward(
                x,
                condition,
                mean=mean,
                std=std
            )[0, 0]
            pred = x[0, base_index] + diff.clamp(min=0)

        return pred.cpu().numpy(), diff.cpu().numpy()
