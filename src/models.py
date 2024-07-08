import pytorch_lightning as pl
import torch
from torchvision.utils import make_grid

import criterions
import networks

from typing import List, Tuple


class SmartContrastModel(pl.LightningModule):
    def __init__(self, cfg, fast: bool = False):
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters()

        models = {
            "condunet": networks.CondUNetModel,
            "condgatedunet": networks.CondGatedUNetModel
        }

        self.network = models[self.cfg.model.type](config=self.cfg.model, fast=fast)

        # selected slices for validation
        self._val_slices = [104-32, 218, 142, 87, 57, 203]  # account for center cropping

        if not fast:
            # define the loss criterion
            self.loss_criterion = criterions.get_criterion(
                cfg.train.get("criterion", "l1"),
                beta=cfg.train.get("beta", 0.1)
            )

    def get_loss_masking(
            self,
            grad: torch.Tensor,
            smooth: torch.Tensor,
            mask: torch.Tensor,
            mask_atlas: torch.Tensor,
            p_signal: torch.Tensor = None
    ) -> Tuple[torch.Tensor]:

        loss_small = grad
        loss_small *= mask_atlas

        loss_smooth = smooth
        loss_smooth *= mask_atlas

        loss_signal = p_signal
        loss_signal *= mask_atlas

        # determine metric
        M = mask_atlas + \
            self.cfg.train.w0 * mask + \
            self.cfg.train.w1 * loss_small + \
            self.cfg.train.w2 * loss_smooth + \
            self.cfg.train.w3 * loss_signal

        return M, (loss_small, loss_smooth, loss_signal)

    def compute_loss(
            self,
            x: torch.Tensor,
            y: torch.Tensor,
            p_signal: torch.Tensor,
            M: torch.Tensor
    ) -> torch.Tensor:
        border = self.cfg.train.border
        d, h, w = x.shape[2:]

        if "wloss" in self.cfg.train.criterion:
            loss = self.loss_criterion(x, y, M, border)
            loss_threshold = loss
        else:
            diff = x - y
            loss_threshold = torch.where(p_signal >= self.cfg.train.loss_balancing.eps,
                                         torch.where(diff > 0.,
                                                     torch.Tensor([self.cfg.train.loss_balancing.v[0]]).type_as(x),
                                                     torch.Tensor([self.cfg.train.loss_balancing.v[1]]).type_as(x)),
                                         torch.Tensor([1.]).type_as(x)) * self.loss_criterion(x, y)
            loss = M * loss_threshold
            loss = loss[:, :, border:d - border, border:h - border, border:w - border]

        return loss.mean()

    def sequence_loss(
            self,
            x: List[torch.Tensor],
            y: torch.Tensor,
            p_signal: torch.Tensor,
            M: torch.Tensor,
            gamma: float = 0.8
    ) -> torch.Tensor:

        n_predictions = len(x)
        loss = 0.0

        for i in range(n_predictions):
            i_weight = gamma**(n_predictions - i - 1)
            loss += i_weight * self.compute_loss(x[i], y, p_signal, M)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.network.parameters(), lr=self.cfg.optim.learning_rate)
        scheduler = {
            'scheduler': torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                              milestones=self.cfg.optim.milestones,
                                                              gamma=self.cfg.optim.gamma),
            'name': 'learning_rate',
            'interval': 'step',
            'frequency': 1
        }
        return [optimizer], [scheduler]

    def on_train_start(self):
        self.logger.log_hyperparams(self.hparams, {"hp/val_loss": 0, "hp/val_psnr": 0})

    def training_step(self, batch, batch_idx):
        loss = []
        sample = batch

        x = sample['data']
        condition = sample['condition']
        target = sample['target']

        mean = sample['mean']
        std = sample['std']

        l_mask = sample['mask']
        l_mask_atlas = sample['mask_atlas']
        p_signal = sample["p_signal"]
        grad = sample['grad']
        smooth = sample['smooth']

        y_hat = self.network(x, condition, mean=mean, std=std)

        M, _ = self.get_loss_masking(grad, smooth, l_mask, l_mask_atlas, p_signal)
        if isinstance(y_hat, list):
            loss = self.sequence_loss(y_hat, target, p_signal, M, gamma=self.cfg.train.gamma)
            # visualize only the last prediction
            y_hat = y_hat[-1]
        else:
            loss = self.compute_loss(y_hat, target, p_signal, M)

        with torch.no_grad():
            self.log('psnr/train', criterions.psnr(y_hat, target))
            self.log('psnr-brain/train', criterions.psnr(y_hat, target, mask=l_mask_atlas))

        if batch_idx % self.trainer.accumulate_grad_batches == 0:
            if self.global_step % self.cfg.train.num_log == 0 or self.global_step == self.cfg.optim.num_iter - 1:
                s = 48
                prefix = "train"
                # training set visualization
                self.logger.experiment.add_image(
                    f'{prefix}/0-T1_sub',
                    make_grid(x[:16, 0:1, s, :, :], nrow=4, normalize=True),
                    self.global_step
                )
                self.logger.experiment.add_image(
                    f'{prefix}/1-T1_nativ',
                    make_grid(x[:16, 1:2, s, :, :], nrow=4, normalize=True),
                    self.global_step
                )
                self.logger.experiment.add_image(
                    f'{prefix}/2-T1_low',
                    make_grid(x[:16, 2:3, s, :, :], nrow=4, normalize=True),
                    self.global_step
                )
                if "T2" in self.cfg.data.inputs:
                    self.logger.experiment.add_image(
                        f'{prefix}/3-T2',
                        make_grid(x[:16, 3:4, s, :, :], nrow=4, normalize=True),
                        self.global_step
                    )
                if "Diff" in self.cfg.data.inputs:
                    self.logger.experiment.add_image(
                        f'{prefix}/4-Diff0',
                        make_grid(x[:16, 4:5, s, :, :], nrow=4, normalize=True),
                        self.global_step
                    )
                    self.logger.experiment.add_image(
                        f'{prefix}/5-Diff500',
                        make_grid(x[:16, 5:6, s, :, :], nrow=4, normalize=True),
                        self.global_step
                    )
                    self.logger.experiment.add_image(
                        f'{prefix}/6-Diff1000',
                        make_grid(x[:16, 6:7, s, :, :], nrow=4, normalize=True),
                        self.global_step
                    )
                self.logger.experiment.add_image(
                    f'{prefix}/7-pred',
                    make_grid(y_hat[:16, :, s, :, :], nrow=4, normalize=True),
                    self.global_step
                )
                self.logger.experiment.add_image(
                    f'{prefix}/8-target',
                    make_grid(target[:16, :, s, :, :], nrow=4, normalize=True),
                    self.global_step
                )
                self.logger.experiment.add_image(
                    f'{prefix}/9-mask',
                    make_grid(l_mask[:16, :, s, :, :], nrow=4, normalize=True),
                    self.global_step
                )
                self.logger.experiment.add_image(
                    f'{prefix}/91-mask-atlas',
                    make_grid(l_mask_atlas[:16, :, s, :, :], nrow=4, normalize=True),
                    self.global_step
                )

        self.log('loss/train', loss)
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        sample = batch

        x = sample['data']
        condition = sample['condition']
        target = sample['target']

        mean = sample['mean']
        std = sample['std']

        l_mask = sample['mask']
        l_mask_atlas = sample['mask_atlas']
        p_signal = sample["p_signal"]
        grad = sample['grad']
        smooth = sample['smooth']

        with torch.no_grad():
            y_hat = self.network(x, condition, mean=mean, std=std)

        M, loss_masks = self.get_loss_masking(grad, smooth, l_mask, l_mask_atlas, p_signal)
        if isinstance(y_hat, list):
            loss = self.sequence_loss(y_hat, target, p_signal, M, gamma=self.cfg.train.gamma)
            psnr = criterions.psnr(y_hat[-1], target)
            psnr_brain = criterions.psnr(y_hat[-1], target, mask=l_mask_atlas)
        else:
            loss = self.compute_loss(y_hat, target, p_signal, M)
            psnr = criterions.psnr(y_hat, target)
            psnr_brain = criterions.psnr(y_hat, target, mask=l_mask_atlas)

        if batch_idx < len(self._val_slices):
            prefix = f'val-{batch_idx}'
            s = self._val_slices[batch_idx]
            if self.global_step == 0:
                self.logger.experiment.add_image(
                    f'{prefix}/0-T1_sub',
                    make_grid(x[:, 0:1, s, :, :], nrow=4, normalize=True),
                    global_step=self.global_step
                )
                self.logger.experiment.add_image(
                    f'{prefix}/1-T1_nativ',
                    make_grid(x[:, 1:2, s, :, :], nrow=4, normalize=True),
                    global_step=self.global_step
                )
                # self.logger.experiment.add_image(
                #     f'{prefix}/2-T1_low',
                #     make_grid(x[:, 2:3, s, :, :], nrow=4, normalize=True),
                #     global_step=self.global_step
                # )
                if "T2" in self.cfg.data.inputs:
                    self.logger.experiment.add_image(
                        f'{prefix}/3-T2',
                        make_grid(x[:, 3:4, s, :, :], nrow=4, normalize=True),
                        global_step=self.global_step
                    )
                if "Diff" in self.cfg.data.inputs:
                    self.logger.experiment.add_image(
                        f'{prefix}/4-Diff0',
                        make_grid(x[:, 4:5, s, :, :], nrow=4, normalize=True),
                        global_step=self.global_step
                    )
                    self.logger.experiment.add_image(
                        f'{prefix}/5-Diff500',
                        make_grid(x[:, 5:6, s, :, :], nrow=4, normalize=True),
                        global_step=self.global_step
                    )
                    self.logger.experiment.add_image(
                        f'{prefix}/6-Diff1000',
                        make_grid(x[:, 6:7, s, :, :], nrow=4, normalize=True),
                        global_step=self.global_step
                    )
                self.logger.experiment.add_image(
                    f'{prefix}/8-target',
                    make_grid(target[:, :, s, :, :], nrow=4, normalize=True),
                    global_step=self.global_step
                )
                self.logger.experiment.add_image(
                    f'{prefix}/9-mask',
                    make_grid(l_mask[:, :, s, :, :], nrow=4, normalize=True),
                    global_step=self.global_step
                )
                self.logger.experiment.add_image(
                    f'{prefix}/91-mask-atlas',
                    make_grid(l_mask_atlas[:, :, s, :, :], nrow=4, normalize=True),
                    global_step=self.global_step
                )
                self.logger.experiment.add_image(
                    f'{prefix}/91-Loss_Small',
                    make_grid(loss_masks[0][:, :, s, :, :], nrow=4, normalize=True),
                    global_step=self.global_step
                )
                self.logger.experiment.add_image(
                    f'{prefix}/92-Loss_Smooth',
                    make_grid(loss_masks[1][:, :, s, :, :], nrow=4, normalize=True),
                    global_step=self.global_step
                )
            if "wloss" not in self.cfg.train.criterion:
                self.logger.experiment.add_image(
                    f'{prefix}/90-Loss_Threshold',
                    make_grid(loss_masks[2][:, :, s, :, :], nrow=4, normalize=True),
                    global_step=self.global_step
                )

            if isinstance(y_hat, list):
                y_hat = torch.concat(y_hat, 0)
            self.logger.experiment.add_image(
                f'{prefix}/7-pred',
                make_grid(y_hat[:, :, s, :, :], nrow=4, normalize=True),
                global_step=self.global_step
            )

        return {'loss': loss.item(), 'psnr': psnr, 'psnr_brain': psnr_brain}

    def validation_epoch_end(self, val_step_outputs):
        avg_val_loss = torch.tensor([x['loss'] for x in val_step_outputs]).mean()
        avg_val_psnr = torch.tensor([x['psnr'] for x in val_step_outputs]).mean()
        avg_val_psnr_brain = torch.tensor([x['psnr_brain'] for x in val_step_outputs]).mean()

        self.logger.log_metrics({'psnr/val': avg_val_psnr,
                                 'psnr-brain/val': avg_val_psnr_brain,
                                 'loss/val': avg_val_loss}, self.global_step)

        return {'val_loss': avg_val_loss}
