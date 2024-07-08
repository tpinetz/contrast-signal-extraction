import hydra
import logging
from omegaconf import OmegaConf, DictConfig
import os
import pytorch_lightning as pl
import pytorch_lightning.callbacks as callbacks
from pytorch_lightning.loggers import TensorBoardLogger

from models import SmartContrastModel
import data

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"


# logging ---------------------------------------------------------------------
ch = logging.StreamHandler()

logging.basicConfig(level=logging.INFO,
                    handlers=[
                        logging.FileHandler("train.log", mode="w"),
                        ch,
                    ])


# Formatter for result path
OmegaConf.register_new_resolver("format", lambda x, pattern: f"{x:{pattern}}")


# training ---------------------------------------------------------------------
@hydra.main(config_path="config", config_name="train")
def main(cfg: DictConfig) -> None:
    logging.info(f"Saving results to '{os.getcwd()}'")

    pl.seed_everything(42, workers=True)

    model = SmartContrastModel(cfg)
    dm = data.MRIDataModule(cfg)

    logger = TensorBoardLogger(".", name="")

    trainer = pl.Trainer(max_steps=cfg.optim.num_iter,
                         gpus=cfg.train.gpu,
                         strategy='dp',
                         num_sanity_val_steps=-1,
                         logger=logger,
                         callbacks=[
                            callbacks.LearningRateMonitor(logging_interval='step'),
                            callbacks.ModelCheckpoint(save_top_k=-1, every_n_train_steps=cfg.train.num_save)
                         ])
    trainer.fit(model, dm)

    logging.info('Successfully finished training!')


if __name__ == '__main__':
    main()
