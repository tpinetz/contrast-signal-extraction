from datetime import date
import hydra
import logging
import numpy as np
from omegaconf import DictConfig
import os
import torch

import data
import models


ch = logging.StreamHandler()

logging.basicConfig(level=logging.INFO,
                    handlers=[
                        logging.FileHandler("test.log", mode="w"),
                        ch,
                    ])


def process_series(save_dir: str, model: torch.nn.Module,
                   cfg: DictConfig, data_cfg: dict, sd: data.SeriesData):
    device = torch.device('cuda:' + str(cfg.gpu) if cfg.gpu and torch.cuda.is_available() else 'cpu')

    model = model.to(device)

    # Remove unused parts of the config that are slow for inference
    data_cfg['add_grad'] = False
    data_cfg['adaptive_threshold'] = False

    logging.info("Computing prediction:")
    logging.info(f"Dose: {sd['T1_low'].attrs['dose']:.2f}ml / {sd['T1_low'].attrs['percent']:.1f}%")
    logging.info(f"Field strength: {sd['T1_low'].attrs['field strength']:.1f}T")

    inputs = sd.to_input(data_cfg)

    base_index = data_cfg['inputs'].index(data.SeriesData.low) + 1
    prediction, diff = model.inference(inputs, device, base_index=base_index)

    # save the result
    logging.info("Saving prediction")
    prediction, window = sd.to_reference(np.maximum(prediction, 0), data_cfg, return_contrast_window=True)
    prediction.save(os.path.join(save_dir, "t1_ai.nii.gz"), format="nib")


@hydra.main(config_path="config", config_name="test")
def main(cfg: DictConfig):
    logging.info(cfg)

    ckpt = cfg.ckpt
    logging.info(f"Using ckpt: {ckpt}")
    
    # load the model
    checkpoint = torch.load(ckpt, map_location='cpu')
    model_cfg = checkpoint['hyper_parameters']['cfg']
    logging.info(model_cfg)
    
    smartcontrast = models.SmartContrastModel(model_cfg, fast=True)
    smartcontrast.load_state_dict(checkpoint['state_dict'])
    model = smartcontrast.network
    model.eval()
    
    today = date.today()
    save_dir = os.path.join(cfg.output_path, '{}'.format(str(today)))
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    sd = data.SeriesData(register_images=True, fast=cfg.fast).load(input_files={
        "T1_zero": cfg.T1_zero,
        "T1_low": cfg.T1_low
    })

    process_series(save_dir, model, cfg, model_cfg['data'], sd)


if __name__ == '__main__':
    main()
