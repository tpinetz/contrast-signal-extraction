# contrast-signal-extraction

Code for "Artificial T1-weighted Post-Contrast Brain MRI: A Deep Learning Method for Contrast Signal Extraction" (tag v1.0, model.ckpt) in Investigative Radiology 2024 by Haase, Pinetz et al.

Code for "Metastasis Detection Using True and Artificial T1-Weighted Postcontrast Images in Brain MRI" (tag v2.0, model\_33p.ckpt) in Investigative Radiology 2024 by Haase, Pinetz et al..


## Test script

The file `src/test.py` is used to improve the contrast of T1 weighted post contrast enanced images.
To execute the script the `.yml.exmaple` have to renamed to `.yml` and the paths have to be adjusted.
The config and paths are defined in `src/config/test.yml`.

Then to execute the script navigate to the `src` folder and execute using `python test.py`.

The used environment is described in the file `requirements.yml` and a virtual environment can be constructed using `conda env create -f requirements.yml`. 


## Example Scan

We used "Faithful Synthesis of Low-Dose Contrast-Enhanced Brain MRI Scans Using Noise-Preserving Conditional GANs." by Pinetz, Kobler et al. [MICCAI 2023][^1] to produce an example scan which is used to showcase the capabilities.
The scan is located in the example folder.


### Citation

If you use the code from this repository for your own research, please cite the following studies:

- *Haase, Robert and Pinetz, Thomas and Kobler, Erich and Bendella, Zeynep and Gronemann, Christian and Paech, Daniel and Radbruch, Alexander and Effland, Alexander and Deike, Katerina.* (2024) **Artificial T1-weighted Post-Contrast Brain MRI: A Deep Learning Method for Contrast Signal Extraction**. In: Investigative radiology.
- *Haase, Robert and Pinetz, Thomas and Kobler, Erich and Bendella, Zeynep and Paech, Daniel and Clauberg, Ralph and ... and Deike, Katerina* (2024) **Metastasis Detection Using True and Artificial T1-Weighted Postcontrast Images in Brain MRI**. In: Investigative radiology published ahead of print.
- *Pinetz, Thomas and Kobler, Erich and Haase, Robert and Luetkens. Julian A. and Meetschen, Mathias, and Haubold, Johannes and Deuschl, Cornelius and Radbruch, Alexander and Deike, Katerina and and Effland, Alexander* (2024) **Gadolinium dose reduction for brain MRI using conditional deep learning**. In: arXiv.

[^1]: https://github.com/tpinetz/low-dose-gadolinium-mri-synthesis
