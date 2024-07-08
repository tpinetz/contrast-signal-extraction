# contrast-signal-extraction

Code for "Artificial T1-weighted Post-Contrast Brain MRI: A Deep Learning Method for Contrast Signal Extraction" in Investigative Radiology 2024 by Haase, Pinetz et al.


## Test script

The file `src/test.py` is used to improve the contrast of 20p images.
To execute the script the `.yml.exmaple` have to renamed to `.yml` and the paths have to be adjusted.
The config and paths are defined in `src/config/test.yml`.

Then to execute the script navigate to the `src` folder and execute using `python test.py`.

The used environment is described in the file `requirements.yml` and a virtual environment can be constructed using `conda env create -f requirements.yml`. 


## Example Scan

We used "Faithful Synthesis of Low-Dose Contrast-Enhanced Brain MRI Scans Using Noise-Preserving Conditional GANs." by Pinetz, Kobler et al. [MICCAI 2023] to produce an example scan which is used to showcase the capabilities.
The scan is located in the example folder.


### Citation

If you use anything from this repository for your own research, please cite:

- *Haase, Robert and Pinetz, Thomas and Kobler, Erich and Bendella, Zeynep and Gronemann, Christian and Paech, Daniel and Radbruch, Alexander and Effland, Alexander and Deike, Katerina.* (2024) **Artificial T1-weighted Post-Contrast Brain MRI: A Deep Learning Method for Contrast Signal Extraction**. In: Investigative radiology published ahead of print.