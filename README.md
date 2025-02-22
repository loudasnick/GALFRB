# GALFRB

Welcome to the GALFRB repository!

<p align="center">
  <img src="FRB_sketch.jpg" alt="Description" width="1000">
  <br>
  <sub><em>Artist’s impression of CSIRO’s Australian SKA Pathfinder (ASKAP) radio telescope finding a fast radio burst and determining its precise location. The KECK, VLT and Gemini South optical telescopes joined ASKAP with follow-up observations to image the host galaxy. Credit: CSIRO/Dr Andrew Howells</em></sub>
</p>


## Description

GALFRB is a project aimed at unveiling the origing of Fast Radio Bursts (FRBs) by modeling their hosts' properties. This repository contains flexible modules for generating joint distributions of the properties of mock galaxies to compare with FRB hosts and provide stringent constraints on their origin.

## Installation and Usage

To get started with GALFRB, follow these steps:

1. Clone the repository: `git clone https://github.com/loudasnick/GALFRB.git`
2. Navigate to GALFRB directory: `cd GALFRB/`
3. Create a new conda environment devoted for executing GALFRB: `conda create -n GALFRB_evn python==3.8.19 ipykernel`
4. Activate conda env: `conda activate GALFRB_evn` <!-- 2. Install the required dependencies: `pip install -r requirements.txt` -->
5. Install required libraries: `pip install -r requirements.txt`
6. Install GALFRB: `pip install -e .`
7. Download SDSS+WISE galaxy catalog (used in modeling the probability density in color-sfr plane): `python download_sdss_wise_data.py`
8. Run the tutorial found in `examples/` to verify the correct installation of the package

You are all set!

## Contributing

We welcome contributions from the community! 
<!--If you would like to contribute to GALFRB, please follow our [contribution guidelines](CONTRIBUTING.md). -->

## License

This project is licensed under the [MIT License](LICENSE).
