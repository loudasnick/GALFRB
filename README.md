# GALFRB

Welcome to the GALFRB repository!

<p align="center">
  <img src="FRB_sketch.jpg" alt="Description" width="400">
  <br>
  <em>Figure 1: Your legend text here</em>
</p>


## Description

GALFRB is a project aimed at unveiling the origing of Fast Radio Bursts (FRBs) by modeling their hosts' properties. This repository contains flexible modules for generating joint distributions of the properties of mock galaxies to compare with FRB hosts and provide stringent constraints on their origin.

## Installation and Usage

To get started with GALFRB, follow these steps:

1. Clone the repository: `git clone https://github.com/your-username/GALFRB.git`
2. Navigate to GALFRB directory.
3. Create a new conda environment devoted for executing GALFRB: `conda env create -f environment.yml`
4. Activate conda env: `conda activate galfrb_evn`
<!-- 2. Install the required dependencies: `pip install -r requirements.txt` -->
5. Install GALFRB: `pip install -e .`
6. Download SDSS+WISE galaxy catalog (used in modeling the ): `python download_sdss_wise_data.py`
7. Run the tutorial found in `examples/` to verify the correct installation of the package

You are all set!

## Contributing

We welcome contributions from the community! 
<!--If you would like to contribute to GALFRB, please follow our [contribution guidelines](CONTRIBUTING.md). -->

## License

This project is licensed under the [MIT License](LICENSE).
