# AASIST Audio Deepfake Detection Implementation

This repository contains the submission for the ML/Data Curation Specialist assessment, focusing on the detection of audio deepfakes using the AASIST model.

## Overview

Audio deepfakes pose a significant threat to digital trust. This project explores deepfake detection methods, focusing on implementing the **AASIST (Audio Anti-Spoofing using Integrated Spectro-Temporal Graph Attention Networks)** approach. The implementation uses the official [clovaai/aasist](https://github.com/clovaai/aasist) codebase and the standard ASVspoof 2019 Logical Access (LA) dataset.

This repository includes:
* A Jupyter/Colab notebook demonstrating the implementation (Part 2).
* A detailed report covering the research, selection, and analysis (Parts 1 & 3).
* Setup instructions and dependency information.

## Repository Structure

├── AASIST_Implementation.ipynb      
 ├── Assessment_Report.md   
 ├── README.md
## Dataset: ASVspoof 2019 Logical Access (LA)

This implementation uses the **ASVspoof 2019 Logical Access (LA)** dataset.

* **Source:** The dataset can be obtained from the official ASVspoof challenge organizers, typically via resources like the [Edinburgh DataShare](https://datashare.ed.ac.uk/handle/10283/3336) (requires registration and agreement to terms).
* **Download:** The original `clovaai/aasist` repository (and likely this repository's version) includes a `download_dataset.py` script. Running this script within the cloned `aasist` directory (as shown in the notebook) should download and extract the dataset automatically.
* **Structure:** The script extracts the dataset into a subdirectory named `LA` relative to the script's location (i.e., `./LA/`). The codebase expects this structure, containing `ASVspoof2019_LA_train`, `ASVspoof2019_LA_dev`, `ASVspoof2019_LA_eval`, and `ASVspoof2019_LA_protocols`.

## Setup

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/mangotraabhi/AASIST-Audio-Deepfake-Detection.git](https://github.com/mangotraabhi/AASIST-Audio-Deepfake-Detection.git)
    cd AASIST-Audio-Deepfake-Detection
    ```
2.  **Python Environment:** It's recommended to use a virtual environment (e.g., `venv` or `conda`). Python 3.8+ is recommended (tested with Python 3.10/3.11 in Colab).
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

## Running the Implementation (Notebook)

The core implementation is contained within the Jupyter notebook: `AASIST_Implementation.ipynb`.

1.  Ensure your environment is set up and activated (see Setup).
2.  Launch Jupyter Notebook or Jupyter Lab:
    ```bash
    jupyter notebook AASIST_Implementation.ipynb
    # or
    # jupyter lab
    ```
3.  Open the `AASIST_Implementation.ipynb` notebook.
4.  Run the cells sequentially. The notebook handles:
    * Cloning the original `clovaai/aasist` code (if not already present, though this repo contains the necessary code).
    * Installing dependencies (redundant if done via `requirements.txt` but included for completeness in the notebook).
    * Downloading and extracting the ASVspoof 2019 LA dataset using the provided script.
    * Running the training process for the AASIST-L model (configured for 10 epochs).

**Note:** The first time you run the dataset download cell, it will take a significant amount of time and require considerable disk space (~10-15 GB during extraction).

## Code Modifications Note

To ensure compatibility and practical execution within typical environments (like Google Colab), the following modifications were made to the original `clovaai/aasist` codebase (these changes are reflected in the code used within the notebook/this repository):

1.  **`evaluation.py`:** Replaced deprecated `np.float` with `np.float64` when casting ASV/CM scores to avoid errors with recent NumPy versions.
2.  **`config/AASIST-L.conf`:** Reduced `"num_epochs"` from `100` to `10` to allow for a "light re-training" demonstration within typical GPU time constraints.

## Key Findings / Results Summary (Light Re-training)

Running the AASIST-L model for the configured 10 epochs demonstrated successful setup and initial learning. Key observations from the training logs include:
* Progressive decrease in training loss.
* Significant improvement in the development set Equal Error Rate (EER) and tandem Detection Cost Function (t-DCF) over the 10 epochs (e.g., EER dropping from ~20% to ~5% within the first few epochs).

These results confirm the pipeline's correctness but **do not** represent the model's fully trained performance. Achieving benchmark results requires completing the full training duration specified by the authors.

## Detailed Report

For a comprehensive discussion covering the research background, model selection rationale (Part 1), and detailed documentation & analysis (Part 3), please refer to the report file in this repository:

* **[Assessment_Report.md](./Assessment_Report.md)**

