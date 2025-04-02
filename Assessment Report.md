# Audio Deepfake Detection Assessment Submission

## Introduction

This document outlines the work undertaken for the audio deepfake detection assessment. It covers the research into potential detection models, the rationale behind the implementation choices, and a detailed analysis of the process and findings. The goal was to explore current techniques and demonstrate practical skills in applying them to the challenge of identifying AI-generated speech, aligning with the need for robust detection systems in the face of emerging digital threats. This work draws upon initial research shared in supporting documents and practical implementation experience.

---

## Part 1: Research & Selection

**Objective:** Identify 3 promising audio deepfake detection models suitable for detecting AI-generated human speech, with potential for near real-time analysis of real conversations.

After reviewing the landscape, including resources like the [Audio-Deepfake-Detection collection](https://github.com/media-sec-lab/Audio-Deepfake-Detection) and relevant papers, and considering the criteria (AI speech detection, near real-time potential, conversation analysis), the following three approaches were selected as particularly promising:

**1. AASIST (Audio Anti-Spoofing using Integrated Spectro-Temporal Graph Attention Networks)**

* **Key Technical Innovation:** Employs Graph Attention Networks (specifically HS-GAL layers) to simultaneously model subtle artifacts across both spectral and temporal domains directly from the raw audio waveform. This joint modeling aims to capture inconsistencies often missed by methods analyzing single domains. [Source: [clovaai/aasist GitHub](https://github.com/clovaai/aasist)]
* **Reported Performance:** Achieved top-tier results on the ASVspoof 2019 LA benchmark (EER 0.83%, min t-DCF 0.0275). Its lightweight variant, AASIST-L, also performs strongly (EER 0.99%) with significantly reduced complexity (~85k parameters). [Source: AASIST Paper]
* **Promise for Use Case:** Working with raw waveforms minimizes information loss. AASIST-L's efficiency makes near real-time detection feasible. Strong benchmark results suggest good generalization potential for conversational analysis.
* **Potential Limitations:** Real-world conversational noise and channel effects might pose challenges not fully captured by ASVspoof. GNN interpretability can be difficult. Performance needs validation beyond benchmarks.

**2. M2S-ADD (Mono-to-Stereo Audio DeepFake Detection)**

* **Key Technical Innovation:** Introduces a novel pre-processing step where mono audio is converted to synthetic stereo. The underlying idea is that the conversion process might amplify or expose subtle forgery artifacts not apparent in the original mono signal, which are then analyzed by a dual-branch network. [Source: [ResearchGate M2S-ADD Paper](https://www.researchgate.net/publication/371123819_Betray_Oneself_A_Novel_Audio_DeepFake_Detection_Model_via_Mono-to-Stereo_Conversion)]
* **Reported Performance:** Demonstrated improved performance over several mono-based baselines on ASVspoof 2019 LA (EER 1.34%). [Source: M2S-ADD Paper]
* **Promise for Use Case:** Offers a unique angle for artifact detection, potentially effective against high-quality deepfakes where mono cues are minimal. Could be adapted for conversational analysis if stereo effects reveal artifacts.
* **Potential Limitations:** Heavily dependent on the quality and nature of the stereo synthesizer used. Effectiveness on genuine stereo or multi-channel audio found in real conversations is uncertain. Needs validation beyond ASVspoof. Real-time feasibility depends on synthesizer efficiency.

**3. Fine-tuned SSL Models (e.g., Wav2Vec 2.0, WavLM)**

* **Key Technical Innovation:** Leverages powerful, general-purpose audio representations learned by large Self-Supervised Learning (SSL) models pre-trained on massive unlabeled speech datasets. These foundational models are then fine-tuned specifically for the deepfake detection task using labeled spoofing datasets. [Source: [Eurecom Wav2Vec Paper](https://www.eurecom.edu/publication/6851/download/sec-publi-6851.pdf)]
* **Reported Performance:** Consistently rank among the top performers on ASVspoof challenges, achieving very low EERs (often below 1%) due to the robustness of the pre-trained features. [Source: Various ASVspoof papers/leaderboards]
* **Promise for Use Case:** Pre-trained features capture rich acoustic and speaker information, making them robust to variations found in real conversations. Fine-tuning is relatively data-efficient compared to training large models from scratch. High accuracy potential.
* **Potential Limitations:** Larger SSL models can be computationally demanding, potentially hindering real-time deployment without significant optimization. Performance can be sensitive to the choice of SSL model, fine-tuning strategy, and domain match between pre-training and target conversational data.

---

## Part 2: Implementation Selection & Setup

**Objective:** Select one approach and an appropriate dataset for implementation, focusing on the rationale and setup overview.

**Approach Selection Rationale: AASIST**

For the implementation part of this assessment, the **AASIST** model, specifically the **AASIST-L** (lightweight) variant, was chosen. This decision was based on several key factors aligning with the assessment's goals and the desired use case:

1.  **Performance vs. Efficiency Trade-off:** While fine-tuned SSL models often achieve the absolute lowest error rates, AASIST-L provides a compelling balance. It delivers near state-of-the-art performance on standard benchmarks but with a drastically smaller model size (~85k parameters). This efficiency is crucial for the target requirement of *potential for real-time or near real-time detection*. Large SSL models often present significant challenges for low-latency inference.
2.  **Raw Audio Processing:** AASIST operates directly on raw audio waveforms. This approach is promising because it avoids potential information loss that can occur during traditional feature extraction (like MFCCs or spectrograms), potentially allowing the model to capture more subtle, low-level artifacts introduced by AI generation processes.
3.  **Novel Architecture:** The spectro-temporal graph attention mechanism is an innovative technique specifically designed for anti-spoofing. Implementing and evaluating this approach provides valuable insight into non-standard deep learning architectures for audio analysis.
4.  **Code Availability and Reproducibility:** The availability of a well-maintained and clear official PyTorch implementation ([clovaai/aasist](https://github.com/clovaai/aasist)) significantly lowers the barrier to entry for implementation and ensures a degree of reproducibility, which is vital for assessment and future development.

**Dataset Selection Rationale: ASVspoof 2019 Logical Access (LA)**

The **ASVspoof 2019 LA** dataset was selected as the most appropriate choice for this implementation task:

1.  **Standard Benchmark:** It serves as the primary benchmark dataset for evaluating audio anti-spoofing systems, including the original AASIST model. Using this dataset allows for meaningful comparison with published results and understanding the model's baseline performance in a controlled environment.
2.  **Relevance and Diversity:** The dataset includes spoofed audio generated using various Text-to-Speech (TTS) and Voice Conversion (VC) algorithms representative of the state-of-the-art at the time of its creation. While newer threats exist, ASVspoof 2019 LA provides a foundational challenge set.
3.  **Compatibility:** The official AASIST codebase is designed to work directly with the structure and protocols of the ASVspoof 2019 LA dataset. The repository even includes a script (`download_dataset.py`) to facilitate acquisition and setup, streamlining the implementation process.
4.  **Focus on Logical Access:** The LA scenario (distinguishing spoofed from genuine speech for system access) aligns well with the core task of detecting AI-generated human speech.

**Implementation Details:**

The step-by-step implementation, including environment setup, data download/preparation using the provided script, necessary code modifications (NumPy type fix, epoch adjustment), and the execution of the light re-training process, is detailed in the Jupyter notebook within the project repository.

* **Code Location:** Please refer to the `AASIST Audio Deepfake Detection Implementation.ipynb` notebook in the repository: [github.com/mangotraabhi/AASIST-Audio-Deepfake-Detection](https://github.com/mangotraabhi/AASIST-Audio-Deepfake-Detection)

---

## Part 3: Documentation & Analysis

**Objective:** Document the implementation journey, analyze the chosen model's workings and results, and reflect on broader considerations.

**Implementation Process Documentation:**

* **Setup:** The environment was prepared by cloning the repository (`github.com/clovaai/aasist` and installing dependencies listed in `requirements.txt` using `pip` within a Python virtual environment. This process was straightforward.
* **Data Handling:** The ASVspoof 2019 LA dataset was successfully downloaded and extracted using the `download_dataset.py` script included in the original AASIST repository. This placed the data into the expected `./LA` directory structure relative to the main scripts.
* **Code Modifications:** Two key modifications were applied to the original `clovaai/aasist` code before execution:
    * In `evaluation.py`, instances of `.astype(np.float)` were changed to `.astype(np.float64)` to ensure compatibility with modern NumPy versions where `np.float` is deprecated.
    * In the configuration file `config/AASIST-L.conf`, the `"num_epochs"` parameter was reduced from `100` to `10`. This was a practical adjustment to enable a "light re-training" run within the time limits of typical free GPU resources (like Google Colab), sufficient for verifying the pipeline's functionality.
* **Execution:** The training was initiated via the command line using `python main.py --config ./config/AASIST-L.conf`. The script correctly identified the dataset location and proceeded through the 10 configured training epochs.
* **Challenges & Solutions:**
    * *NumPy Deprecation:* The primary technical hurdle was the error caused by the deprecated `np.float`. This was identified through runtime errors and resolved by switching to the explicit `np.float64` type.
    * *Resource Constraints:* Full training was impractical. Reducing the epoch count in the configuration file was the necessary solution for demonstrating the process within available time/compute limits.
    * *Dataset Download:* While automated by the script, the large dataset size meant the download and extraction step was time-consuming.

**Analysis Section:**

* **Model Functionality (AASIST High-Level):** AASIST processes raw audio waveforms using a convolutional front-end inspired by RawNet2. The core innovation lies in its use of Heterogeneous Stacking Graph Attention Layers (HS-GAL). These layers construct and analyze graph representations of the audio, learning intricate patterns simultaneously within the spectral domain, within the temporal domain, and crucially, *across* these domains via specialized attention mechanisms and a "stack" node aggregator. A subsequent Max Graph Operation (MGO) module uses parallel graph branches to capture diverse spoofing cues robustly before a final classification layer determines if the audio is bona fide or spoofed.
* **Performance Results (Light Re-training on ASVspoof 2019 LA):** The 10-epoch training run of AASIST-L yielded indicative results:
    * **Training Loss:** Showed a clear downward trend, indicating learning was occurring (e.g., starting around 0.66 and decreasing significantly).
    * **Development Set EER/t-DCF:** Improved markedly over the 10 epochs (e.g., EER dropping from ~20% towards ~5-6% range, t-DCF showing similar improvement).
    * **Interpretation:** These results validate that the implementation pipeline is working correctly. However, they represent only initial learning. Achieving the benchmark performance reported in the paper (EER ~0.99% for AASIST-L) would require completing the full training schedule (100+ epochs) and potentially specific hyperparameter tuning.
* **Observed Strengths:**
    * The model demonstrates rapid initial learning, with significant performance gains even in few epochs.
    * The official codebase is well-structured and integrates relatively easily with the standard ASVspoof dataset format.
    * The lightweight nature of AASIST-L makes it feasible to experiment even in resource-constrained environments.
* **Observed Weaknesses / Limitations:**
    * Full training is computationally intensive.
    * Performance is benchmarked on ASVspoof; generalization to diverse, noisy, real-world conversational audio requires further investigation and likely adaptation.
    * Interpreting the specific features learned by the GNN components remains challenging.
* **Suggestions for Future Improvements:**
    * **Complete Full Training:** Run the training for the originally specified number of epochs to properly evaluate benchmark performance.
    * **Augment Training Data:** Incorporate diverse data augmentation techniques (e.g., adding background noise, simulating codec compression, applying reverberation) to improve model robustness to real-world conditions.
    * **Cross-Dataset/Domain Evaluation:** Test the trained model on different deepfake datasets and, ideally, on custom datasets representative of the target conversational environment.
    * **Hyperparameter Optimization:** Conduct systematic tuning of learning rates, batch sizes, and potentially model architecture parameters if deviating from the benchmark setup.

**Reflection Questions:**

* **Most Significant Challenges in Implementation:** Beyond managing compute resources for training, the main challenge was ensuring perfect alignment between the dataset structure and the code's expectations, and diagnosing/fixing subtle compatibility issues like the NumPy deprecation. Understanding the nuances of the GNN architecture required careful reading of the source paper.
* **Real-World Performance vs. Research Datasets:** A performance gap is almost certain. Real-world audio contains a much wider range of acoustic variability (noise, microphones, room acoustics, channel effects like phone lines/VoIP) and potentially novel spoofing techniques not present in curated datasets like ASVspoof. The model would likely be less accurate and require significant adaptation (e.g., fine-tuning on in-domain data) for reliable real-world deployment.
* **Additional Data/Resources for Improvement:**
    * **In-Domain Data:** Access to large amounts of audio data representative of the specific deployment scenario (e.g., call center audio, meeting recordings) is crucial for fine-tuning and robust evaluation.
    * **Diverse & Recent Deepfakes:** Datasets containing audio generated by the latest state-of-the-art TTS and VC models are needed to ensure the detector keeps pace with evolving threats.
    * **More Compute Power:** Enables longer training runs, larger batch sizes, more extensive data augmentation, and hyperparameter optimization experiments.
* **Approach for Production Deployment:**
    1.  **Domain Adaptation:** Fine-tune the AASIST-L model on a large, representative dataset from the target production environment.
    2.  **Performance Optimization:** Apply model optimization techniques (e.g., quantization using PyTorch tools, ONNX conversion) to minimize inference latency and computational footprint, especially if targeting real-time or edge deployment. Profile on target hardware.
    3.  **Threshold Tuning:** Carefully select the classification threshold based on the specific application's tolerance for false positives versus false negatives.
    4.  **Scalable Serving:** Deploy the optimized model as a microservice using a robust serving framework (e.g., TorchServe, Triton Inference Server, or a custom FastAPI wrapper) within a containerized environment (e.g., Docker, Kubernetes).
    5.  **Comprehensive Monitoring:** Implement logging and monitoring for key metrics: inference latency, throughput, prediction confidence scores, error rates, and input data characteristics (to detect drift). Set up alerts for anomalies.
    6.  **Feedback & Retraining Pipeline:** Establish a system for collecting difficult or misclassified examples from production traffic to feed into a continuous evaluation and retraining loop, ensuring the model stays effective over time.

---
