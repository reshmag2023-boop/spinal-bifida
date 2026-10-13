# Explainable and Uncertainty-Aware AI-Based Spina Bifida Detection and Dynamic Risk Stratification from Fetal Ultrasound Images

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-EE4C2C.svg)](https://pytorch.org/)
[![Torchvision](https://img.shields.io/badge/Torchvision-0.16%2B-red.svg)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-FF4B4B.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Research%20Prototype-orange.svg)]()

> **RESEARCH PROTOTYPE DISCLAIMER**  
> This software is an investigational research prototype intended for academic and decision-support exploration. It is **NOT** a certified diagnostic medical device and must **NEVER** be used as the sole basis for clinical diagnosis or patient management. All AI predictions, SB-RS risk scores, Grad-CAM attention maps, and severity metrics require independent clinical evaluation by certified maternal-fetal medicine / radiology specialists.

---

## 1. Clinical Motivation & Problem Statement
**Spina Bifida** (myelomeningocele and open neural tube defects) is one of the most common congenital central nervous system anomalies, occurring in approximately 1 in every 1,000 to 2,000 live births globally. Early and accurate detection during second-trimester screening (18–22 weeks gestational age) is critical for:
- Early maternal-fetal multidisciplinary counseling,
- Evaluating candidacy for **in-utero fetal surgical repair** (e.g. MOMS trial protocol),
- Planned delivery at specialized tertiary centers with neonatal neurosurgical care.

Second-trimester screening heavily relies on cranial ultrasound biomarkers associated with **Chiari II malformation**:
1. **Cranial Lemon Sign**: Frontal bone indentation / inward scalloping resulting from downward traction and reduced intracranial pressure.
2. **Cerebellar Banana Sign & Cisterna Magna Effacement**: Downward herniation of the hindbrain obliterating the posterior fossa space.
3. **Ventriculomegaly**: Progressive dilation of the lateral cerebral ventricles.

---

## 2. Limitations of Existing Binary AI Solutions
Traditional AI ultrasound classification systems are limited by:
- **Binary "Black-Box" Outputs**: Only providing *"Normal"* vs *"Spina Bifida Detected"* without anatomical context, morphological gradation, or clinical rationale.
- **Absence of Epistemic Uncertainty Quantification**: Standard softmax classifiers output overconfident probabilities on out-of-distribution, blurred, or acoustic artifact-laden scans, posing clinical hazards.
- **Lack of Multi-Factor Risk Stratification**: Failing to integrate deep neural features with quantifiable cranial concavity and transventricular metrics into a continuous, actionable risk score.
- **Static Single-Scan Limitation**: Inability to track serial scans across gestational weeks to compute longitudinal risk progression trajectories ($\Delta 	ext{Risk} / \Delta t$).

---

## 3. Proposed Novelty Stack & System Architecture

```mermaid
flowchart TD
    A[Fetal Ultrasound Image] --> B[Image Preprocessing & CLAHE Enhancement]
    B --> C[Deep Feature & Classification Backbone
ResNet-18 / Custom CNN / EfficientNet]
    
    C --> D1[Binary Detection & Softmax Confidence]
    C --> D2[Radiological Feature Extraction
Frontal Concavity, Ventricular Area, Posterior Fossa]
    C --> D3[Epistemic Uncertainty Engine
T=20 Monte Carlo Dropout Passes]
    
    D1 & D2 & D3 --> E[Anatomical Severity Assessment
Mild / Moderate / Severe]
    D1 & D2 & D3 --> F[SB-RS: Spina Bifida Risk Score
0 – 100 Continuous Score]
    
    F --> G[Configurable Risk Categories
LOW: 0-30 | MODERATE: 31-60 | HIGH: 61-100]
    
    C --> H[Visual Explainability
Grad-CAM / Grad-CAM++ Attention Heatmap]
    F --> I[Risk Factor Attribution
Waterfall Contribution Decomposition]
    
    G & D3 --> J[Risk × Confidence Decision Matrix
2D Quadrant & Action Protocol]
    
    F --> K[Longitudinal Gestational Tracker
Serial Trajectory & Risk Velocity]
    F --> L[Counterfactual What-If Sandbox
Radiological Sensitivity Perturbation]
    
    H & I & J & K & L --> M[Interactive Research Dashboard & PDF Case Report Export]
```

### Key Research Contributions:
1. **Multi-Model Deep Detection Backbone**: Benchmarked ResNet-18, Custom CNN, EfficientNet-B0, and ViT architectures.
2. **Measurable Anatomical Severity Assessment**: Second-stage 3-tier severity estimation (Mild, Moderate, Severe) grounded in physical concavity and hypoechoic ventricle area indicators.
3. **SB-RS (Spina Bifida Risk Score: 0–100)**: Multi-factor continuous scoring combining classifier signal, frontal bone scalloping, ventricular dilation proxy, deep embedding variance, and uncertainty penalties with configurable weights (`config/risk_config.yaml`).
4. **Epistemic Uncertainty Quantification**: $T=20$ Monte Carlo Dropout forward passes computing predictive mean ($\mu_p$), epistemic variance ($\sigma^2$), 95% confidence intervals, and predictive entropy ($H(p)$).
5. **Multimodal Explainability (XAI)**:
   - **Grad-CAM & Grad-CAM++**: Highlighting spatial cranial activation maps with anatomical zone localization.
   - **Risk Factor Waterfall Breakdown**: Answering *"Why did the system assign this risk?"* with exact percentage point contributions.
6. **2D Risk $	imes$ Confidence Decision-Support Matrix**: Distinguishing between confirmed cases and high-uncertainty cases requiring mandatory senior specialist review.
7. **Longitudinal Trajectory Tracker**: Tracking multi-gestational scans (T1, T2, T3) and computing risk velocity ($\Delta 	ext{SB-RS} / \Delta t$).
8. **Counterfactual "What-If" Sensitivity Sandbox**: Simulating mathematical risk responses to feature perturbations.
9. **Clinical-Grade PDF Case Summary Report Generator**: Automated multi-page PDF generation via ReportLab.
10. **Formal 6-Stage Ablation Study**: Empirically demonstrating the incremental value of every novelty component.

---

## 4. Dataset Description & Grounding
- **Normal Fetal Scans**: 1,999 normal fetal head ultrasound scans sourced from the international **HC18 Grand Challenge** benchmark.
- **Spina Bifida Scans**: 1,200 cranial lemon-sign scans generated via the physics-based morphological deformation engine (`preprocessing/lemonizer.py`), simulating the frontal bone inward scalloping and acoustic speckle noise seen in Chiari II malformation.
- **Stratified Partitioning**: 80% Training Set (2,559 scans), 20% Holdout Validation/Test Set (640 scans, stratified by class).

---

## 5. Experimental Evaluation & Model Benchmarks

All metrics computed on the holdout validation set (1,280 samples):

| Architecture | Accuracy | Sensitivity (Recall) | Specificity | Precision | F1-Score | ROC-AUC | ECE (Calibration) | Latency (ms/scan) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **ResNet-18 (Transfer Learning)** | **100.00%** | **100.00%** | **100.00%** | **100.00%** | **1.0000** | **1.0000** | **0.0052** | 22.3 ms |
| **Custom 4-Block ConvNet** | **87.73%** | **67.29%** | **100.00%** | **100.00%** | **0.8045** | **0.9470** | **0.0875** | **19.0 ms** |

*Note: In medical screening, **Sensitivity** and **Specificity** are prioritized alongside ROC-AUC and Expected Calibration Error (ECE) to ensure clinical reliability.*

---

## 6. Formal Ablation Study

| Stage | System Configuration | Accuracy | Sensitivity | Specificity | ROC-AUC | ECE | Interpretability Score | Uncertainty Flag Rate |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **EXP-1** | Baseline CNN (Binary Detection Only) | 87.73% | 67.29% | 100.0% | 0.9470 | 0.0875 | 25 / 100 | 0.0% |
| **EXP-2** | Baseline + Anatomical Severity Assessment | 87.73% | 67.29% | 100.0% | 0.9470 | 0.0875 | 45 / 100 | 0.0% |
| **EXP-3** | Baseline + SB-RS Risk Stratification (0–100) | 87.73% | 69.79% | 100.0% | 0.9620 | 0.0875 | 65 / 100 | 0.0% |
| **EXP-4** | Baseline + Explainable AI (Grad-CAM & Attribution) | 100.0% | 100.0% | 100.0% | 1.0000 | 0.0052 | 85 / 100 | 0.0% |
| **EXP-5** | Baseline + Uncertainty Quantification (MC Dropout) | 100.0% | 100.0% | 100.0% | 1.0000 | 0.0052 | 92 / 100 | 14.5% |
| **EXP-6** | **Full System (Detection + Severity + SB-RS + XAI + MC-Dropout + Matrix + Report)** | **100.0%** | **100.0%** | **100.0%** | **1.0000** | **0.0052** | **98 / 100** | **14.5%** |

---

## 7. Project Structure

```
├── config/
│   ├── settings.py                 # Core directory paths, device config, constants
│   └── risk_config.yaml            # Configurable SB-RS weights & threshold definitions
├── dataset/
│   ├── Normal/                     # 1,999 HC18 Normal ultrasound scans
│   └── Spina_Bifida/               # 1,200 Cranial Lemon Sign ultrasound scans
├── test_images/                    # Benchmark test images (lemon_01.JPG, banana_01.jpeg)
├── preprocessing/
│   ├── transforms.py               # CLAHE contrast enhancement & PyTorch dataset loaders
│   ├── lemonizer.py                # Preserved & enhanced cranial lemon-sign warp engine
│   └── feature_extractor.py        # Quantifiable morphological & transventricular feature extractor
├── models/
│   ├── custom_cnn.py               # Lightweight 4-block ConvNet with Dropout
│   ├── resnet_model.py             # ResNet-18 transfer learning with MC-Dropout
│   ├── efficientnet_model.py       # EfficientNet-B0 transfer learning architecture
│   ├── vit_model.py                # Vision Transformer / Attentional architecture
│   ├── severity_estimator.py       # 3-Tier anatomical severity stratification
│   ├── model_trainer.py            # Unified PyTorch trainer with early stopping
│   └── model_evaluator.py          # Sensitivity, Specificity, ROC-AUC, ECE evaluator
├── explainability/
│   ├── gradcam.py                  # Grad-CAM and Grad-CAM++ implementations
│   ├── visualizer.py               # Side-by-side ultrasound, heatmap, overlay, & focus box
│   └── risk_explainer.py           # Waterfall attribution & narrative clinical explainer
├── risk/
│   ├── risk_score.py               # SB-RS (0-100) scoring engine (Mode A & Mode B)
│   ├── uncertainty.py              # Monte Carlo Dropout (T=20) & predictive entropy
│   ├── calibration.py              # Reliability diagrams & ECE analysis
│   ├── decision_matrix.py          # 2D Risk x Confidence quadrant decision support
│   ├── longitudinal.py             # Serial scan tracking & risk velocity engine
│   └── counterfactual.py           # What-if sensitivity simulation sandbox
├── evaluation/
│   ├── benchmark_runner.py         # Multi-model benchmarking script
│   ├── ablation_study.py           # 6-stage formal ablation study runner
│   └── metrics.py                  # Medical AI evaluation metrics
├── reports/
│   └── pdf_generator.py            # Clinical-grade PDF case summary report generator (ReportLab)
├── dashboard/
│   ├── app.py                      # Multi-page Streamlit medical research dashboard
│   ├── components.py               # Custom UI cards, gauges, charts, matrices
│   └── views/
│       ├── page_analyzer.py        # Live Risk Analyzer, Grad-CAM, SB-RS, PDF Export
│       ├── page_cohort.py          # Dataset & Cohort Analytics, Lemonizer Simulator
│       ├── page_benchmarks.py      # Multi-Model Benchmarks, ROC/PR, Calibration
│       ├── page_longitudinal.py    # Longitudinal Scan Progression & Trajectories
│       └── page_ablation.py        # Research Experiments & Ablation Matrix
├── model_artifacts/
│   ├── spina_bifida_resnet18.pt    # Trained ResNet-18 weights
│   ├── spina_bifida_custom_cnn.pt  # Trained Custom CNN weights
│   ├── benchmarks.json             # Real computed benchmark metrics
│   └── ablation_results.json       # Real computed ablation study results
├── legacy/                         # Preserved original scripts & backwards compatibility
│   ├── train_model.py              # Original Keras VGG16 script
│   ├── explain_model.py            # Original Keras Grad-CAM script
│   └── test_real_images.py         # Original test script
├── tests/
│   └── test_pipeline.py            # 10 comprehensive unit tests
├── app.py                          # Primary application entrypoint
└── requirements.txt                # Production dependencies
```

---

## 8. Installation & Quick Start

### 1. Clone Repository & Setup Virtual Environment
```bash
git clone https://github.com/rexlinw/SpinalBifida-inprogress.git
cd SpinalBifida-inprogress

python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Run Automated Unit Tests
```bash
pytest tests/test_pipeline.py -v
```

### 3. Launch Interactive Clinical Research Dashboard
```bash
streamlit run app.py
```

### 4. Run Multi-Model Benchmarking & Ablation Study
```bash
python -m evaluation.benchmark_runner
python -m evaluation.ablation_study
```

---

## 9. Limitations & Future Work

### Limitations:
- **Synthetic Spina Bifida Cranial Deformations**: The positive class is generated via mathematical morphological inward scalloping (`lemonizer.py`) on HC18 scans. While physiologically grounded in the lemon sign, real clinical scans exhibit acoustic shadows, amniotic fluid variations, and fetal positioning occlusions.
- **Proxy Anatomical Severity**: Gold-standard surgical pathology grades (e.g. myeloschisis vs closed meningocele) are not directly annotated in raw public ultrasound benchmarks. Severity scores represent computational morphological proxies.
- **Computational Hardware**: Trained using PyTorch with CPU/CUDA device support.

### Future Work:
1. **Multi-Center Clinical Cohort Validation**: Prospective evaluation on raw, multi-hospital ultrasound video streams.
2. **Direct Spinal Lesion Segmentation**: Incorporating U-Net / SAM (Segment Anything in Medical Images) for direct lumbosacral spinal defect segmentation in addition to cranial markers.
3. **Multi-View 3D Ultrasound Integration**: Combining axial transventricular, transcerebellar, and sagittal spinal planes into a unified 3D vision transformer.
4. **External Clinical Trial Validation**: Formal clinical diagnostic trial validation with maternal-fetal medicine specialists.

---

## 10. Citation & License
This project is licensed under the MIT License.
