# NeuroScript AI 🧠
> *From handwritten strokes to intelligent recognition.*

**NeuroScript AI** is an end-to-end deep learning application designed to recognize handwritten digits (0–9) and English alphabet characters (A–Z) from an interactive drawing canvas or uploaded images. The system implements, trains, and rigorously benchmarks both feedforward **Artificial Neural Networks (ANN)** and **Convolutional Neural Networks (CNN)** on real labeled datasets (MNIST and EMNIST Letters).

---

## 🌟 Key Features

1. **Interactive Recognition Workspace**:
   - **Freehand Drawing Canvas**: Draw digits or letters directly on the canvas with adjustable stroke thickness and instant preprocessing.
   - **Image Upload**: Upload PNG, JPG, or JPEG images with automated centering, bounding-box cropping, and polarity normalization.
   - **Live Preprocessing Preview**: Displays the exact 28×28 centered matrix processed by the neural network.
   - **Top-5 Confidence Distribution**: Interactive Plotly probability charts and sub-millisecond inference latency telemetry.

2. **Genuine ANN & CNN Architectures**:
   - **ANN (Multilayer Perceptron)**: Vectorized 784-dimensional flat input, multi-tier dense layers with ReLU activation, Batch Normalization, and Dropout.
   - **CNN (Convolutional Network)**: Dual convolution blocks (32 and 64 3×3 filters), Batch Normalization, ReLU activations, Max-Pooling (28×28 → 14×14 → 7×7), and dense classification head.

3. **Interactive Model Training Center**:
   - Live telemetry with real-time epoch loss and accuracy convergence curves.
   - Configurable hyperparameters: epochs, batch size, learning rate, validation split, and CPU subsampling.
   - Checkpoint persistence in Keras 3 format with companion metadata.

4. **Rigorous Model Comparison Center**:
   - Direct head-to-head empirical benchmarking on identical held-out test splits.
   - Accuracy, Macro Precision, Recall, Macro F1, training duration, and latency comparisons.
   - Side-by-side interactive confusion matrix heatmaps.
   - Automated scientific synthesis explaining observed variance.

5. **Deep Feature Visualizer**:
   - **CNN**: Inspect genuine 3×3 learned convolution kernels and extracted intermediate feature maps across all conv and pooling layers.
   - **ANN**: Visualize layer topologies and weight magnitude distributions.

6. **Dataset Management & Explorer**:
   - Explore MNIST Digits (10 classes: 0–9) and EMNIST Letters (26 classes: A–Z).
   - Class frequency distribution histograms and authentic sample image galleries.
   - Step-by-step LeCun standardization visualizer.

7. **Audit Logging & Report Exporter**:
   - Persistent SQLite history database (`predictions.db`) with thumbnail storage.
   - Export training curves (CSV), metrics (JSON), confusion matrices (CSV & PNG), and printable multi-page PDF evaluation reports.

---

## 🛠️ Technology Stack

- **Python 3.11**
- **Streamlit**: Web workspace & reactive dashboard
- **TensorFlow / Keras 3**: Neural network modeling & GPU/CPU inference
- **NumPy & Pandas**: Array manipulations & tabular audit records
- **OpenCV & Pillow**: LeCun centering, contour bounding-box extraction, and polarity normalization
- **scikit-learn**: Classification reports, macro/weighted F1, and confusion matrices
- **Plotly**: Dynamic charts and interactive heatmaps
- **Matplotlib**: Confusion matrix rendering
- **streamlit-drawable-canvas**: Interactive drawing canvas
- **Pytest**: Automated unit test suite
- **fpdf2**: Formal PDF report generation

---

## 📐 Preprocessing & Label Handling

### Coordinate Transposition (EMNIST Letters)
Raw EMNIST binary data is stored in column-major orientation by NIST. NeuroScript AI transposes axes `(0, 2, 1)` during parsing to ensure characters are correctly upright.

### Label Normalization
- **MNIST Digits**: 10 classes (`0` to `9`), 0-indexed.
- **EMNIST Letters**: Raw NIST labels are 1-indexed (`1=A` to `26=Z`). NeuroScript AI normalizes them to 0-indexed targets (`0=A` to `25=Z`).
- **Combined Mode**: 36 classes with digits at indices 0–9 and letters at indices 10–35.

### LeCun Image Standardization
Input strokes (canvas drawings or image uploads) undergo:
1. Grayscale conversion.
2. Background polarity detection and inversion (ensuring white strokes on black canvas).
3. Contour bounding box extraction (tight crop around strokes).
4. Aspect-ratio preserving scaling to a 20×20 bounding box.
5. Canvas centering inside a 28×28 frame using center-of-mass refinement.
6. Normalization to $[0.0, 1.0]$ float32 tensors.

---

## 🚀 Quickstart & Installation (Windows)

### 1. Prerequisites
Ensure Python 3.11 is installed.

### 2. Activate Virtual Environment
```powershell
# Navigate to project root
cd d:\DIGILAB

# Activate the virtual environment
.\venv\Scripts\Activate.ps1
```

### 3. Install Dependencies
```powershell
pip install -r requirements.txt
```

### 4. Run Automated Tests
```powershell
pytest -v tests
```

### 5. Launch the Application
```powershell
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 📁 Project Structure

```
d:\DIGILAB\
├── app.py                     # Main Streamlit application entrypoint
├── requirements.txt           # Dependency specifications
├── pytest.ini                 # Pytest configuration
├── README.md                  # Comprehensive documentation
├── .gitignore                 # Version control exclusion rules
│
├── config/
│   └── settings.py            # Global paths, theme palette, and constants
│
├── data/
│   ├── labels.py              # LabelMapper & class conventions
│   ├── loaders.py             # Unified dataset loading & caching
│   ├── emnist_loader.py       # EMNIST Letters IDX parser & orientation fix
│   ├── preprocessing.py       # LeCun centering & polarity normalization
│   └── validation.py          # Input image validation
│
├── models/
│   ├── ann.py                 # Artificial Neural Network architecture
│   ├── cnn.py                 # Convolutional Neural Network architecture
│   ├── model_factory.py       # Factory pattern for model generation
│   └── model_registry.py      # Checkpoint manager & metadata catalog
│
├── training/
│   ├── trainer.py             # Training orchestrator & dataset preparation
│   ├── callbacks.py           # Real-time Streamlit epoch callback
│   └── evaluation.py          # Metric calculations & confusion matrix generator
│
├── inference/
│   └── predictor.py           # End-to-end inference & top-k ranking engine
│
├── storage/
│   ├── history.py             # SQLite prediction audit log
│   └── experiment_store.py    # JSON experiment store & curves tracker
│
├── reports/
│   └── exporters.py           # CSV, JSON, PNG, and PDF exporters
│
├── ui/
│   ├── theme.py               # Custom CSS & dark theme design system
│   ├── components.py          # Reusable metric cards, Plotly charts
│   └── pages/
│       ├── overview.py        # Landing dashboard
│       ├── recognize.py       # Drawing canvas & upload workspace
│       ├── model_comparison.py# Head-to-head ANN vs CNN benchmark
│       ├── training_center.py # Live model training interface
│       ├── dataset_explorer.py# Split statistics & sample gallery
│       ├── model_visualizer.py# Kernel weights & intermediate feature maps
│       ├── prediction_history.py # SQLite history search & CSV export
│       ├── reports.py         # Download center & PDF report generator
│       └── settings.py        # System diagnostics & cache controls
│
├── tests/
│   ├── test_labels.py         # Label convention & conversion tests
│   ├── test_preprocessing.py  # Image standardization tests
│   ├── test_models.py         # ANN & CNN forward pass tests
│   ├── test_inference.py      # Predictor pipeline tests
│   ├── test_history.py        # SQLite database tests
│   └── test_metrics.py        # Evaluation computation tests
│
└── scripts/
    └── train_baseline_models.py # Script to train initial models
```

---

## 🔬 Dataset Attribution & Licensing

- **MNIST**: Yann LeCun, Corinna Cortes, Christopher J.C. Burges (Creative Commons Attribution-Share Alike 3.0).
- **EMNIST**: Gregory Cohen, Saeed Afshar, Jonathan Tapson, André van Schaik (NIST Open Research Data).

---

## 🛡️ Privacy Guarantee

All image processing, neural model training, and inference calculations run **strictly on-device**. No images or strokes are transmitted to any cloud or third-party service.
