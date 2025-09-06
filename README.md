# Seperation_Coefficient_app
This project is a Streamlit-based Gaussian Deconvolution and Peak Resolution Analysis Tool.  It is designed for analyzing spectroscopic or chromatographic datasets
# 🧪 Gaussian Deconvolution and Peak Separation Analysis

A **Streamlit app** for Gaussian deconvolution of spectroscopic or chromatographic datasets, with baseline correction and peak resolution analysis.

---

## 🚀 Features
- 📂 Upload CSV or paste data directly (two-column format: Distance, Intensity).
- 🖼️ Upload an optional reference image for each dataset.
- 🔧 Flexible **baseline correction**:
  - Constant offset
  - Linear baseline
  - Polynomial baseline (manual coefficients or auto-fit with endpoint selection)
- 📊 Fit multiple Gaussian peaks automatically.
- 📋 View detailed **peak parameters** (amplitude, center, width, FWHM).
- 📐 Compute the **effective separation coefficient (Cf)**:
  - Formula: `Cf = |Center2 - Center1| / (0.5*(FWHM1 + FWHM2))`
  - Interpretation:
    - Cf > 1.5 → well-separated peaks
    - Cf ≈ 1.0 → moderate overlap
    - Cf < 0.8 → significant overlap

---

## 📦 Installation

Clone this repository and install dependencies:

```bash
git clone https://github.com/yourusername/gaussian-deconv.git
cd gaussian-deconv
pip install -r requirements.txt
