# <p align="center">Space Wanderer</p>

## <p align="center">🌌 Prediction of Stellar/Exoplanetary Parameters</p>

### <p align="center">Summary:</p>
<p align="center">I - Objectives</p>
<p align="center">II - Abstract</p>
<p align="center">III - Plan</p>
<p align="center">IV - Methodology</p>
<p align="center">V - Results</p>

---

### 🎯 Objective
Develop a regression model (neural network : CNN) to predict the physical parameters of stars or exoplanets (e.g., radius, temperature, orbital period, density, etc.) based on light curves or observable parameters from TESS dataset.

---

### 📖 Abstract
The use of deep learning for automatically classifying candidate exoplanet transit signals has been previously explored by Shallue & Vanderburg (2018), who developed a convolutional neural network trained on Kepler data (see also Zucker & Giryes 2018 and Pearson et al. 2018 for applications of neural networks to simulated transit data). High-precision space telescopes provide a massive amount of data, allowing us to explore space. According to Megan Ansdell, co-author of [1], deep learning has proven to be efficient for exoplanet transit classification. Machine learning can identify exoplanets by analyzing telescope data, most of which rely on the transit phenomenon.

🚀 **Here are some key missions:**  
NASA’s **Kepler** (2009–2018) – Detected exoplanets using stellar transits.  
**K2** (2014–2018) – Extension of Kepler with a different field of view.  
**TESS** (Transiting Exoplanet Survey Satellite, since 2018) – Searches for exoplanets near their host stars, sector by sector.  
**CHEOPS** (Characterizing Exoplanets Satellite, since 2019) – Studies exoplanets (size, density).  
**JWST** (James Webb Space Telescope, since 2021) – Analyzes exoplanet atmospheres using infrared.  
**Hubble Space Telescope** (1990–present) – Provides crucial observations of exoplanets.  
**PLATO** (Planetary Transits and Oscillations of Stars, expected 2026) – Designed for high-precision exoplanet detection and habitability studies.  
While current missions focus on discovering new exoplanets, reducing false positives, and analyzing planetary composition, future missions will address habitability and high-precision detection methods.  

🪐 **The Transit Phenomenon**  
The transit method is one of the most effective techniques for detecting exoplanets. It relies on observing a small, periodic dip in a star’s brightness caused by a planet passing in front of it.
Each transit event is characterized by:  
Period: The time between two consecutive transits.  
Epoch: The exact moment a transit is detected.  
Duration: The time during which the star's brightness decreases.  
However, not all transit signals correspond to exoplanets. Many false positives can arise due to :  
Eclipsing binary stars (EBs or BEBs - Background Eclipsing Binaries), where another star mimics a planetary transit.  
Instrumental noise or artifacts, which can create misleading signals.  

☄️ **Motivation**  
TESS, launched in 2018, detects exoplanets by measuring transit signals in stellar flux using The Transit method. It records high-cadence observations of thousands of stars and full-frame images covering 75% of the sky. Machine learning, especially neural networks, helps efficiently filter and analyze this vast dataset.
**This project combines two rapidly evolving fields: exoplanet research in astronomy and deep learning in computer science. Training a deep neural network (CNN) that can adapt to future missions, such as PLATO (2026), will be a crucial step forward in the search for habitability.**

---

## Linux/Mac
```bash
python -m venv venv
source venv/bin/activate
```

## Windows
```bash
python -m venv venv
venv\Scripts\activate    
```

## Requirements
```bash
pip install --upgrade -r requirements.txt
```

---

### 🔣 Methodology

- #### **Scientific Exploration**
🔭 Understanding the physical parameters available in catalogs (e.g., radius, mass, temperature, period, etc.).  
🔍 Studying known relationships between these parameters (e.g., mass-radius, temperature-luminosity correlations).

- #### **Features and Targets**
**Available Data:**  

**Light Curve:** Flux over time.  
**Metadata:** Temperature, magnitude, quality, etc.  

**Standard CNN (Main Features):**  
**Input:** Light curve (time series transformed into an image or sequence).  

**Enhanced CNN (Additional Features):**  
**Input 1:** Light curve (image/sequence).  
**Input 2:** Selected useful metadata (numerical values like TEFF, TESSMAG, LOGG).

**Finally:**  
_Possible inputs:_ summarized/statistical light curve data, magnitude, color, temperature, log(g), etc.  
_Target variables:_ radius, mass, density, etc.

- #### **Data Collection**
📦 Downloading data from space missions:
_TESS_: [TESS ExoFOP](https://exofop.ipac.caltech.edu/tess/)  
_MAST_ portal (TESS, Kepler, K2): [MAST Archive](https://mast.stsci.edu)  
_Lightkurve_ (Python): Easy access to light curves.  
🧾 Enriching metadata with additional catalogs such as TIC or Gaia:  
_TIC_ (TESS Input Catalog): [TIC Archive](https://archive.stsci.edu/hlsp/tic)  

- #### **Data Preparation**
Data cleaning: Handling missing values, normalization.  

- #### **Model Development**
**Baseline classical models:**  
Linear Regression, Random Forest, XGBoost.  
**Deep Learning models:**  
Dense Neural Networks (MLP) for tabular data.
1D CNN for time-series data (raw or summarized light curves).

#### **Evaluation and Interpretation**
📈 Performance metrics: RMSE, MAE, R².
🧠 Model interpretation: SHAP values or permutation importance to identify the most informative variables.

#### **Scientific Contribution**
**Objective:** Identifying original correlations or validating predictions on less-studied objects.  
Providing reusable tools or annotated datasets for future research.

---

### Bibliography

[1] Scientific Domain Knowledge Improves Exoplanet Transit Classification with Deep Learning
