<h1 align="center" style="font-size: 2.5rem; color: #1a73e8; margin-bottom: 10px;">
  🌌 Higgs Boson EDA Pipeline
</h1>

<p align="center" style="font-size: 1.15rem; color: #444;">
  A clean, minimal, and well-structured Exploratory Data Analysis pipeline<br>
  for the <strong>UCI Higgs Boson Dataset</strong>.
</p>

<hr style="border: 0; border-top: 1px solid #ddd; margin: 30px 0;">


<h2 style="color:#1a73e8;">📌 Project Overview</h2>
<p style="font-size: 1rem;">
This repository provides a reproducible end-to-end EDA workflow for the 
<strong>HIGGS dataset</strong>, a large-scale physics dataset used for binary classification 
(background vs. signal events).  
<br><br>
The project contains three main components:
</p>

<ul style="margin-left: 20px; font-size: 1rem;">
  <li><strong>📁download the dataset from UCI machine learning repository<a href="https://mlphysics.ics.uci.edu/data/higgs/HIGGS.csv.gz" target="_blank">
  Download HIGGS Dataset
</a>
  </li>
  <li><strong>📁 higgs_loader.py</strong> — Efficiently loads the dataset, assigns meaningful column names, and prints basic info.</li>
  <li><strong>📊 EDA_UCI_HIGGS_DATASET.py</strong> — Performs full Exploratory Data Analysis, generates plots, summaries, and saves all outputs automatically.</li>
</ul>

<p style="font-size: 1rem;">
The goal of this project is to help students & beginners understand:
</p>

<ul style="margin-left: 20px;">
  <li>How to load large datasets efficiently</li>
  <li>How to build a clean EDA pipeline</li>
  <li>How to analyze features, correlations, distributions, and patterns</li>
  <li>How to generate meaningful plots for ML model development</li>
</ul>


<h2 style="color:#1a73e8;">🚀 Run the EDA Pipeline</h2>

<p>Default execution (loads 500,000 rows):</p>
<pre style="
background:#111827; color:#e5e7eb; padding:15px; border-radius:8px;">
python EDA_UCI_HIGGS_DATASET.py
</pre>

<p>Load a custom number of rows:</p>
<pre style="
background: #17; color:#e5e7eb; padding:15px; border-radius:8px;">
python EDA_UCI_HIGGS_DATASET.py --nrows 100000
</pre>


<h2 style="color:#1a73e8;">📦 Installation Requirements</h2>

<pre style="
background:#f3f4f6; padding:12px; border-radius:8px; font-size:0.95rem;">
pip install pandas numpy matplotlib seaborn
</pre>


<h2 style="color:#1a73e8;">📁 Output Directory Structure</h2>

<pre style="
background:#f3f4f6; padding:12px; border-radius:8px; font-size:0.95rem;">
results/
└── eda/
    ├── 0_info.txt
    ├── 1_describe.csv
    ├── 2_missing_values.csv
    ├── 1_target_distribution.png
    ├── 2_feature_histograms.png
    ├── 3_correlation_matrix.png
    ├── 4_missing_value_heatmap.png
    ├── 5_corr_with_target.png
    ├── 6_kde_plots.png
    ├── 7_boxplots_by_label.png
    ├── 8_pairplot_sample.png
    ├── 9_feature_variance.png
    ├── 10_jointplot_scatter.png
    ├── 11_outlier_iqr.png
    └── 12_summary_overview.csv
</pre>


<h2 style="color:#1a73e8;">📊 Key EDA Insights Provided</h2>

<ul style="margin-left: 20px; line-height: 1.6;">
  <li>Distribution of target (signal vs background)</li>
  <li>Histogram comparison of important physics features</li>
  <li>Correlation heatmap to detect feature relationships</li>
  <li>KDE smooth distributions for deeper visual analysis</li>
  <li>Boxplots to detect class-wise differences & outliers</li>
  <li>Pairplots for multivariate exploration</li>
  <li>Variance ranking for ML feature selection</li>
  <li>Scatterplots to observe feature interactions</li>
  <li>IQR-based outlier detection</li>
</ul>


<h2 style="color:#1a73e8;">📘 About This Dataset</h2>

<p style="font-size: 1rem;">
The <strong>HIGGS</strong> dataset contains 11 million samples and 28 features, 
representing low-level and high-level physics measurements from proton–proton collision simulations.<br>
It is widely used for:
</p>

<ul style="margin-left: 20px;">
  <li>Machine Learning classification benchmarks</li>
  <li>Physics-informed ML research</li>
  <li>High-Energy Particle Physics (HEP) studies</li>
</ul>


<h2 style="color:#1a73e8;">👤 Author</h2>
<p style="font-size: 1.1rem;">
<strong>Aather Nabi Shiekh</strong><br>
AI/ML & Data Science Enthusiast<br>
Email: <a href="mailto:nabiaatir1@gmail.com">nabiaatir1@gmail.com
</a>  
</p>

<hr style="border: 0; border-top: 1px solid #ddd; margin: 40px 0;">
<p align="center" style="font-size: 0.9rem; color:#777;">
  Built with 💙 for learning, exploration & high-energy physics.
</p>
