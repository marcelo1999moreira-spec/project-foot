Football Player Market Value Prediction

"Which model can predict better the value of a player in the FIFA 21 dataset?"
This project aims to predict professional football players’ market values using
data from the FIFA 21 dataset.
It combines exploratory data analysis, feature engineering, and machine learning
models to estimate player value and analyze the key drivers behind it.
\*\*\*SETUP
You go to GitHub, in my repository https://github.com/marcelo1999moreira-spec/Projet-foot.git, you copy the URL.
You download the ZIP, there you can see the results.
After you go to the terminal, and you do:

git clone https://github.com/marcelo1999moreira-spec/Projet-foot.git
cd Projet-foot

\*\*\*CREATE ENVIRONMENT

conda env create -f environment.yml
conda activate player-value
python main.py
Expected output: ✅ Pipeline finished successfully.

\*\*\*PROJECT STRUCTURE
Projet-foot/

<br>
├── main.py # Main pipeline entry point
├── README.md # Project documentation
├── PROPOSAL.md # Initial project proposal
├── environment.yml # Conda environment definition
├── requirements.txt # Python dependencies (pip)

<br>
├── data/
│ └── raw/
│ └── players\_21.csv # Raw FIFA 21 dataset

<br>
├── src/
│ ├── <strong>init</strong>.py
│ ├── data\_loader.py # Data loading functions
│ ├── features.py # Feature engineering
│ ├── eda.py # Exploratory Data Analysis (EDA)
│ ├── modeling.py # Preprocessing and model definitions
│ ├── training.py # Model training and selection
│ ├── evaluation.py # Error analysis & feature importance
│ └── prediction.py # Demo predictions

<br>
├── results/
│ ├── figures/ # Saved plots and visualizations
│ └── models/ # Model outputs and evaluation tables

\*\*\*RESULTS

Model: Linear Regression
CV RMSE (mean): 3.6955 M€ (+/- 0.3226)
Test MAE : 2.0243 M€
Test RMSE : 3.8926 M€
Test R² : 0.4661

Model: Ridge Regression
CV RMSE (mean): 3.6955 M€ (+/- 0.3226)
Test MAE : 2.0242 M€
Test RMSE : 3.8925 M€
Test R² : 0.4661

Model: Lasso Regression
CV RMSE (mean): 3.6955 M€ (+/- 0.3227)
Test MAE : 2.0233 M€
Test RMSE : 3.8916 M€
Test R² : 0.4663

Model: ElasticNet 
CV RMSE (mean): 3.6955 M€ (+/- 0.3227)
Test MAE : 2.0234 M€
Test RMSE : 3.8919 M€
Test R² : 0.4662

Model: Decision Tree 
CV RMSE (mean): 1.2405 M€ (+/- 0.2219)
Test MAE : 0.2755 M€
Test RMSE : 1.0243 M€
Test R² : 0.9630

Model: Random Forest 
CV RMSE (mean): 0.8758 M€ (+/- 0.1559)
Test MAE : 0.1370 M€
Test RMSE : 0.6078 M€
Test R² : 0.9870

 Model: Gradient Boosting 
CV RMSE (mean): 0.7802 M€ (+/- 0.1036)
Test MAE : 0.1749 M€
Test RMSE : 0.5672 M€
Test R² : 0.9887

\*\*\*REQUIREMENTS

PYTHON: 3.11
pandas, numpy, matplotlib, seaborn, scikit-learn, joblib