For this project, I aim to predict the market value of football players using the FIFA 21 dataset available on Kaggle.
This dataset contains thousands of players and provides detailed information such as age, overall rating, potential, position, and club. 
The main goal is to build a machine learning model that can estimate how much a player is worth based on their characteristics.
The first step of the project will be data cleaning. 
I will remove unnecessary columns such as player IDs, image links, and metadata that are not useful for prediction. 
I will also handle missing values by removing or replacing incomplete rows, ensuring the dataset is clean and consistent.
Next, I will focus on selecting the most relevant variables for predicting player value. 
These include:

-Age, as younger players with strong potential tend to have higher market values;
-Overall rating, which reflects a player’s current ability;
-Potential, indicating the player’s possible future development;
-Position, since the value often varies depending on whether the player is an attacker, midfielder, defender, or goalkeeper.

I will also create new simple variables, such as the difference between potential and overall rating, to capture the player’s growth potential.
Once the dataset is ready, I will conduct exploratory data analysis (EDA) to better understand the relationships between variables.
I plan to use histograms to visualize distributions, scatterplots to explore the link between overall or potential and value, boxplots to compare value across positions, and a heatmap to show correlations among variables.
Finally, I will train several regression models, including Linear Regression, Ridge Regression, and Random Forest, and evaluate their performance using R² and RMSE metrics. This comparison will help determine which model best predicts player market value.
In conclusion, this project aims to identify the key features that influence a football player’s market value and to build a simple yet effective predictive model based on the FIFA 21 dataset.