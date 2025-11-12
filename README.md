# 🧠 IndustryForecast: Machine Learning–Based Industrial Workforce Prediction

This project analyzes changes in regional industrial structures using real-world employment data from 2017 to 2022.  
By combining statistical analysis and machine learning, it predicts the direction of industrial growth and decline for 2023.  
Chi-square tests were used to verify that industrial compositions differ significantly between population-increasing and population-decreasing regions.  
Subsequently, Random Forest and K-Nearest Neighbor classifiers were trained to model these regional trends, and a soft-voting ensemble achieved the highest accuracy.  

Results show that:
- **Population-decreasing regions** are dominated by **construction (26%)**, **health & social welfare (16%)**, and **manufacturing (12%)**.  
- **Population-increasing regions** are led by **manufacturing (18%)**, **professional & technical services (13%)**, and **retail trade (13%)**.  
- The model achieved an accuracy of **83%**, with strong generalization in predicting industrial growth patterns.  

These findings provide an interpretable foundation for **regional policy planning** and **industrial strategy design** based on workforce transitions.

--- 
## 🧰 Technologies Used
**Python**, **scikit-learn**, **pandas**, **matplotlib**, **seaborn**, **NumPy**

---
## 📂 Project Structure
```
├── dataset/                            # Raw and intermediate CSV data files
├── graphfolder/                        # Generated industry distribution graphs by region
├── NanumGodic/                         # Korean font file (NanumGothic.ttf)
│
├── df_industry.csv                     # Processed dataset with labeling results
├── industry_distribution_analysis.py   # Regional industry ratio visualization & Chi-square analysis
├── MLmodel.py                          # Machine learning model (RF + KNN Voting Classifier)

```
---

## 🚀 How to Run
### Clone the repository
```bash
git clone https://github.com/daewook1004/IndustryForecast.git
cd IndustryForecast
```

### Install dependencies

```
pip install pandas numpy matplotlib seaborn scikit-learn
```
### Run the industry distribution analysis

This script performs:
- Regional 2022 industry ratio calculations
- Automatic graph generation by region
- Chi-square test comparing increasing vs decreasing population regions

```
python industry_distribution_analysis.py
```
Output:
- df_industry.csv (with labeling and ratios)
- graphfolder/*.png (region-wise bar charts)
- graphfolder/decrease_increase_2022_industry_distribution.png

### Run the machine learning model
This script trains RandomForest and KNN models to predict 2023 trends.
```
python MLmodel.py
```
Output:
- Model accuracy and confusion matrices printed in console
- Predicted labels for 2023 saved in df_goal.csv
---
## Example Results
### Example Results










