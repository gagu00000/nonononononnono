🛒 NovaMart Marketing Analytics Dashboard

A Streamlit-powered analytics platform for campaign performance, customer behavior, product sales, attribution modeling, and ML evaluation.

This dashboard is designed for NovaMart’s executive team and provides 20+ interactive visualizations across marketing, customer, product, geographic, funnel, and machine-learning insights.

🚀 Features
📊 Marketing & Campaign Analytics

Revenue trends

Channel performance

Regional comparisons

Campaign type spend distribution

Calendar (GitHub-style) heatmap

👥 Customer Insights

Age distribution

LTV by customer segment

Satisfaction score violin plots

Scatter & bubble relationship charts

🛍️ Product Performance

Treemap (Category → Subcategory → Product)

Margin-based color coding

Regional product analytics

🌍 Geographic Analysis

Choropleth map (State-wise revenue, customers, penetration)

Bubble map (Store performance)

🔄 Attribution & Funnel

Multi-model attribution donut chart

Full marketing funnel visualization

Correlation heatmap

🤖 ML Model Evaluation

Confusion matrix

ROC curve

Learning curve

Feature importance with error bars

📁 File Structure
/repo
│── app.py
│── requirements.txt
│── README.md
│── .streamlit/runtime.txt
│── campaign_performance.csv
│── channel_attribution.csv
│── correlation_matrix.csv
│── customer_data.csv
│── customer_journey.csv
│── feature_importance.csv
│── funnel_data.csv
│── geographic_data.csv
│── learning_curve.csv
│── lead_scoring_results.csv
│── product_sales.csv


📌 All CSV files must be in the same folder as app.py (root directory).

🛠️ Installation

Clone your GitHub repository:

git clone https://github.com/your-username/your-repo.git
cd your-repo


Install dependencies:

pip install -r requirements.txt


Run the Streamlit app:

streamlit run app.py

🌐 Deploying on Streamlit Cloud

Commit all files to GitHub (including all CSVs in the root folder)

Go to: https://streamlit.io/cloud

Click New App

Select your GitHub repo

Set:

Main file: app.py

Python version: 3.10+

Dependencies: auto-detected from requirements.txt

🎉 Your dashboard will deploy automatically.

📝 Notes

Make sure all 11 CSV files remain next to app.py (no /data folder needed)

The app includes caching for fast performance

Plotly ensures responsive, board-ready interactive visualizations
