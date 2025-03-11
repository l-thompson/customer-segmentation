# Customer Segmentation with K-Means and SHAP Visualizations

This project demonstrates customer segmentation using unsupervised machine learning (K-Means clustering) on the Mall Customers dataset. It includes advanced visualizations with Plotly and feature importance analysis using SHAP (SHapley Additive exPlanations), showcasing a full data science pipeline from preprocessing to interpretation.

## Project Overview

The goal of this project is to segment customers into distinct groups based on their demographic and behavioral data (Age, Annual Income, and Spending Score). The project uses K-Means clustering to identify 5 customer segments, visualizes the results in an interactive 3D scatter plot, and interprets the clustering decisions using SHAP to understand which features drive the segmentation.

### Key Features
- **Unsupervised Learning**: Applies K-Means clustering to segment customers into 5 groups.
- **Advanced Visualization**: Creates an interactive 3D scatter plot using Plotly to visualize customer segments.
- **Feature Importance**: Uses SHAP to explain which features (Age, Income, Spending Score) most influence cluster assignments.
- **End-to-End Workflow**: Covers data preprocessing, modeling, visualization, and interpretation.

### Technologies Used
- **Python**: Core programming language.
- **scikit-learn**: For K-Means clustering, PCA, and Random Forest (proxy model for SHAP).
- **Plotly**: For interactive 3D scatter plots.
- **SHAP**: For feature importance analysis.
- **Matplotlib**: For static bar plots.
- **pandas & numpy**: For data manipulation.

## 📊 Dataset

The dataset (`Mall_Customers.csv`) contains 200 customer records with the following columns:
- `CustomerID`: Unique identifier for each customer.
- `Gender`: Customer gender (Male/Female).
- `Age`: Customer age in years.
- `Annual Income (k$)`: Annual income in thousands of dollars.
- `Spending Score (1-100)`: Score assigned by the mall based on customer spending behavior (1 to 100).

**Note**: The `Gender` column is not used in clustering but could be analyzed post-clustering for deeper insights.

## Results

### Cluster Centers
The K-Means algorithm identified 5 distinct customer segments:
- **Cluster 0**: Age=55.64, Income=54.38k$, Spending Score=48.85 (Older, moderate income, moderate spenders)
- **Cluster 1**: Age=32.88, Income=86.10k$, Spending Score=81.53 (Young, high-income, high spenders)
- **Cluster 2**: Age=25.19, Income=41.09k$, Spending Score=62.24 (Young, moderate-income, moderate-high spenders)
- **Cluster 3**: Age=46.25, Income=26.75k$, Spending Score=18.35 (Middle-aged, low-income, low spenders)
- **Cluster 4**: Age=39.87, Income=86.10k$, Spending Score=19.36 (Middle-aged, high-income, low spenders)

### Visualizations
- **3D Scatter Plot**: An interactive 3D scatter plot (`customer_segmentation_3d.html`) shows the customer segments in PCA-reduced space.
- **SHAP Bar Plot**: A bar plot displays the average feature importance for Cluster 0, highlighting which features (e.g., Spending Score) most influence cluster assignment.

## Setup

### Prerequisites
- Python 3.7 or higher
- pip (Python package manager)

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/customer-segmentation.git
   cd customer-segmentation
