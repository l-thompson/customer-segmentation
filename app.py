# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
import plotly.express as px
import plotly.graph_objects as go
import shap
import matplotlib.pyplot as plt
import os

# Create output directory
os.makedirs('outputs', exist_ok=True)

# Load the dataset
df = pd.read_csv("Mall_Customers.csv")

# Validate data
print("Dataset Overview:")
print(f"Number of records: {len(df)}")
print(f"Columns: {df.columns.tolist()}")
print("\nMissing Values:")
print(df.isnull().sum())
if df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']].isnull().any().any():
    raise ValueError("Missing values detected in required features.")

# Select relevant features for clustering
X = df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA for 3D visualization
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X_scaled)

# Compute PCA loadings
feature_names = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
loadings = pd.DataFrame(
    pca.components_.T,
    columns=[f'PCA Component {i+1}' for i in range(pca.n_components_)],
    index=feature_names
)
print("\nPCA Loadings (Feature Contributions to Components):")
print(loadings.round(3))
loadings.to_csv('outputs/pca_loadings.csv')

# Assign axis labels based on dominant features, avoiding duplicates
dominant_features = []
used_features = set()
for i in range(pca.n_components_):
    component_loadings = loadings.iloc[:, i].abs()
    available_features = component_loadings.index.difference(used_features)
    if available_features.empty:
        dominant_features.append(f'Mixed Features (PCA {i+1})')
    else:
        dominant_feature = component_loadings[available_features].idxmax()
        dominant_features.append(f'Mostly {dominant_feature}')
        used_features.add(dominant_feature)
axis_labels = {
    'x': dominant_features[0],
    'y': dominant_features[1],
    'z': dominant_features[2]
}

# Determine optimal number of clusters using the Elbow method
inertia = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

# Plot Elbow curve with arrow
plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), inertia, marker='o')
plt.title('Elbow Method for Optimal Number of Clusters', fontsize=16)
plt.xlabel('Number of Clusters (k)', fontsize=12)
plt.ylabel('Inertia (Within-Cluster Sum of Squares)', fontsize=12)
plt.annotate('Optimal k=5', xy=(5, inertia[4]), xytext=(5, inertia[4] + 50),
             arrowprops=dict(facecolor='red', shrink=0.05), fontsize=10)
plt.grid(True)
plt.savefig('outputs/elbow_plot.png')
plt.show()

# Fit K-Means with optimal k (5)
kmeans = KMeans(n_clusters=5, n_init=10, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

# Add cluster labels to the DataFrame
df['Cluster'] = clusters

# Create interactive 3D scatter plot with Plotly
pca_df = pd.DataFrame({
    'x': X_pca[:, 0],
    'y': X_pca[:, 1],
    'z': X_pca[:, 2],
    'Cluster': clusters,
    'Age': df['Age'],
    'Income': df['Annual Income (k$)'],
    'Spending': df['Spending Score (1-100)']
})
fig = px.scatter_3d(
    pca_df,
    x='x',
    y='y',
    z='z',
    color='Cluster',
    title='Customer Segments by Age, Income, and Spending (PCA-Reduced)',
    labels=axis_labels,
    hover_data={
        'Age': ':.0f',
        'Income': ':.0f',
        'Spending': ':.0f',
        'Cluster': True
    },
    template='plotly_white',
    size_max=10
)

# Add cluster centers in PCA space
centers_scaled = kmeans.cluster_centers_
centers_unscaled = scaler.inverse_transform(centers_scaled)
centers_pca = pca.transform(centers_scaled)
for i, (center, unscaled) in enumerate(zip(centers_pca, centers_unscaled)):
    fig.add_trace(go.Scatter3d(
        x=[center[0]], y=[center[1]], z=[center[2]],
        mode='markers+text',
        text=[f'Cluster {i} Center'],
        textposition='top center',
        marker=dict(size=10, color='red', symbol='diamond'),
        hovertemplate=(
            f'Cluster {i} Center<br>' +
            f'Age: {unscaled[0]:.0f}<br>' +
            f'Income: {unscaled[1]:.0f}k$<br>' +
            f'Spending Score: {unscaled[2]:.0f}'
        ),
        showlegend=False
    ))

fig.update_layout(
    font=dict(color='black'),
    scene=dict(
        xaxis_title=axis_labels['x'],
        yaxis_title=axis_labels['y'],
        zaxis_title=axis_labels['z']
    ),
    margin=dict(l=0, r=0, b=0, t=50)
)

# Save 3D plot as HTML and PNG
try:
    fig.write_html('outputs/customer_segmentation_3d.html')
    try:
        fig.write_image('outputs/customer_segmentation_3d.png')
    except ImportError as e:
        print(f"Error saving 3D plot as PNG: {e}")
        print("Install 'kaleido' for PNG export: pip install kaleido")
except Exception as e:
    print(f"Error saving 3D plot: {e}")

fig.show()

# Use SHAP for interpretability with a proxy Random Forest model
X_train = X_scaled
y_train = clusters
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Debug: Print shapes and model details before SHAP computation
print("\n--- Pre-SHAP Computation Debug ---")
print("Shape of X_train:", X_train.shape)
print("Number of classes in y_train:", len(np.unique(y_train)))
print("Random Forest classes:", rf.classes_)
print("Feature names:", feature_names)
print("Sample of X_train (first 2 rows):", X_train[:2])
print("Sample of y_train (first 5 labels):", y_train[:5])

# Compute SHAP values using TreeExplainer
print("\n--- Computing SHAP Values ---")
explainer = shap.TreeExplainer(rf)
shap_values = explainer.shap_values(X_train)

# Debug: Print SHAP values structure
print("SHAP values type:", type(shap_values))
print("Length of shap_values:", len(shap_values))
if isinstance(shap_values, list):
    print("SHAP values shapes:", [v.shape for v in shap_values])
    print("Sample of shap_values for first element:", shap_values[0])
else:
    print("SHAP values shape:", shap_values.shape)
    print("Sample of shap_values (first 2 samples, all classes):", shap_values[:2])

# Handle SHAP values for multi-class
if isinstance(shap_values, np.ndarray) and shap_values.ndim == 3 and shap_values.shape[1:] == (3, 5):
    print("\n--- Handling SHAP Values (200, 3, 5) ---")
    print("SHAP values structure confirmed: samples × features × classes")
    shap_values_class0 = shap_values[:, :, 0]  # Shape: (200, 3)
    print("SHAP values for Cluster 0 shape:", shap_values_class0.shape)
    print("Sample of SHAP values for Cluster 0 (first 2 rows):", shap_values_class0[:2])
    shap_values_mean_class0 = np.abs(shap_values_class0).mean(axis=0)  # Shape: (3,)
    print("\n--- Plotting Feature Importance for Cluster 0 ---")
    print("Mean absolute SHAP values for Cluster 0:", shap_values_mean_class0)
    plt.figure(figsize=(8, 5))
    plt.bar(feature_names, shap_values_mean_class0)
    plt.title('Average Feature Importance for Cluster 0 (SHAP)')
    plt.xlabel('Features')
    plt.ylabel('Mean |SHAP Value|')
    plt.savefig('outputs/shap_bar_plot.png')
    plt.show()
else:
    print("\n--- Unexpected SHAP Structure ---")
    print("Expected a (200, 3, 5) array. Received:", shap_values.shape if isinstance(shap_values, np.ndarray) else "List with shapes:", [v.shape for v in shap_values])
    print("Please check SHAP version or model output.")

# Print and save cluster centers
cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)
cluster_summary = pd.DataFrame(
    cluster_centers,
    columns=['Age', 'Annual Income (k$)', 'Spending Score (1-100)'],
    index=[f'Cluster {i}' for i in range(len(cluster_centers))]
)
print("\n--- Cluster Centers (Unscaled) ---")
print(cluster_summary.round(2))
cluster_summary.to_csv('outputs/cluster_summary.csv')

# Save the DataFrame with clusters
df.to_csv("outputs/customer_segments.csv", index=False)