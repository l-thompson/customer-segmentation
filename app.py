# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
import plotly.express as px
import shap
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv("Mall_Customers.csv")

# Select relevant features for clustering
X = df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA for 3D visualization
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X_scaled)

# Determine optimal number of clusters using the Elbow method
inertia = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)  # Explicitly set n_init
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

# Plot Elbow curve
plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), inertia, marker='o')
plt.title('Elbow Method for Optimal k')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.show()

# Fit K-Means with optimal k (e.g., 5 based on Elbow plot)
kmeans = KMeans(n_clusters=5, n_init=10, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

# Add cluster labels to the DataFrame
df['Cluster'] = clusters

# Create interactive 3D scatter plot with Plotly
fig = px.scatter_3d(
    df,
    x=X_pca[:, 0],
    y=X_pca[:, 1],
    z=X_pca[:, 2],
    color='Cluster',
    title='Customer Segmentation (3D PCA)',
    labels={'x': 'PCA Component 1', 'y': 'PCA Component 2', 'z': 'PCA Component 3'},
    template='plotly_white'
)
fig.update_layout(
    font=dict(color='black'),
    scene=dict(
        xaxis_title='PCA Component 1',
        yaxis_title='PCA Component 2',
        zaxis_title='PCA Component 3'
    )
)
fig.show()

# Use SHAP for interpretability with a proxy Random Forest model
# Create a supervised proxy problem to predict cluster labels
X_train = X_scaled
y_train = clusters
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Compute SHAP values using KernelExplainer as a fallback
# TreeExplainer produced incorrect shapes, so we use KernelExplainer
print("Computing SHAP values using KernelExplainer (this may take a few minutes)...")
explainer = shap.KernelExplainer(rf.predict_proba, X_train)
# Use a subset of samples to speed up computation (optional)
shap_values = explainer.shap_values(X_train, nsamples=50)

# Debug: Print shapes to diagnose the issue
print("Shape of X_train:", X_train.shape)
print("Number of classes:", len(np.unique(y_train)))
print("SHAP values structure:", [v.shape for v in shap_values])

# Handle SHAP values structure
if len(shap_values) == len(np.unique(y_train)):  # Should be 5 arrays for 5 clusters
    # Use SHAP values for Cluster 0 for beeswarm plot
    print("SHAP Beeswarm Plot for Cluster 0:")
    shap.summary_plot(shap_values[0], X_train, feature_names=['Age', 'Annual Income (k$)', 'Spending Score (1-100)'])

    # Bar plot for average feature importance across Cluster 0
    shap_values_mean_class0 = np.abs(shap_values[0]).mean(axis=0)  # Shape: (3,)
    plt.figure(figsize=(8, 5))
    plt.bar(['Age', 'Annual Income (k$)', 'Spending Score (1-100)'], shap_values_mean_class0)
    plt.title('Average Feature Importance for Cluster 0 (SHAP)')
    plt.xlabel('Features')
    plt.ylabel('Mean |SHAP Value|')
    plt.show()
else:
    # Fallback: Use TreeExplainer and manually aggregate (previous approach)
    print("Falling back to TreeExplainer with manual aggregation...")
    explainer = shap.TreeExplainer(rf)
    shap_values = explainer.shap_values(X_train)
    print("SHAP values structure (TreeExplainer):", [v.shape for v in shap_values])
    
    if len(shap_values) == len(X_train):  # 200 samples, each with (3, 5)
        # Transpose each (3, 5) array to (5, 3) to align features with classes
        shap_values_transposed = [sv.T for sv in shap_values]  # Now (5, 3) per sample
        # Aggregate across samples to get mean SHAP values per class
        shap_values_mean_per_class = np.array([np.mean([sv[i] for sv in shap_values_transposed], axis=0) for i in range(5)])
        print("Aggregated SHAP values shape per class:", shap_values_mean_per_class.shape)  # Should be (5, 3)

        # Bar plot for average feature importance across Cluster 0
        shap_values_mean_class0 = np.abs(shap_values_mean_per_class[0])  # Shape: (3,)
        plt.figure(figsize=(8, 5))
        plt.bar(['Age', 'Annual Income (k$)', 'Spending Score (1-100)'], shap_values_mean_class0)
        plt.title('Average Feature Importance for Cluster 0 (SHAP, TreeExplainer)')
        plt.xlabel('Features')
        plt.ylabel('Mean |SHAP Value|')
        plt.show()
    else:
        print("Unexpected SHAP values structure. Please check SHAP version or model output.")

# Print cluster centers for interpretation
cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)
print("Cluster Centers (unscaled):")
for i, center in enumerate(cluster_centers):
    print(f"Cluster {i}: Age={center[0]:.2f}, Income={center[1]:.2f}k$, Spending Score={center[2]:.2f}")

# Save the DataFrame with clusters for further analysis
df.to_csv("customer_segments.csv", index=False)

# Save the 3D plot as HTML
fig.write_html("customer_segmentation_3d.html")