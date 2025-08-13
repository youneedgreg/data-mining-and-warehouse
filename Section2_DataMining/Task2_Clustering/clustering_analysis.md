
# Clustering Analysis Report

## Overview
K-Means clustering was applied to the Iris dataset to identify natural groupings without using class labels.

## Results
- **Optimal k=3**: Aligns with the three known Iris species
- **Adjusted Rand Index: 0.73**: Substantial agreement with true species
- **Silhouette Score: 0.55**: Well-separated clusters
- **Accuracy: ~89%**: When optimally mapping cluster labels to species

## Key Findings
1. Setosa is perfectly separated (100% accuracy)
2. Versicolor and Virginica show overlap (~10-15% misclassification)
3. This reflects biological reality - these species are morphologically similar

## Real-World Applications
1. **Customer Segmentation**: Group customers by purchasing behavior
2. **Product Categorization**: Automatic inventory grouping
3. **Anomaly Detection**: Identify outliers for quality control
4. **Image Segmentation**: Medical imaging analysis

## Conclusion
K-Means successfully identifies the natural structure in the Iris dataset with high accuracy.
