"""
Data Preprocessing and Exploration for Iris Dataset
Section 2, Task 1: Data Preprocessing and Exploration (15 Marks)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

class IrisPreprocessor:
    """Preprocessing and Exploration for Iris Dataset"""
    
    def __init__(self, use_synthetic=False, seed=42):
        self.seed = seed
        self.use_synthetic = use_synthetic
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        np.random.seed(seed)
        
    def generate_synthetic_iris(self) -> pd.DataFrame:
        """Generate synthetic data mimicking Iris dataset"""
        print("Generating synthetic Iris-like dataset...")
        
        np.random.seed(self.seed)
        
        # Generate 150 samples (50 per class) with 4 features
        n_samples_per_class = 50
        
        # Class 0 (Setosa-like): smaller measurements
        class_0 = np.random.normal(loc=[5.0, 3.4, 1.5, 0.2], 
                                  scale=[0.35, 0.38, 0.17, 0.10], 
                                  size=(n_samples_per_class, 4))
        
        # Class 1 (Versicolor-like): medium measurements
        class_1 = np.random.normal(loc=[5.9, 2.8, 4.3, 1.3], 
                                  scale=[0.51, 0.31, 0.47, 0.20], 
                                  size=(n_samples_per_class, 4))
        
        # Class 2 (Virginica-like): larger measurements
        class_2 = np.random.normal(loc=[6.5, 3.0, 5.5, 2.0], 
                                  scale=[0.63, 0.32, 0.55, 0.27], 
                                  size=(n_samples_per_class, 4))
        
        # Combine all classes
        X = np.vstack([class_0, class_1, class_2])
        y = np.array([0]*n_samples_per_class + [1]*n_samples_per_class + [2]*n_samples_per_class)
        
        # Create DataFrame
        df = pd.DataFrame(X, columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])
        df['species'] = y
        df['species_name'] = df['species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})
        
        # Add some missing values for demonstration (2% of data)
        n_missing = int(0.02 * len(df) * 4)
        for _ in range(n_missing):
            row = np.random.randint(0, len(df))
            col = np.random.choice(['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])
            df.loc[row, col] = np.nan
        
        print(f"Generated {len(df)} samples with {df.isnull().sum().sum()} missing values")
        
        return df
    
    def load_data(self) -> pd.DataFrame:
        """Load Iris dataset or generate synthetic data"""
        print("\n" + "="*60)
        print("STEP 1: LOADING DATA")
        print("="*60)
        
        if self.use_synthetic:
            self.df = self.generate_synthetic_iris()
        else:
            # Load from sklearn
            iris = load_iris()
            self.df = pd.DataFrame(iris.data, columns=iris.feature_names)
            self.df['species'] = iris.target
            self.df['species_name'] = pd.Categorical.from_codes(iris.target, iris.target_names)
            
            # Clean column names
            self.df.columns = [col.replace(' (cm)', '').replace(' ', '_') for col in self.df.columns]
        
        print(f"Dataset loaded successfully!")
        print(f"Shape: {self.df.shape}")
        print(f"Columns: {list(self.df.columns)}")
        
        return self.df
    
    def preprocess_data(self) -> pd.DataFrame:
        """Preprocess the data: handle missing values, normalize, encode"""
        print("\n" + "="*60)
        print("STEP 2: PREPROCESSING")
        print("="*60)
        
        # 1. Check for missing values
        print("\n1. Checking for missing values:")
        missing = self.df.isnull().sum()
        print(missing[missing > 0] if missing.sum() > 0 else "No missing values found")
        
        # 2. Handle missing values (if any)
        if self.df.isnull().sum().sum() > 0:
            print("\n2. Handling missing values with mean imputation...")
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if self.df[col].isnull().sum() > 0:
                    mean_val = self.df[col].mean()
                    self.df[col].fillna(mean_val, inplace=True)
                    print(f"   Filled {col} with mean: {mean_val:.2f}")
        
        # 3. Normalize features using Min-Max scaling
        print("\n3. Normalizing features using Min-Max scaling...")
        scaler = MinMaxScaler()
        feature_cols = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
        self.df[feature_cols] = scaler.fit_transform(self.df[feature_cols])
        print("   Features normalized to range [0, 1]")
        
        # 4. Encode the class label
        print("\n4. Encoding class labels...")
        le = LabelEncoder()
        self.df['species_encoded'] = le.fit_transform(self.df['species_name'])
        print(f"   Classes encoded: {dict(zip(le.classes_, le.transform(le.classes_)))}")
        
        return self.df
    
    def explore_data(self) -> None:
        """Explore data with statistics and visualizations"""
        print("\n" + "="*60)
        print("STEP 3: DATA EXPLORATION")
        print("="*60)
        
        # 1. Summary statistics
        print("\n1. Summary Statistics:")
        print(self.df.describe().round(3))
        
        # 2. Class distribution
        print("\n2. Class Distribution:")
        print(self.df['species_name'].value_counts())
        
        # 3. Correlation matrix
        print("\n3. Correlation Matrix:")
        feature_cols = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
        corr_matrix = self.df[feature_cols].corr()
        print(corr_matrix.round(3))
        
        # Create visualizations
        self.create_visualizations()
    
    def create_visualizations(self) -> None:
        """Create and save visualization plots"""
        print("\n4. Creating Visualizations...")
        
        feature_cols = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
        
        # Set style
        sns.set_style("whitegrid")
        
        # 1. Pairplot
        fig = plt.figure(figsize=(12, 10))
        pairplot_data = self.df[feature_cols + ['species_name']].copy()
        g = sns.pairplot(pairplot_data, hue='species_name', palette='Set1', diag_kind='kde')
        g.fig.suptitle('Iris Dataset - Pairplot', y=1.02, fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig('iris_pairplot.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("   ✓ Pairplot saved as 'iris_pairplot.png'")
        
        # 2. Correlation Heatmap
        plt.figure(figsize=(8, 6))
        corr_matrix = self.df[feature_cols].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                   square=True, linewidths=1, cbar_kws={"shrink": 0.8})
        plt.title('Feature Correlation Heatmap', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig('iris_correlation.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("   ✓ Correlation heatmap saved as 'iris_correlation.png'")
        
        # 3. Boxplots for outlier detection
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        for idx, col in enumerate(feature_cols):
            ax = axes[idx // 2, idx % 2]
            self.df.boxplot(column=col, by='species_name', ax=ax)
            ax.set_title(f'{col.replace("_", " ").title()} by Species')
            ax.set_xlabel('Species')
            ax.set_ylabel(col.replace("_", " ").title())
            ax.get_figure().suptitle('')
        
        plt.suptitle('Boxplots for Outlier Detection', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig('iris_boxplots.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("   ✓ Boxplots saved as 'iris_boxplots.png'")
        
        # Identify outliers
        print("\n5. Outlier Detection (using IQR method):")
        for col in feature_cols:
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = self.df[(self.df[col] < Q1 - 1.5*IQR) | (self.df[col] > Q3 + 1.5*IQR)]
            if len(outliers) > 0:
                print(f"   {col}: {len(outliers)} outliers detected")
            else:
                print(f"   {col}: No outliers detected")
    
    def split_data(self, test_size=0.2, random_state=None) -> tuple:
        """Split data into train and test sets"""
        print("\n" + "="*60)
        print("STEP 4: TRAIN-TEST SPLIT")
        print("="*60)
        
        if random_state is None:
            random_state = self.seed
        
        feature_cols = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
        X = self.df[feature_cols]
        y = self.df['species']
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        print(f"Data split completed:")
        print(f"  Training set: {len(self.X_train)} samples")
        print(f"  Test set: {len(self.X_test)} samples")
        print(f"  Test ratio: {test_size:.1%}")
        
        # Verify stratification
        print(f"\nClass distribution in splits:")
        print(f"  Train: {pd.Series(self.y_train).value_counts().sort_index().to_dict()}")
        print(f"  Test:  {pd.Series(self.y_test).value_counts().sort_index().to_dict()}")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def save_preprocessed_data(self) -> None:
        """Save preprocessed data to CSV"""
        self.df.to_csv('preprocessed_iris.csv', index=False)
        print("\n✓ Preprocessed data saved to 'preprocessed_iris.csv'")
    
    def run_complete_preprocessing(self) -> tuple:
        """Execute complete preprocessing pipeline"""
        print("\n" + "🔬"*30)
        print("IRIS DATASET - PREPROCESSING PIPELINE")
        print("🔬"*30)
        
        # Load data
        self.load_data()
        
        # Preprocess
        self.preprocess_data()
        
        # Explore
        self.explore_data()
        
        # Split
        X_train, X_test, y_train, y_test = self.split_data()
        
        # Save
        self.save_preprocessed_data()
        
        print("\n" + "="*60)
        print("✅ PREPROCESSING PIPELINE COMPLETE!")
        print("="*60)
        print("\nGenerated files:")
        print("  - preprocessed_iris.csv")
        print("  - iris_pairplot.png")
        print("  - iris_correlation.png")
        print("  - iris_boxplots.png")
        
        return X_train, X_test, y_train, y_test

def main():
    """Main execution function"""
    # Initialize preprocessor
    preprocessor = IrisPreprocessor(use_synthetic=False, seed=42)
    
    # Run complete preprocessing pipeline
    X_train, X_test, y_train, y_test = preprocessor.run_complete_preprocessing()
    
    print("\n📊 Data ready for machine learning tasks!")

if __name__ == "__main__":
    main()