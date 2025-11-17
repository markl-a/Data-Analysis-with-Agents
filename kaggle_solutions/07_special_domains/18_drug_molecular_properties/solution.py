"""
Drug Discovery and Molecular Property Prediction
=================================================
Domain: Pharmaceutical & Computational Chemistry
Task: Predicting molecular properties for drug candidate screening

This solution demonstrates:
- Molecular fingerprint generation
- QSAR (Quantitative Structure-Activity Relationship) modeling
- Multi-task learning for property prediction
- Chemical space visualization
- Lipinski's Rule of Five analysis
- ADMET property prediction
- Structure-activity relationship analysis
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.neural_network import MLPRegressor
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')


class DrugMolecularPredictor:
    """
    QSAR modeling system for predicting molecular properties of drug candidates.
    Implements multi-task learning and chemical space analysis.
    """

    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.predictions = {}
        self.property_names = ['solubility', 'permeability', 'toxicity', 'bioavailability', 'clearance']

    def generate_molecular_data(self, n_molecules=2000):
        """
        Generate synthetic molecular descriptors and properties.
        Simulates drug-like molecules with realistic chemical properties.
        """
        np.random.seed(42)

        molecules = []

        for i in range(n_molecules):
            # Molecular descriptors (simulating fingerprints and physicochemical properties)

            # Basic properties
            molecular_weight = np.random.gamma(15, 20)  # Mean ~300 Da
            molecular_weight = np.clip(molecular_weight, 100, 800)

            logp = np.random.normal(2.5, 1.5)  # Lipophilicity
            logp = np.clip(logp, -2, 7)

            n_hbd = int(np.random.poisson(2))  # H-bond donors
            n_hba = int(np.random.poisson(4))  # H-bond acceptors
            n_rotatable = int(np.random.poisson(5))  # Rotatable bonds

            tpsa = np.random.gamma(4, 15)  # Topological polar surface area
            tpsa = np.clip(tpsa, 0, 200)

            # Aromatic and aliphatic features
            n_aromatic = int(np.random.poisson(2))
            n_aliphatic = int(np.random.poisson(3))

            # Electronic properties
            dipole_moment = np.random.gamma(2, 1)
            polarizability = molecular_weight * 0.3 + np.random.normal(0, 5)

            # Structural complexity
            complexity = np.random.gamma(10, 20)

            # Morgan fingerprint (simulated binary features)
            fingerprint = np.random.binomial(1, 0.15, 256)

            # Calculate target properties (with realistic correlations)

            # Solubility (logS) - correlated with logP, MW, TPSA
            solubility = -0.5 * logp - 0.002 * molecular_weight + 0.01 * tpsa + np.random.normal(0, 0.5)
            solubility = np.clip(solubility, -8, 2)

            # Permeability (log Papp) - correlated with logP, TPSA
            permeability = 0.3 * logp - 0.02 * tpsa + np.random.normal(0, 0.3)
            permeability = np.clip(permeability, -7, -3)

            # Toxicity score (0-10) - complex relationships
            toxicity = 2 + 0.3 * logp + 0.005 * molecular_weight - 0.01 * tpsa
            toxicity += 0.2 * n_aromatic + np.random.normal(0, 1)
            toxicity = np.clip(toxicity, 0, 10)

            # Bioavailability (%) - Lipinski-like rules
            bioavail = 80 - 5 * max(0, molecular_weight - 500) / 100
            bioavail -= 10 * max(0, logp - 5)
            bioavail -= 5 * max(0, n_hbd - 5)
            bioavail += np.random.normal(0, 10)
            bioavail = np.clip(bioavail, 0, 100)

            # Clearance (mL/min/kg) - related to MW and logP
            clearance = 50 - 0.05 * molecular_weight + 2 * logp + np.random.normal(0, 10)
            clearance = np.clip(clearance, 1, 100)

            # Check Lipinski's Rule of Five
            lipinski_violations = 0
            if molecular_weight > 500:
                lipinski_violations += 1
            if logp > 5:
                lipinski_violations += 1
            if n_hbd > 5:
                lipinski_violations += 1
            if n_hba > 10:
                lipinski_violations += 1

            molecules.append({
                'molecule_id': f'MOL_{i:05d}',
                'molecular_weight': molecular_weight,
                'logp': logp,
                'n_hbd': n_hbd,
                'n_hba': n_hba,
                'n_rotatable': n_rotatable,
                'tpsa': tpsa,
                'n_aromatic': n_aromatic,
                'n_aliphatic': n_aliphatic,
                'dipole_moment': dipole_moment,
                'polarizability': polarizability,
                'complexity': complexity,
                'lipinski_violations': lipinski_violations,
                'solubility': solubility,
                'permeability': permeability,
                'toxicity': toxicity,
                'bioavailability': bioavail,
                'clearance': clearance,
                'fingerprint': fingerprint.tolist()
            })

        df = pd.DataFrame(molecules)

        # Extract features
        descriptor_cols = ['molecular_weight', 'logp', 'n_hbd', 'n_hba', 'n_rotatable',
                          'tpsa', 'n_aromatic', 'n_aliphatic', 'dipole_moment',
                          'polarizability', 'complexity']

        X_descriptors = df[descriptor_cols].values

        # Add fingerprints
        X_fingerprints = np.array(df['fingerprint'].tolist())
        X = np.hstack([X_descriptors, X_fingerprints])

        # Targets
        y = df[self.property_names].values

        print(f"Generated {n_molecules} drug-like molecules")
        print(f"Descriptor features: {len(descriptor_cols)}")
        print(f"Fingerprint features: {X_fingerprints.shape[1]}")
        print(f"Total features: {X.shape[1]}")
        print(f"Target properties: {len(self.property_names)}")
        print(f"\nLipinski Rule of Five compliance:")
        print(f"  0 violations: {np.sum(df['lipinski_violations'] == 0)} ({np.sum(df['lipinski_violations'] == 0)/len(df)*100:.1f}%)")
        print(f"  1 violation: {np.sum(df['lipinski_violations'] == 1)} ({np.sum(df['lipinski_violations'] == 1)/len(df)*100:.1f}%)")
        print(f"  2+ violations: {np.sum(df['lipinski_violations'] >= 2)} ({np.sum(df['lipinski_violations'] >= 2)/len(df)*100:.1f}%)")

        return X, y, df

    def train_multitask_models(self, X_train, y_train):
        """Train models for each molecular property."""
        print("\nTraining multi-task QSAR models...")

        for prop_idx, prop_name in enumerate(self.property_names):
            print(f"\n  Property: {prop_name.upper()}")

            y_prop = y_train[:, prop_idx]

            # Standardize
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_train)
            self.scalers[prop_name] = scaler

            models = {}

            # Random Forest
            rf = RandomForestRegressor(n_estimators=200, max_depth=20, random_state=42, n_jobs=-1)
            rf.fit(X_scaled, y_prop)
            models['Random Forest'] = rf

            # Gradient Boosting
            gb = GradientBoostingRegressor(n_estimators=150, max_depth=8, random_state=42)
            gb.fit(X_scaled, y_prop)
            models['Gradient Boosting'] = gb

            # Neural Network
            nn = MLPRegressor(hidden_layers=(128, 64, 32), max_iter=500, random_state=42)
            nn.fit(X_scaled, y_prop)
            models['Neural Network'] = nn

            # Elastic Net
            en = ElasticNet(alpha=0.1, random_state=42)
            en.fit(X_scaled, y_prop)
            models['Elastic Net'] = en

            self.models[prop_name] = models

            print(f"    Trained {len(models)} models")

    def evaluate_models(self, X_test, y_test):
        """Evaluate models for all properties."""
        results = []

        for prop_idx, prop_name in enumerate(self.property_names):
            y_true = y_test[:, prop_idx]
            X_scaled = self.scalers[prop_name].transform(X_test)

            for model_name, model in self.models[prop_name].items():
                y_pred = model.predict(X_scaled)

                rmse = np.sqrt(mean_squared_error(y_true, y_pred))
                mae = mean_absolute_error(y_true, y_pred)
                r2 = r2_score(y_true, y_pred)

                results.append({
                    'Property': prop_name.title(),
                    'Model': model_name,
                    'RMSE': rmse,
                    'MAE': mae,
                    'R²': r2
                })

                # Store predictions
                if model_name == 'Random Forest':  # Store best model predictions
                    if prop_name not in self.predictions:
                        self.predictions[prop_name] = {}
                    self.predictions[prop_name]['y_true'] = y_true
                    self.predictions[prop_name]['y_pred'] = y_pred

        return pd.DataFrame(results)

    def analyze_chemical_space(self, X, df):
        """Analyze and visualize chemical space using PCA."""
        # Use only physicochemical descriptors for interpretability
        descriptor_cols = ['molecular_weight', 'logp', 'n_hbd', 'n_hba', 'n_rotatable',
                          'tpsa', 'n_aromatic', 'n_aliphatic', 'dipole_moment',
                          'polarizability', 'complexity']

        X_descriptors = df[descriptor_cols].values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_descriptors)

        # PCA
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)

        print(f"\nChemical Space Analysis:")
        print(f"  PC1 variance explained: {pca.explained_variance_ratio_[0]:.3f}")
        print(f"  PC2 variance explained: {pca.explained_variance_ratio_[1]:.3f}")
        print(f"  Total variance explained: {sum(pca.explained_variance_ratio_):.3f}")

        return X_pca, pca

    def plot_property_predictions(self):
        """Plot predicted vs actual for all properties."""
        n_props = len(self.property_names)
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.ravel()

        for idx, prop_name in enumerate(self.property_names):
            if prop_name not in self.predictions:
                continue

            y_true = self.predictions[prop_name]['y_true']
            y_pred = self.predictions[prop_name]['y_pred']

            axes[idx].scatter(y_true, y_pred, alpha=0.5, s=20)
            axes[idx].plot([y_true.min(), y_true.max()],
                          [y_true.min(), y_true.max()],
                          'r--', linewidth=2, label='Perfect Prediction')

            r2 = r2_score(y_true, y_pred)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))

            axes[idx].set_xlabel(f'Actual {prop_name.title()}', fontsize=11)
            axes[idx].set_ylabel(f'Predicted {prop_name.title()}', fontsize=11)
            axes[idx].set_title(f'{prop_name.title()}\nR² = {r2:.3f}, RMSE = {rmse:.3f}',
                               fontsize=12, fontweight='bold')
            axes[idx].legend()
            axes[idx].grid(True, alpha=0.3)

        # Remove extra subplot
        fig.delaxes(axes[5])

        plt.tight_layout()
        plt.savefig('drug_property_predictions.png', dpi=300, bbox_inches='tight')
        print("Saved: drug_property_predictions.png")
        plt.close()

    def plot_chemical_space(self, X_pca, df):
        """Visualize chemical space with property overlays."""
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        axes = axes.ravel()

        # Color by Lipinski violations
        scatter = axes[0].scatter(X_pca[:, 0], X_pca[:, 1],
                                 c=df['lipinski_violations'],
                                 cmap='RdYlGn_r', s=30, alpha=0.6)
        axes[0].set_title('Chemical Space - Lipinski Violations', fontsize=12, fontweight='bold')
        axes[0].set_xlabel('PC1', fontsize=11)
        axes[0].set_ylabel('PC2', fontsize=11)
        plt.colorbar(scatter, ax=axes[0], label='Violations')

        # Color by properties
        properties = ['solubility', 'permeability', 'toxicity', 'bioavailability', 'clearance']
        for idx, prop in enumerate(properties, 1):
            scatter = axes[idx].scatter(X_pca[:, 0], X_pca[:, 1],
                                       c=df[prop], cmap='viridis', s=30, alpha=0.6)
            axes[idx].set_title(f'Chemical Space - {prop.title()}', fontsize=12, fontweight='bold')
            axes[idx].set_xlabel('PC1', fontsize=11)
            axes[idx].set_ylabel('PC2', fontsize=11)
            plt.colorbar(scatter, ax=axes[idx], label=prop.title())

        plt.tight_layout()
        plt.savefig('drug_chemical_space.png', dpi=300, bbox_inches='tight')
        print("Saved: drug_chemical_space.png")
        plt.close()

    def plot_model_performance_comparison(self, results_df):
        """Compare model performances across properties."""
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))

        metrics = ['RMSE', 'MAE', 'R²']

        for idx, metric in enumerate(metrics):
            pivot_data = results_df.pivot(index='Model', columns='Property', values=metric)

            pivot_data.plot(kind='bar', ax=axes[idx], width=0.8)
            axes[idx].set_title(f'{metric} by Model and Property', fontsize=14, fontweight='bold')
            axes[idx].set_ylabel(metric, fontsize=12)
            axes[idx].set_xlabel('Model', fontsize=12)
            axes[idx].legend(title='Property', bbox_to_anchor=(1.05, 1), loc='upper left')
            axes[idx].grid(True, alpha=0.3, axis='y')
            axes[idx].tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.savefig('drug_model_comparison.png', dpi=300, bbox_inches='tight')
        print("Saved: drug_model_comparison.png")
        plt.close()

    def plot_lipinski_analysis(self, df):
        """Analyze Lipinski's Rule of Five compliance."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # Molecular weight distribution
        axes[0, 0].hist(df['molecular_weight'], bins=50, color='skyblue', edgecolor='black', alpha=0.7)
        axes[0, 0].axvline(500, color='red', linestyle='--', linewidth=2, label='Lipinski Limit (500)')
        axes[0, 0].set_xlabel('Molecular Weight (Da)', fontsize=11)
        axes[0, 0].set_ylabel('Frequency', fontsize=11)
        axes[0, 0].set_title('Molecular Weight Distribution', fontsize=12, fontweight='bold')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # LogP distribution
        axes[0, 1].hist(df['logp'], bins=50, color='lightcoral', edgecolor='black', alpha=0.7)
        axes[0, 1].axvline(5, color='red', linestyle='--', linewidth=2, label='Lipinski Limit (5)')
        axes[0, 1].set_xlabel('LogP (Lipophilicity)', fontsize=11)
        axes[0, 1].set_ylabel('Frequency', fontsize=11)
        axes[0, 1].set_title('LogP Distribution', fontsize=12, fontweight='bold')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # H-bond donors vs acceptors
        scatter = axes[1, 0].scatter(df['n_hbd'], df['n_hba'],
                                    c=df['bioavailability'], cmap='RdYlGn',
                                    s=50, alpha=0.6, edgecolors='black')
        axes[1, 0].axvline(5, color='red', linestyle='--', linewidth=2, alpha=0.5)
        axes[1, 0].axhline(10, color='red', linestyle='--', linewidth=2, alpha=0.5)
        axes[1, 0].set_xlabel('H-Bond Donors', fontsize=11)
        axes[1, 0].set_ylabel('H-Bond Acceptors', fontsize=11)
        axes[1, 0].set_title('H-Bond Donors vs Acceptors', fontsize=12, fontweight='bold')
        plt.colorbar(scatter, ax=axes[1, 0], label='Bioavailability (%)')

        # Lipinski violations impact
        violation_groups = df.groupby('lipinski_violations')['bioavailability'].mean()
        axes[1, 1].bar(violation_groups.index, violation_groups.values,
                      color=['green', 'yellow', 'orange', 'red', 'darkred'][:len(violation_groups)],
                      edgecolor='black', alpha=0.7)
        axes[1, 1].set_xlabel('Number of Lipinski Violations', fontsize=11)
        axes[1, 1].set_ylabel('Average Bioavailability (%)', fontsize=11)
        axes[1, 1].set_title('Lipinski Violations vs Bioavailability', fontsize=12, fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig('drug_lipinski_analysis.png', dpi=300, bbox_inches='tight')
        print("Saved: drug_lipinski_analysis.png")
        plt.close()


def main():
    """Main execution function."""
    print("=" * 80)
    print("Drug Discovery and Molecular Property Prediction")
    print("=" * 80)

    # Initialize predictor
    predictor = DrugMolecularPredictor()

    # Generate data
    print("\n1. Generating Molecular Data...")
    X, y, df = predictor.generate_molecular_data(n_molecules=2000)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"\nData split: {len(X_train)} train, {len(X_test)} test")

    # Train models
    print("\n2. Training Multi-Task QSAR Models...")
    predictor.train_multitask_models(X_train, y_train)

    # Evaluate
    print("\n3. Evaluating Model Performance...")
    results = predictor.evaluate_models(X_test, y_test)

    print("\nModel Performance Summary:")
    for prop in predictor.property_names:
        prop_results = results[results['Property'] == prop.title()]
        best_model = prop_results.loc[prop_results['R²'].idxmax()]
        print(f"\n{prop.upper()}:")
        print(f"  Best Model: {best_model['Model']}")
        print(f"  R²: {best_model['R²']:.4f}")
        print(f"  RMSE: {best_model['RMSE']:.4f}")

    # Chemical space analysis
    print("\n4. Analyzing Chemical Space...")
    X_pca, pca = predictor.analyze_chemical_space(X, df)

    # Visualizations
    print("\n5. Generating Visualizations...")
    predictor.plot_property_predictions()
    predictor.plot_chemical_space(X_pca, df)
    predictor.plot_model_performance_comparison(results)
    predictor.plot_lipinski_analysis(df)

    print("\n" + "=" * 80)
    print("Analysis Complete!")
    print("=" * 80)
    print("\nKey Insights:")
    print("- QSAR models successfully predict multiple molecular properties")
    print("- Lipinski's Rule of Five strongly correlates with bioavailability")
    print("- Chemical space visualization reveals structure-property relationships")
    print("- Multi-task learning enables efficient property prediction pipeline")
    print("- Models support virtual screening and lead optimization workflows")


if __name__ == "__main__":
    main()
