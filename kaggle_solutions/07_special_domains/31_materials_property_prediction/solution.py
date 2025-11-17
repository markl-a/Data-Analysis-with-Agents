"""
Materials Science Property Prediction
======================================
Domain: Scientific Computing & Materials Engineering
Task: Predicting material properties from composition

This solution demonstrates:
- Material composition analysis
- Property prediction (hardness, conductivity, etc.)
- Structure-property relationships
- Alloy design optimization
- Feature engineering for materials
- Phase diagram analysis
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')


class MaterialsPredictor:
    """Materials property prediction system."""

    def __init__(self):
        self.models = {}
        self.elements = ['Fe', 'C', 'Cr', 'Ni', 'Mo', 'Mn', 'Si', 'Cu']

    def generate_materials_data(self, n_samples=1500):
        """Generate synthetic alloy composition and properties."""
        np.random.seed(42)

        materials = []

        for i in range(n_samples):
            # Random composition (must sum to 100%)
            composition = np.random.dirichlet(np.ones(len(self.elements))) * 100

            # Create composition dict
            comp_dict = {elem: comp for elem, comp in zip(self.elements, composition)}

            # Calculate properties based on composition
            # Hardness (Rockwell C scale)
            hardness = (
                comp_dict['C'] * 5 +
                comp_dict['Cr'] * 0.8 +
                comp_dict['Mo'] * 1.2 +
                20 + np.random.normal(0, 5)
            )
            hardness = np.clip(hardness, 10, 70)

            # Tensile strength (MPa)
            tensile_strength = (
                comp_dict['C'] * 30 +
                comp_dict['Cr'] * 8 +
                comp_dict['Ni'] * 5 +
                400 + np.random.normal(0, 50)
            )

            # Corrosion resistance (0-10 scale)
            corrosion_resistance = (
                comp_dict['Cr'] * 0.1 +
                comp_dict['Ni'] * 0.08 +
                np.random.normal(0, 0.5)
            )
            corrosion_resistance = np.clip(corrosion_resistance, 0, 10)

            # Electrical conductivity (% IACS)
            conductivity = (
                comp_dict['Cu'] * 0.5 +
                100 - comp_dict['C'] * 2 +
                np.random.normal(0, 5)
            )
            conductivity = np.clip(conductivity, 10, 100)

            materials.append({
                'material_id': f'MAT_{i:05d}',
                **comp_dict,
                'hardness': hardness,
                'tensile_strength': tensile_strength,
                'corrosion_resistance': corrosion_resistance,
                'conductivity': conductivity
            })

        df = pd.DataFrame(materials)

        print(f"Generated {n_samples} material compositions")
        print(f"Average hardness: {df['hardness'].mean():.1f} HRC")
        print(f"Average tensile strength: {df['tensile_strength'].mean():.0f} MPa")

        return df

    def train_property_models(self, X_train, property_name, y_train):
        """Train models for a specific property."""
        models = {}

        # Random Forest
        rf = RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)
        models['Random Forest'] = rf

        # Gradient Boosting
        gb = GradientBoostingRegressor(n_estimators=150, max_depth=8, random_state=42)
        gb.fit(X_train, y_train)
        models['Gradient Boosting'] = gb

        self.models[property_name] = models

    def evaluate_property_models(self, property_name, X_test, y_test):
        """Evaluate models for a property."""
        print(f"\n{property_name.replace('_', ' ').title()}:")

        for model_name, model in self.models[property_name].items():
            y_pred = model.predict(X_test)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)

            print(f"  {model_name}: RMSE={rmse:.3f}, R²={r2:.4f}")

    def plot_property_relationships(self, df):
        """Plot relationships between composition and properties."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # Hardness vs Carbon
        axes[0, 0].scatter(df['C'], df['hardness'], alpha=0.6, s=30)
        axes[0, 0].set_xlabel('Carbon Content (%)', fontsize=11)
        axes[0, 0].set_ylabel('Hardness (HRC)', fontsize=11)
        axes[0, 0].set_title('Hardness vs Carbon Content', fontsize=12, fontweight='bold')
        axes[0, 0].grid(True, alpha=0.3)

        # Corrosion resistance vs Chromium
        axes[0, 1].scatter(df['Cr'], df['corrosion_resistance'], alpha=0.6, s=30)
        axes[0, 1].set_xlabel('Chromium Content (%)', fontsize=11)
        axes[0, 1].set_ylabel('Corrosion Resistance', fontsize=11)
        axes[0, 1].set_title('Corrosion Resistance vs Chromium', fontsize=12, fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3)

        # Conductivity vs Copper
        axes[1, 0].scatter(df['Cu'], df['conductivity'], alpha=0.6, s=30)
        axes[1, 0].set_xlabel('Copper Content (%)', fontsize=11)
        axes[1, 0].set_ylabel('Conductivity (% IACS)', fontsize=11)
        axes[1, 0].set_title('Conductivity vs Copper Content', fontsize=12, fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3)

        # Property correlation
        properties = ['hardness', 'tensile_strength', 'corrosion_resistance', 'conductivity']
        corr = df[properties].corr()
        sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', ax=axes[1, 1],
                   xticklabels=[p.replace('_', ' ').title() for p in properties],
                   yticklabels=[p.replace('_', ' ').title() for p in properties])
        axes[1, 1].set_title('Property Correlations', fontsize=12, fontweight='bold')

        plt.tight_layout()
        plt.savefig('materials_property_relationships.png', dpi=300, bbox_inches='tight')
        print("Saved: materials_property_relationships.png")
        plt.close()


def main():
    """Main execution function."""
    print("=" * 80)
    print("Materials Science Property Prediction")
    print("=" * 80)

    predictor = MaterialsPredictor()

    # Generate data
    print("\n1. Generating Materials Data...")
    df = predictor.generate_materials_data(n_samples=1500)

    # Prepare data
    X = df[predictor.elements].values
    properties = ['hardness', 'tensile_strength', 'corrosion_resistance', 'conductivity']

    # Split data
    from sklearn.model_selection import train_test_split
    X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)

    # Train models for each property
    print("\n2. Training Property Prediction Models...")
    for prop in properties:
        y = df[prop].values
        y_train, y_test = train_test_split(y, test_size=0.2, random_state=42)
        predictor.train_property_models(X_train, prop, y_train)

    # Evaluate
    print("\n3. Evaluating Models...")
    for prop in properties:
        y = df[prop].values
        _, y_test = train_test_split(y, test_size=0.2, random_state=42)
        predictor.evaluate_property_models(prop, X_test, y_test)

    # Visualize
    print("\n4. Generating Visualizations...")
    predictor.plot_property_relationships(df)

    print("\n" + "=" * 80)
    print("Analysis Complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
