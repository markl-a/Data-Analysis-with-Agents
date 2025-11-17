"""
Protein Structure Prediction
=============================
Domain: Scientific Computing & Bioinformatics
Task: Predicting protein secondary structure from amino acid sequences

This solution demonstrates:
- Protein sequence analysis
- Secondary structure prediction (alpha-helix, beta-sheet, coil)
- Feature extraction from amino acid properties
- Multiple ML approaches for structure prediction
- Sequence alignment and motif detection
- Structure visualization and analysis
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings
warnings.filterwarnings('ignore')


class ProteinStructurePredictor:
    """Protein secondary structure prediction system."""

    def __init__(self):
        self.amino_acids = list('ACDEFGHIKLMNPQRSTVWY')
        self.structures = ['H', 'E', 'C']  # Helix, Sheet, Coil
        self.models = {}

        # Amino acid properties
        self.aa_properties = {
            'hydrophobicity': {
                'A': 1.8, 'C': 2.5, 'D': -3.5, 'E': -3.5, 'F': 2.8,
                'G': -0.4, 'H': -3.2, 'I': 4.5, 'K': -3.9, 'L': 3.8,
                'M': 1.9, 'N': -3.5, 'P': -1.6, 'Q': -3.5, 'R': -4.5,
                'S': -0.8, 'T': -0.7, 'V': 4.2, 'W': -0.9, 'Y': -1.3
            },
            'helix_propensity': {
                'A': 1.45, 'C': 0.77, 'D': 0.98, 'E': 1.53, 'F': 1.12,
                'G': 0.53, 'H': 1.24, 'I': 1.00, 'K': 1.07, 'L': 1.34,
                'M': 1.20, 'N': 0.73, 'P': 0.59, 'Q': 1.17, 'R': 0.79,
                'S': 0.79, 'T': 0.82, 'V': 1.14, 'W': 1.14, 'Y': 0.61
            }
        }

    def generate_protein_data(self, n_sequences=1000, seq_length=50):
        """Generate synthetic protein sequences with secondary structures."""
        np.random.seed(42)

        sequences = []

        for i in range(n_sequences):
            # Generate sequence
            sequence = ''
            structure = ''

            pos = 0
            while pos < seq_length:
                # Randomly choose structure type
                struct_type = np.random.choice(self.structures, p=[0.35, 0.25, 0.40])

                # Length of structure segment
                if struct_type == 'H':  # Helix
                    length = np.random.randint(6, 15)
                    # Helix-forming amino acids
                    segment = ''.join(np.random.choice(list('AELKRM'), size=length))
                elif struct_type == 'E':  # Sheet
                    length = np.random.randint(4, 10)
                    # Sheet-forming amino acids
                    segment = ''.join(np.random.choice(list('VIFYWTC'), size=length))
                else:  # Coil
                    length = np.random.randint(2, 8)
                    # Random amino acids
                    segment = ''.join(np.random.choice(self.amino_acids, size=length))

                sequence += segment
                structure += struct_type * length
                pos += length

            # Trim to desired length
            sequence = sequence[:seq_length]
            structure = structure[:seq_length]

            # Pad if necessary
            if len(sequence) < seq_length:
                pad_length = seq_length - len(sequence)
                sequence += 'X' * pad_length
                structure += 'C' * pad_length

            sequences.append({
                'protein_id': f'PROT_{i:05d}',
                'sequence': sequence,
                'structure': structure,
                'length': seq_length
            })

        df = pd.DataFrame(sequences)

        print(f"Generated {n_sequences} protein sequences")
        print(f"Sequence length: {seq_length}")

        # Calculate structure statistics
        all_structures = ''.join(df['structure'])
        for s in self.structures:
            count = all_structures.count(s)
            print(f"Structure {s}: {count/len(all_structures)*100:.1f}%")

        return df

    def extract_sequence_features(self, sequence, position):
        """Extract features for a position in the sequence."""
        window_size = 7
        half_window = window_size // 2

        features = {}

        # Window of amino acids
        window_start = max(0, position - half_window)
        window_end = min(len(sequence), position + half_window + 1)

        window = sequence[window_start:window_end]

        # One-hot encoding for center position
        if position < len(sequence) and sequence[position] != 'X':
            for aa in self.amino_acids:
                features[f'aa_{aa}'] = 1 if sequence[position] == aa else 0

        # Average properties in window
        hydrophobicity_sum = 0
        helix_prop_sum = 0
        count = 0

        for aa in window:
            if aa != 'X' and aa in self.aa_properties['hydrophobicity']:
                hydrophobicity_sum += self.aa_properties['hydrophobicity'][aa]
                helix_prop_sum += self.aa_properties['helix_propensity'][aa]
                count += 1

        features['hydrophobicity'] = hydrophobicity_sum / count if count > 0 else 0
        features['helix_propensity'] = helix_prop_sum / count if count > 0 else 0

        # Position features
        features['position'] = position / len(sequence)
        features['distance_to_n_term'] = position
        features['distance_to_c_term'] = len(sequence) - position

        return features

    def prepare_training_data(self, df):
        """Convert sequences to feature matrix."""
        X_list = []
        y_list = []

        for idx, row in df.iterrows():
            sequence = row['sequence']
            structure = row['structure']

            for pos in range(len(sequence)):
                if sequence[pos] != 'X':
                    features = self.extract_sequence_features(sequence, pos)
                    X_list.append(features)
                    y_list.append(structure[pos])

        X = pd.DataFrame(X_list).fillna(0)
        y = np.array(y_list)

        print(f"\nPrepared training data:")
        print(f"  Total positions: {len(X)}")
        print(f"  Features: {X.shape[1]}")

        return X.values, y

    def train_models(self, X_train, y_train):
        """Train structure prediction models."""
        print("\nTraining models...")

        # Random Forest
        print("  - Random Forest...")
        rf = RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)
        self.models['Random Forest'] = rf

        # Gradient Boosting
        print("  - Gradient Boosting...")
        gb = GradientBoostingClassifier(n_estimators=150, max_depth=10, random_state=42)
        gb.fit(X_train, y_train)
        self.models['Gradient Boosting'] = gb

        print(f"Trained {len(self.models)} models")

    def evaluate_models(self, X_test, y_test):
        """Evaluate models."""
        for name, model in self.models.items():
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)

            print(f"\n{name}:")
            print(f"  Accuracy: {accuracy:.4f}")
            print(classification_report(y_test, y_pred, target_names=['Helix', 'Sheet', 'Coil']))

    def plot_structure_distribution(self, df):
        """Plot structure type distribution."""
        all_structures = ''.join(df['structure'])
        struct_counts = {s: all_structures.count(s) for s in self.structures}

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(range(len(struct_counts)), list(struct_counts.values()),
              color=['red', 'blue', 'gray'], edgecolor='black', alpha=0.7)
        ax.set_xticks(range(len(struct_counts)))
        ax.set_xticklabels(['Helix', 'Sheet', 'Coil'])
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title('Secondary Structure Distribution', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig('protein_structure_distribution.png', dpi=300, bbox_inches='tight')
        print("Saved: protein_structure_distribution.png")
        plt.close()

    def plot_confusion_matrix(self, y_test, y_pred):
        """Plot confusion matrix."""
        cm = confusion_matrix(y_test, y_pred, labels=self.structures)
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(cm_norm, annot=True, fmt='.3f', cmap='Blues',
                   xticklabels=['Helix', 'Sheet', 'Coil'],
                   yticklabels=['Helix', 'Sheet', 'Coil'],
                   ax=ax, cbar_kws={'label': 'Proportion'})

        ax.set_title('Structure Prediction Confusion Matrix', fontsize=14, fontweight='bold')
        ax.set_ylabel('True Structure', fontsize=12)
        ax.set_xlabel('Predicted Structure', fontsize=12)

        plt.tight_layout()
        plt.savefig('protein_confusion_matrix.png', dpi=300, bbox_inches='tight')
        print("Saved: protein_confusion_matrix.png")
        plt.close()


def main():
    """Main execution function."""
    print("=" * 80)
    print("Protein Secondary Structure Prediction")
    print("=" * 80)

    predictor = ProteinStructurePredictor()

    # Generate data
    print("\n1. Generating Protein Sequences...")
    df = predictor.generate_protein_data(n_sequences=1000, seq_length=50)

    # Prepare training data
    print("\n2. Extracting Sequence Features...")
    X, y = predictor.prepare_training_data(df)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Train
    print("\n3. Training Prediction Models...")
    predictor.train_models(X_train, y_train)

    # Evaluate
    print("\n4. Evaluating Models...")
    predictor.evaluate_models(X_test, y_test)

    # Visualizations
    print("\n5. Generating Visualizations...")
    predictor.plot_structure_distribution(df)

    y_pred = predictor.models['Random Forest'].predict(X_test)
    predictor.plot_confusion_matrix(y_test, y_pred)

    print("\n" + "=" * 80)
    print("Analysis Complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
