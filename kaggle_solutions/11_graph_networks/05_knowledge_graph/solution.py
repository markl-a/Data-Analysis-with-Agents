"""
Knowledge Graph Construction - Kaggle Solution
==============================================
Constructs and analyzes knowledge graphs with entities and relationships.
"""

import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

class KnowledgeGraphBuilder:
    """Knowledge graph construction and analysis"""

    def __init__(self, seed=42):
        """Initialize knowledge graph builder"""
        self.seed = seed
        np.random.seed(seed)
        self.G = None
        self.entities = None
        self.relations = None

    def generate_knowledge_graph(self, n_entities=80):
        """
        Generate knowledge graph with entities and relations

        Args:
            n_entities: Number of entities
        """
        print("Generating knowledge graph...")

        self.G = nx.MultiDiGraph()  # Multi-directed graph for multiple edge types

        # Define entity types
        entity_types = {
            'Person': ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve', 'Frank', 'Grace', 'Henry'],
            'Organization': ['TechCorp', 'DataInc', 'AILabs', 'WebSystems', 'CloudNet'],
            'Location': ['New York', 'San Francisco', 'London', 'Tokyo', 'Berlin'],
            'Technology': ['Python', 'Machine Learning', 'Cloud Computing', 'Blockchain',
                          'Databases', 'AI', 'Networks', 'Security'],
            'Product': ['Product_A', 'Product_B', 'Product_C', 'Product_D'],
            'Concept': ['Innovation', 'Research', 'Development', 'Quality', 'Efficiency']
        }

        # Create entities
        entities = []
        entity_id = 0

        for etype, names in entity_types.items():
            for name in names:
                entities.append({
                    'id': entity_id,
                    'name': name,
                    'type': etype
                })
                self.G.add_node(entity_id, name=name, type=etype)
                entity_id += 1

        self.entities = pd.DataFrame(entities)

        # Define relation types and valid entity type pairs
        relation_rules = {
            'works_for': [('Person', 'Organization')],
            'located_in': [('Organization', 'Location'), ('Person', 'Location')],
            'develops': [('Organization', 'Product'), ('Person', 'Product')],
            'uses': [('Organization', 'Technology'), ('Person', 'Technology')],
            'collaborates_with': [('Person', 'Person'), ('Organization', 'Organization')],
            'specializes_in': [('Person', 'Technology'), ('Person', 'Concept')],
            'produces': [('Organization', 'Product')],
            'related_to': [('Product', 'Technology'), ('Technology', 'Concept')]
        }

        # Create relations
        relations = []
        relation_id = 0

        for relation_type, rules in relation_rules.items():
            for source_type, target_type in rules:
                # Get entities of each type
                sources = self.entities[self.entities['type'] == source_type]['id'].values
                targets = self.entities[self.entities['type'] == target_type]['id'].values

                if len(sources) > 0 and len(targets) > 0:
                    # Create some relations
                    n_relations = min(len(sources) * 2, 10)

                    for _ in range(n_relations):
                        source = np.random.choice(sources)
                        target = np.random.choice(targets)

                        # Avoid self-loops for collaborates_with
                        if source != target or relation_type != 'collaborates_with':
                            self.G.add_edge(source, target,
                                          relation=relation_type,
                                          id=relation_id)

                            relations.append({
                                'id': relation_id,
                                'source': source,
                                'target': target,
                                'relation': relation_type,
                                'source_type': source_type,
                                'target_type': target_type
                            })
                            relation_id += 1

        self.relations = pd.DataFrame(relations)

        print(f"Created knowledge graph with {self.G.number_of_nodes()} entities")
        print(f"Created {len(self.relations)} relations")
        print(f"\nEntity types: {dict(self.entities['type'].value_counts())}")
        print(f"\nRelation types: {dict(self.relations['relation'].value_counts())}")

        return self.G, self.entities, self.relations

    def analyze_graph_structure(self):
        """Analyze knowledge graph structure"""
        print("\n" + "="*60)
        print("KNOWLEDGE GRAPH STRUCTURE ANALYSIS")
        print("="*60)

        # Basic statistics
        print(f"\nBasic Statistics:")
        print(f"  Nodes: {self.G.number_of_nodes()}")
        print(f"  Edges: {self.G.number_of_edges()}")
        print(f"  Density: {nx.density(self.G):.4f}")

        # Connectivity
        # Convert to undirected for connectivity analysis
        G_undirected = self.G.to_undirected()
        is_connected = nx.is_connected(G_undirected)
        print(f"  Connected: {is_connected}")

        if not is_connected:
            n_components = nx.number_connected_components(G_undirected)
            print(f"  Number of components: {n_components}")

        # Degree statistics
        in_degrees = dict(self.G.in_degree())
        out_degrees = dict(self.G.out_degree())

        print(f"\nDegree Statistics:")
        print(f"  Avg in-degree: {np.mean(list(in_degrees.values())):.2f}")
        print(f"  Avg out-degree: {np.mean(list(out_degrees.values())):.2f}")

        return {
            'n_nodes': self.G.number_of_nodes(),
            'n_edges': self.G.number_of_edges(),
            'density': nx.density(self.G)
        }

    def query_knowledge_graph(self):
        """Perform queries on knowledge graph"""
        print("\n" + "="*60)
        print("KNOWLEDGE GRAPH QUERIES")
        print("="*60)

        # Query 1: Find all people working for organizations
        print("\nQuery 1: People and their Organizations")
        works_for = self.relations[self.relations['relation'] == 'works_for']
        for _, row in works_for.head(5).iterrows():
            person = self.entities[self.entities['id'] == row['source']]['name'].values[0]
            org = self.entities[self.entities['id'] == row['target']]['name'].values[0]
            print(f"  {person} works for {org}")

        # Query 2: Find technologies used by organizations
        print("\nQuery 2: Organizations and Technologies")
        uses = self.relations[self.relations['relation'] == 'uses']
        uses_org = uses[uses['source_type'] == 'Organization']
        for _, row in uses_org.head(5).iterrows():
            org = self.entities[self.entities['id'] == row['source']]['name'].values[0]
            tech = self.entities[self.entities['id'] == row['target']]['name'].values[0]
            print(f"  {org} uses {tech}")

        # Query 3: Find products and their related technologies
        print("\nQuery 3: Products and Related Technologies")
        related = self.relations[
            (self.relations['relation'] == 'related_to') &
            (self.relations['source_type'] == 'Product')
        ]
        for _, row in related.head(5).iterrows():
            product = self.entities[self.entities['id'] == row['source']]['name'].values[0]
            tech = self.entities[self.entities['id'] == row['target']]['name'].values[0]
            print(f"  {product} is related to {tech}")

    def find_entity_connections(self, entity_name):
        """Find all connections for an entity"""
        print(f"\nConnections for '{entity_name}':")

        # Find entity ID
        entity_row = self.entities[self.entities['name'] == entity_name]
        if len(entity_row) == 0:
            print(f"  Entity '{entity_name}' not found")
            return

        entity_id = entity_row['id'].values[0]

        # Outgoing relations
        outgoing = self.relations[self.relations['source'] == entity_id]
        if len(outgoing) > 0:
            print(f"\n  Outgoing relations:")
            for _, row in outgoing.iterrows():
                target = self.entities[self.entities['id'] == row['target']]['name'].values[0]
                print(f"    - {row['relation']} -> {target}")

        # Incoming relations
        incoming = self.relations[self.relations['target'] == entity_id]
        if len(incoming) > 0:
            print(f"\n  Incoming relations:")
            for _, row in incoming.iterrows():
                source = self.entities[self.entities['id'] == row['source']]['name'].values[0]
                print(f"    - {source} -{row['relation']}->")

    def compute_entity_importance(self):
        """Compute importance scores for entities"""
        print("\n" + "="*60)
        print("ENTITY IMPORTANCE ANALYSIS")
        print("="*60)

        # PageRank
        pagerank = nx.pagerank(self.G)

        # In-degree (how many point to this entity)
        in_degree = dict(self.G.in_degree())

        # Out-degree (how many this entity points to)
        out_degree = dict(self.G.out_degree())

        # Betweenness centrality
        betweenness = nx.betweenness_centrality(self.G)

        # Add to entities dataframe
        self.entities['pagerank'] = self.entities['id'].map(pagerank)
        self.entities['in_degree'] = self.entities['id'].map(in_degree)
        self.entities['out_degree'] = self.entities['id'].map(out_degree)
        self.entities['betweenness'] = self.entities['id'].map(betweenness)

        # Top entities by PageRank
        print("\nTop 10 Most Important Entities (by PageRank):")
        top_entities = self.entities.nlargest(10, 'pagerank')[
            ['name', 'type', 'pagerank', 'in_degree', 'out_degree']
        ]
        print(top_entities.to_string(index=False))

        # Importance by entity type
        print("\nAverage PageRank by Entity Type:")
        type_importance = self.entities.groupby('type')['pagerank'].mean().sort_values(ascending=False)
        print(type_importance)

        return pagerank

    def analyze_relation_patterns(self):
        """Analyze relation patterns"""
        print("\n" + "="*60)
        print("RELATION PATTERN ANALYSIS")
        print("="*60)

        # Relation frequency
        relation_counts = self.relations['relation'].value_counts()
        print("\nRelation Frequencies:")
        print(relation_counts)

        # Entity type pairs
        print("\nMost Common Entity Type Pairs:")
        type_pairs = self.relations.groupby(['source_type', 'target_type']).size()
        type_pairs = type_pairs.sort_values(ascending=False).head(10)
        print(type_pairs)

        return relation_counts

    def visualize_knowledge_graph(self):
        """Visualize knowledge graph"""
        fig, axes = plt.subplots(2, 2, figsize=(18, 14))

        # 1. Full knowledge graph
        ax = axes[0, 0]

        # Use smaller subset for visualization clarity
        # Get nodes with highest PageRank
        top_nodes = self.entities.nlargest(30, 'pagerank')['id'].values

        subgraph = self.G.subgraph(top_nodes)
        pos = nx.spring_layout(subgraph, k=2, iterations=50, seed=self.seed)

        # Node colors by type
        type_colors = {'Person': 0, 'Organization': 1, 'Location': 2,
                      'Technology': 3, 'Product': 4, 'Concept': 5}
        node_colors = [type_colors.get(self.G.nodes[node].get('type', 'Unknown'), 6)
                      for node in subgraph.nodes()]

        # Node sizes by PageRank
        node_sizes = [self.entities[self.entities['id'] == node]['pagerank'].values[0] * 5000
                     for node in subgraph.nodes()]

        nx.draw_networkx_nodes(subgraph, pos, node_color=node_colors, cmap='Set3',
                              node_size=node_sizes, ax=ax, alpha=0.8)
        nx.draw_networkx_edges(subgraph, pos, edge_color='gray', alpha=0.3,
                              arrows=True, arrowsize=10, ax=ax)

        # Labels
        labels = {node: self.G.nodes[node]['name'] for node in subgraph.nodes()}
        nx.draw_networkx_labels(subgraph, pos, labels, font_size=8, ax=ax)

        ax.set_title('Knowledge Graph (Top 30 Entities)', fontsize=14, fontweight='bold')
        ax.axis('off')

        # 2. Entity type distribution
        ax = axes[0, 1]
        type_counts = self.entities['type'].value_counts()
        ax.bar(type_counts.index, type_counts.values, color='steelblue', alpha=0.8)
        ax.set_xlabel('Entity Type', fontsize=11)
        ax.set_ylabel('Count', fontsize=11)
        ax.set_title('Entity Type Distribution', fontsize=14, fontweight='bold')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3, axis='y')

        # 3. Relation type distribution
        ax = axes[1, 0]
        relation_counts = self.relations['relation'].value_counts()
        ax.barh(range(len(relation_counts)), relation_counts.values, color='coral', alpha=0.8)
        ax.set_yticks(range(len(relation_counts)))
        ax.set_yticklabels(relation_counts.index, fontsize=9)
        ax.set_xlabel('Count', fontsize=11)
        ax.set_title('Relation Type Distribution', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')

        # 4. Top entities by PageRank
        ax = axes[1, 1]
        top_entities = self.entities.nlargest(10, 'pagerank')
        ax.barh(range(len(top_entities)), top_entities['pagerank'].values,
               color='mediumseagreen', alpha=0.8)
        ax.set_yticks(range(len(top_entities)))
        ax.set_yticklabels(top_entities['name'].values, fontsize=9)
        ax.set_xlabel('PageRank Score', fontsize=11)
        ax.set_title('Top 10 Entities by Importance', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')

        plt.tight_layout()
        plt.savefig('knowledge_graph_analysis.png', dpi=300, bbox_inches='tight')
        print("\nVisualization saved as 'knowledge_graph_analysis.png'")
        plt.close()

def main():
    """Main execution function"""
    print("="*60)
    print("KNOWLEDGE GRAPH CONSTRUCTION AND ANALYSIS")
    print("="*60)

    # Initialize builder
    builder = KnowledgeGraphBuilder(seed=42)

    # Generate knowledge graph
    G, entities, relations = builder.generate_knowledge_graph(n_entities=80)

    # Analyze structure
    structure_stats = builder.analyze_graph_structure()

    # Query knowledge graph
    builder.query_knowledge_graph()

    # Find connections for specific entities
    builder.find_entity_connections('Alice')
    builder.find_entity_connections('TechCorp')

    # Compute entity importance
    pagerank = builder.compute_entity_importance()

    # Analyze relation patterns
    relation_patterns = builder.analyze_relation_patterns()

    # Visualize
    builder.visualize_knowledge_graph()

    # Save results
    entities_export = builder.entities.copy()
    relations_export = builder.relations.copy()

    entities_export.to_csv('knowledge_graph_entities.csv', index=False)
    relations_export.to_csv('knowledge_graph_relations.csv', index=False)
    print("\nData saved to CSV files")

    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print(f"Total entities: {len(entities_export)}")
    print(f"Total relations: {len(relations_export)}")
    print(f"Unique relation types: {relations_export['relation'].nunique()}")

if __name__ == "__main__":
    main()
