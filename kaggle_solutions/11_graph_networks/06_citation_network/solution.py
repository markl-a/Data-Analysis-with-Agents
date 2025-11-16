"""
Citation Network Analysis - Kaggle Solution
===========================================
Analyzes academic citation networks and identifies influential papers.
"""

import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

class CitationNetworkAnalyzer:
    """Citation network analysis toolkit"""

    def __init__(self, n_papers=100, seed=42):
        """
        Initialize citation network analyzer

        Args:
            n_papers: Number of papers
            seed: Random seed
        """
        self.n_papers = n_papers
        self.seed = seed
        np.random.seed(seed)
        self.G = None
        self.papers = None

    def generate_citation_network(self):
        """Generate realistic citation network"""
        print("Generating citation network...")

        # Create directed graph (citations point backward in time)
        self.G = nx.DiGraph()

        # Generate papers with metadata
        fields = ['Machine Learning', 'Computer Vision', 'NLP', 'Robotics', 'Theory']
        start_date = datetime(2015, 1, 1)

        papers_data = []
        for i in range(self.n_papers):
            # Papers published over time
            pub_date = start_date + timedelta(days=i * 30)
            field = np.random.choice(fields, p=[0.3, 0.25, 0.25, 0.1, 0.1])

            # Generate citations (can only cite earlier papers)
            if i > 0:
                # Number of references (5-15 refs typical)
                n_refs = min(np.random.randint(5, 15), i)

                # Cite recent papers with higher probability
                earlier_papers = list(range(i))
                weights = np.exp(-0.05 * np.array([i - j for j in earlier_papers]))
                weights = weights / weights.sum()

                cited_papers = np.random.choice(earlier_papers, size=n_refs,
                                               replace=False, p=weights)

                for cited in cited_papers:
                    # Prefer same field citations (70%)
                    if np.random.rand() < 0.7:
                        # Check if same field, otherwise skip sometimes
                        cited_field = self.G.nodes[cited]['field'] if cited in self.G.nodes else field
                        if cited_field != field and np.random.rand() < 0.5:
                            continue

                    self.G.add_edge(i, cited)  # i cites cited

            # Add node with attributes
            self.G.add_node(i,
                          title=f"Paper_{i}",
                          field=field,
                          year=pub_date.year,
                          pub_date=pub_date)

            papers_data.append({
                'paper_id': i,
                'title': f"Paper_{i}",
                'field': field,
                'year': pub_date.year,
                'pub_date': pub_date
            })

        self.papers = pd.DataFrame(papers_data)

        # Calculate citation counts
        citations_received = dict(self.G.in_degree())  # Papers citing this one
        citations_made = dict(self.G.out_degree())  # Papers this one cites

        self.papers['citations_received'] = self.papers['paper_id'].map(citations_received)
        self.papers['references_made'] = self.papers['paper_id'].map(citations_made)

        print(f"Created citation network with {self.G.number_of_nodes()} papers")
        print(f"Total citations: {self.G.number_of_edges()}")
        print(f"\nField distribution:")
        print(self.papers['field'].value_counts())

        return self.G, self.papers

    def analyze_citation_patterns(self):
        """Analyze citation patterns"""
        print("\n" + "="*60)
        print("CITATION PATTERN ANALYSIS")
        print("="*60)

        # Basic statistics
        print(f"\nCitation Statistics:")
        print(f"  Total papers: {self.G.number_of_nodes()}")
        print(f"  Total citations: {self.G.number_of_edges()}")
        print(f"  Avg citations per paper: {self.papers['citations_received'].mean():.2f}")
        print(f"  Avg references per paper: {self.papers['references_made'].mean():.2f}")

        # Most cited papers
        print(f"\nTop 10 Most Cited Papers:")
        top_cited = self.papers.nlargest(10, 'citations_received')[
            ['title', 'field', 'year', 'citations_received']
        ]
        print(top_cited.to_string(index=False))

        # Citations by field
        print(f"\nCitations by Field:")
        field_stats = self.papers.groupby('field')['citations_received'].agg(['mean', 'sum', 'max'])
        print(field_stats.round(2))

        # Citations by year
        print(f"\nCitations by Year:")
        year_stats = self.papers.groupby('year')['citations_received'].mean()
        print(year_stats.round(2))

        return field_stats

    def identify_influential_papers(self):
        """Identify influential papers using various metrics"""
        print("\n" + "="*60)
        print("INFLUENTIAL PAPERS IDENTIFICATION")
        print("="*60)

        # PageRank (influence through citation chain)
        pagerank = nx.pagerank(self.G)

        # HITS (hubs and authorities)
        hubs, authorities = nx.hits(self.G)

        # Betweenness (bridge papers)
        betweenness = nx.betweenness_centrality(self.G)

        # Add to dataframe
        self.papers['pagerank'] = self.papers['paper_id'].map(pagerank)
        self.papers['authority_score'] = self.papers['paper_id'].map(authorities)
        self.papers['hub_score'] = self.papers['paper_id'].map(hubs)
        self.papers['betweenness'] = self.papers['paper_id'].map(betweenness)

        # Calculate h-index like metric
        self.papers['h_index'] = self.papers.apply(
            lambda row: self._calculate_h_index(row['paper_id']), axis=1
        )

        print("\nTop 10 Papers by Different Metrics:")

        print("\n1. PageRank (Overall Influence):")
        print(self.papers.nlargest(5, 'pagerank')[['title', 'field', 'pagerank']].to_string(index=False))

        print("\n2. Authority Score (Well-Cited):")
        print(self.papers.nlargest(5, 'authority_score')[['title', 'field', 'authority_score']].to_string(index=False))

        print("\n3. Hub Score (Good References):")
        print(self.papers.nlargest(5, 'hub_score')[['title', 'field', 'hub_score']].to_string(index=False))

        return pagerank

    def _calculate_h_index(self, paper_id):
        """Calculate simple h-index for a paper based on its citations"""
        # Get papers citing this paper
        citers = list(self.G.predecessors(paper_id))
        if not citers:
            return 0

        # Get their citation counts
        citer_citations = [self.G.in_degree(c) for c in citers]
        citer_citations.sort(reverse=True)

        # Calculate h-index
        h = 0
        for i, citations in enumerate(citer_citations, 1):
            if citations >= i:
                h = i
            else:
                break

        return h

    def analyze_citation_cascade(self):
        """Analyze citation cascades and impact propagation"""
        print("\n" + "="*60)
        print("CITATION CASCADE ANALYSIS")
        print("="*60)

        # Find papers with highest citation cascades
        # (citations of papers that cite this paper)
        cascade_impact = {}

        for paper_id in range(self.n_papers):
            # Direct citations
            direct_citers = list(self.G.predecessors(paper_id))

            # Indirect citations (papers citing papers that cite this)
            indirect_citers = set()
            for citer in direct_citers:
                indirect_citers.update(self.G.predecessors(citer))

            cascade_impact[paper_id] = len(indirect_citers)

        self.papers['cascade_impact'] = self.papers['paper_id'].map(cascade_impact)

        print("\nTop 10 Papers by Citation Cascade:")
        top_cascade = self.papers.nlargest(10, 'cascade_impact')[
            ['title', 'field', 'citations_received', 'cascade_impact']
        ]
        print(top_cascade.to_string(index=False))

        return cascade_impact

    def detect_research_communities(self):
        """Detect research communities/clusters"""
        print("\n" + "="*60)
        print("RESEARCH COMMUNITY DETECTION")
        print("="*60)

        # Convert to undirected for community detection
        G_undirected = self.G.to_undirected()

        from networkx.algorithms import community
        communities = community.greedy_modularity_communities(G_undirected)

        # Map communities
        comm_map = {}
        for idx, comm in enumerate(communities):
            for paper in comm:
                comm_map[paper] = idx

        self.papers['community'] = self.papers['paper_id'].map(comm_map)

        print(f"Number of research communities detected: {len(communities)}")

        # Analyze communities
        print("\nCommunity Characteristics:")
        for comm_id in range(min(5, len(communities))):
            comm_papers = self.papers[self.papers['community'] == comm_id]
            dominant_field = comm_papers['field'].mode()[0] if len(comm_papers) > 0 else 'Unknown'
            print(f"  Community {comm_id}: {len(comm_papers)} papers, dominant field: {dominant_field}")

        return communities

    def visualize_citation_network(self):
        """Visualize citation network"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 14))

        # 1. Citation network (top papers)
        ax = axes[0, 0]

        # Select top 40 papers by PageRank for clarity
        top_papers = self.papers.nlargest(40, 'pagerank')['paper_id'].values
        subgraph = self.G.subgraph(top_papers)

        pos = nx.spring_layout(subgraph, k=1.5, iterations=50, seed=self.seed)

        # Node colors by field
        field_colors = {'Machine Learning': 0, 'Computer Vision': 1, 'NLP': 2,
                       'Robotics': 3, 'Theory': 4}
        node_colors = [field_colors.get(self.G.nodes[node].get('field', ''), 5)
                      for node in subgraph.nodes()]

        # Node sizes by citations
        node_sizes = [self.papers[self.papers['paper_id'] == node]['citations_received'].values[0] * 20 + 50
                     for node in subgraph.nodes()]

        nx.draw_networkx_nodes(subgraph, pos, node_color=node_colors, cmap='Set3',
                              node_size=node_sizes, ax=ax, alpha=0.7)
        nx.draw_networkx_edges(subgraph, pos, edge_color='gray', alpha=0.2,
                              arrows=True, arrowsize=8, ax=ax)

        ax.set_title('Citation Network (Top 40 Papers)', fontsize=14, fontweight='bold')
        ax.axis('off')

        # 2. Citation distribution
        ax = axes[0, 1]
        ax.hist(self.papers['citations_received'], bins=20, edgecolor='black',
               alpha=0.7, color='steelblue')
        ax.set_xlabel('Number of Citations', fontsize=11)
        ax.set_ylabel('Number of Papers', fontsize=11)
        ax.set_title('Citation Distribution', fontsize=14, fontweight='bold')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)

        # 3. Citations by field
        ax = axes[1, 0]
        field_citations = self.papers.groupby('field')['citations_received'].mean().sort_values()
        ax.barh(range(len(field_citations)), field_citations.values,
               color='coral', alpha=0.8)
        ax.set_yticks(range(len(field_citations)))
        ax.set_yticklabels(field_citations.index)
        ax.set_xlabel('Average Citations', fontsize=11)
        ax.set_title('Average Citations by Field', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')

        # 4. Citations over time
        ax = axes[1, 1]
        year_citations = self.papers.groupby('year')['citations_received'].mean()
        ax.plot(year_citations.index, year_citations.values, marker='o',
               linewidth=2, markersize=8, color='steelblue')
        ax.set_xlabel('Publication Year', fontsize=11)
        ax.set_ylabel('Average Citations', fontsize=11)
        ax.set_title('Average Citations by Publication Year', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('citation_network_analysis.png', dpi=300, bbox_inches='tight')
        print("\nVisualization saved as 'citation_network_analysis.png'")
        plt.close()

def main():
    """Main execution function"""
    print("="*60)
    print("CITATION NETWORK ANALYSIS")
    print("="*60)

    # Initialize analyzer
    analyzer = CitationNetworkAnalyzer(n_papers=100, seed=42)

    # Generate citation network
    G, papers = analyzer.generate_citation_network()

    # Analyze citation patterns
    field_stats = analyzer.analyze_citation_patterns()

    # Identify influential papers
    pagerank = analyzer.identify_influential_papers()

    # Analyze citation cascades
    cascade_impact = analyzer.analyze_citation_cascade()

    # Detect research communities
    communities = analyzer.detect_research_communities()

    # Visualize
    analyzer.visualize_citation_network()

    # Save results
    papers_export = analyzer.papers.copy()
    papers_export.to_csv('citation_network_papers.csv', index=False)
    print("\nPaper data saved to 'citation_network_papers.csv'")

    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print(f"Total papers: {len(papers_export)}")
    print(f"Total citations: {G.number_of_edges()}")
    most_cited = papers_export.nlargest(1, 'citations_received')
    print(f"Most cited paper: {most_cited['title'].values[0]} ({most_cited['citations_received'].values[0]} citations)")

if __name__ == "__main__":
    main()
