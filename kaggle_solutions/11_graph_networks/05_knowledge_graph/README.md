# Knowledge Graph Construction

## Overview
This solution constructs and analyzes knowledge graphs with entities and typed relationships.

## Problem Statement
Knowledge graphs are essential for:
- Semantic search and information retrieval
- Question answering systems
- Recommendation engines
- Data integration and linking
- AI reasoning systems

## Approach

### 1. Entity Creation
Multiple entity types:
- **Person**: Individuals
- **Organization**: Companies and institutions
- **Location**: Geographic places
- **Technology**: Technical domains
- **Product**: Goods and services
- **Concept**: Abstract ideas

### 2. Relation Definition
Typed relationships between entities:
- `works_for`: Person → Organization
- `located_in`: Entity → Location
- `develops`: Entity → Product
- `uses`: Entity → Technology
- `collaborates_with`: Entity → Entity
- `specializes_in`: Person → Domain
- `related_to`: Entity → Entity

### 3. Graph Construction
- MultiDiGraph for multiple edge types
- Directed edges with relation attributes
- Enforced type constraints
- Realistic relation patterns

### 4. Knowledge Queries
- Path-based queries
- Pattern matching
- Entity connection discovery
- Relation traversal

### 5. Entity Importance
- PageRank for influence
- Degree centrality
- Betweenness (bridge entities)
- Type-based aggregation

## Knowledge Graph Model

### Triple Structure
```
(Subject, Predicate, Object)
(Alice, works_for, TechCorp)
(TechCorp, located_in, San Francisco)
```

### Entity-Relation-Entity
All facts stored as triples forming connected graph.

## Key Findings

### Entity Importance
- Organizations are central hubs
- Persons connect through organizations
- Technologies link multiple entities

### Relation Patterns
- Most common: works_for, uses
- Technology relations are diverse
- Location relations are sparse

### Graph Structure
- Multiple connected components possible
- Hub-and-spoke patterns
- Type-specific subgraphs

## Visualizations
1. **Knowledge Graph**: Top entities visualization
2. **Entity Distribution**: Types and counts
3. **Relation Distribution**: Relation frequencies
4. **Entity Importance**: PageRank rankings

## Output Files
- `knowledge_graph_entities.csv`: All entities with metrics
- `knowledge_graph_relations.csv`: All relations/triples
- `knowledge_graph_analysis.png`: Visualizations

## Requirements
```
networkx
numpy
pandas
matplotlib
seaborn
```

## Usage
```bash
python solution.py
```

## Real-World Applications
- **Google Knowledge Graph**: Search enhancement
- **Enterprise KG**: Data integration
- **Medical KG**: Drug-disease relationships
- **E-commerce**: Product recommendations
- **Research**: Citation and concept linking

## Key Insights
- Typed relations enable semantic queries
- Entity importance varies by type
- Graph structure reveals hidden patterns
- Multiple hops discover connections

## Extensions
- Ontology integration
- Temporal knowledge graphs
- Probabilistic relations
- Graph embeddings (TransE, DistMult)
- SPARQL query support
- RDF/OWL export
