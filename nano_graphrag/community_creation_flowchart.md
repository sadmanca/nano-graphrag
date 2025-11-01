# Community Creation Flowchart

This document provides detailed flowcharts and explanations for the community creation process in nano-graphrag, covering both graph clustering and community report generation.

---

## High-Level Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                   Knowledge Graph (Entities & Relationships)    │
│                   - Nodes: Entities extracted from documents    │
│                   - Edges: Relationships between entities       │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                    PHASE 1: GRAPH CLUSTERING                    │
│              (Using Hierarchical Leiden Algorithm)              │
│                                                                 │
│  Input:  Knowledge graph with nodes and edges                  │
│  Output: Hierarchical community structure                      │
│          (Each node assigned to communities at multiple levels) │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│               PHASE 2: COMMUNITY REPORT GENERATION              │
│                    (Using LLM to Summarize)                     │
│                                                                 │
│  Input:  Community structure + nodes + edges                   │
│  Output: Natural language reports for each community           │
└─────────────────────────────────────────────────────────────────┘
```

---

## Phase 1: Graph Clustering - Detailed Flowchart

```
START: Knowledge Graph Available
│
├─ Step 1: Identify Connected Components
│  │
│  ├─ Action: Find all connected components in graph
│  │          nx.connected_components(graph)
│  │
│  ├─ Why: Disconnected graph parts need separate clustering
│  │        Each component may represent different topic areas
│  │
│  └─ Result: List of component subgraphs
│             Component 1: 150 nodes (e.g., ML papers)
│             Component 2: 80 nodes (e.g., Biology papers)
│             Component 3: 20 nodes (e.g., isolated concepts)
│
├─ Step 2: Process Each Component Independently
│  │
│  │  FOR EACH connected component:
│  │  │
│  │  ├─ Step 2a: Stabilize Graph
│  │  │  │
│  │  │  ├─ Action: Sort nodes and edges deterministically
│  │  │  │         Normalize node names (uppercase, unescape HTML)
│  │  │  │
│  │  │  ├─ Why: Ensures reproducible results across runs
│  │  │  │
│  │  │  └─ Code: gdb_networkx.py:218-222
│  │  │
│  │  ├─ Step 2b: Apply Hierarchical Leiden Algorithm
│  │  │  │
│  │  │  └─ HIERARCHICAL LEIDEN PROCESS (See detailed flowchart below)
│  │  │
│  │  └─ Step 2c: Assign Cluster IDs with Offset
│  │     │
│  │     ├─ Action: Add cluster_offset to all cluster IDs
│  │     │         cluster_id = partition.cluster + cluster_offset
│  │     │
│  │     ├─ Why: Prevents ID collisions between components
│  │     │       Component 1 clusters: 0, 1, 2, 3
│  │     │       Component 2 clusters: 4, 5, 6 (offset by 4)
│  │     │
│  │     └─ Result: Unique cluster IDs across all components
│  │
│  └─ NEXT component
│
├─ Step 3: Store Cluster Assignments
│  │
│  ├─ Action: For each node, store JSON array of cluster assignments
│  │         node["clusters"] = json.dumps([
│  │             {"level": 0, "cluster": "5"},
│  │             {"level": 1, "cluster": "2"},
│  │             {"level": 2, "cluster": "0"}
│  │         ])
│  │
│  ├─ Why: Each node belongs to one community at each level
│  │       Level 0: Most specific community
│  │       Level N: Most general community
│  │
│  └─ Code: gdb_networkx.py:270, gdb_networkx.py:196-198
│
└─ END: Hierarchical community structure created
```

---

## Hierarchical Leiden Algorithm - Detailed Process

```
START: Single Connected Component Graph
│     (e.g., 50 nodes, max_cluster_size = 10)
│
├─ LEVEL 0: Create Base Communities
│  │
│  ├─ Step 1: Run Leiden Clustering on Original Graph
│  │  │
│  │  ├─ Algorithm: Leiden community detection
│  │  │  │
│  │  │  ├─ Initialize: Each node in its own community
│  │  │  │
│  │  │  ├─ Optimize: Move nodes between communities to maximize modularity
│  │  │  │            Modularity = measure of how well-separated communities are
│  │  │  │            Formula: Q = Σ [e_ii - (a_i)²]
│  │  │  │            where e_ii = fraction of edges within community i
│  │  │  │                  a_i = fraction of edge endpoints in community i
│  │  │  │
│  │  │  ├─ Constraint: Respect max_cluster_size (default: 10)
│  │  │  │             If community exceeds size, split it
│  │  │  │
│  │  │  └─ Refine: Local moves to improve modularity
│  │  │
│  │  └─ Result: Level 0 communities
│  │            Example: 8 communities
│  │            - Community 0: [node1, node2, node5, node8] (4 nodes)
│  │            - Community 1: [node3, node4, node6] (3 nodes)
│  │            - Community 2: [node7, node9, node10, node11, node12] (5 nodes)
│  │            - Community 3: [node13, node14, ...] (10 nodes - at max)
│  │            - ... (8 communities total)
│  │
│  └─ Store: Mark each node with Level 0 cluster assignment
│
├─ LEVEL 1: Group Level 0 Communities
│  │
│  ├─ Step 1: Create Super-Graph
│  │  │
│  │  ├─ Action: Each Level 0 community becomes a single "super-node"
│  │  │         Original: 50 nodes → Super-graph: 8 nodes
│  │  │
│  │  ├─ Edge Weights: Count connections between communities
│  │  │               If Community 0 and Community 1 share 12 edges
│  │  │               → Super-edge weight = 12
│  │  │
│  │  └─ Example Super-Graph:
│  │            SuperNode0 (represents Community 0) ←--12-→ SuperNode1
│  │            SuperNode0 ←--3-→ SuperNode2
│  │            SuperNode1 ←--8-→ SuperNode3
│  │            ... etc
│  │
│  ├─ Step 2: Run Leiden Clustering on Super-Graph
│  │  │
│  │  ├─ Algorithm: Same Leiden process as Level 0
│  │  │            But now clustering 8 super-nodes instead of 50 nodes
│  │  │
│  │  └─ Result: Level 1 communities (groups of Level 0 communities)
│  │            Example: 3 Level 1 communities
│  │            - Level1_Community 0: [Community 0, Community 1, Community 2]
│  │            - Level1_Community 1: [Community 3, Community 4, Community 5]
│  │            - Level1_Community 2: [Community 6, Community 7]
│  │
│  └─ Store: Mark each node with Level 1 cluster assignment
│            (inherited from their Level 0 community)
│
├─ LEVEL 2: Group Level 1 Communities
│  │
│  ├─ Step 1: Create Level 1 Super-Graph
│  │  │
│  │  └─ Action: Each Level 1 community becomes a super-node
│  │            Now: 3 super-nodes
│  │
│  ├─ Step 2: Run Leiden Clustering on Level 1 Super-Graph
│  │  │
│  │  └─ Result: Level 2 communities
│  │            Example: 1 Level 2 community (entire graph)
│  │            - Level2_Community 0: [Level1_Community 0, 1, 2]
│  │
│  └─ Store: Mark each node with Level 2 cluster assignment
│
├─ TERMINATION CHECK
│  │
│  ├─ Condition 1: Only 1 community at this level?
│  │              → STOP (entire graph is now one community)
│  │
│  ├─ Condition 2: No improvement in modularity?
│  │              → STOP (no meaningful groupings possible)
│  │
│  ├─ Condition 3: Super-graph too small to cluster?
│  │              → STOP (< 2 nodes in super-graph)
│  │
│  └─ If none of above: Continue to next level
│
└─ END: Hierarchical clustering complete
         Each node assigned to 1 community per level (0 through N)

         FINAL OUTPUT:
         Node1: [Level0: Cluster5, Level1: Cluster2, Level2: Cluster0]
         Node2: [Level0: Cluster5, Level1: Cluster2, Level2: Cluster0]
         Node3: [Level0: Cluster7, Level1: Cluster1, Level2: Cluster0]
         ... etc
```

---

## Example: 50-Node Graph Clustering

```
INITIAL GRAPH: 50 nodes connected by relationships

LEVEL 0 (Finest Granularity):
┌─────────────────────────────────────────────────────────────────┐
│ Run Leiden with max_cluster_size = 10                          │
│                                                                 │
│ Result: 8 communities                                          │
│ ┌────────┐ ┌────────┐ ┌─────────┐ ┌────────┐                 │
│ │ Comm 0 │ │ Comm 1 │ │ Comm 2  │ │ Comm 3 │ ...             │
│ │4 nodes │ │3 nodes │ │5 nodes  │ │10 nodes│                 │
│ └────────┘ └────────┘ └─────────┘ └────────┘                 │
│                                                                 │
│ Example: Community 0 = Deep Learning Optimization              │
│          Community 1 = Transformer Architecture                │
│          Community 2 = Training Techniques                     │
└─────────────────────────────────────────────────────────────────┘

LEVEL 1 (Medium Granularity):
┌─────────────────────────────────────────────────────────────────┐
│ Create super-graph: 8 super-nodes (one per Level 0 community)  │
│ Run Leiden on super-graph                                      │
│                                                                 │
│ Result: 3 Level 1 communities                                  │
│ ┌──────────────────┐ ┌──────────────┐ ┌──────────────┐       │
│ │   Level1_Comm0   │ │ Level1_Comm1 │ │ Level1_Comm2 │       │
│ │  [Comm0,1,2]     │ │  [Comm3,4,5] │ │  [Comm6,7]   │       │
│ │  12 nodes total  │ │  20 nodes    │ │  18 nodes    │       │
│ └──────────────────┘ └──────────────┘ └──────────────┘       │
│                                                                 │
│ Example: Level1_Comm0 = Neural Network Methods                 │
│          Level1_Comm1 = Machine Learning Foundations           │
│          Level1_Comm2 = Practical Applications                 │
└─────────────────────────────────────────────────────────────────┘

LEVEL 2 (Coarsest Granularity):
┌─────────────────────────────────────────────────────────────────┐
│ Create super-graph: 3 super-nodes (one per Level 1 community)  │
│ Run Leiden on super-graph                                      │
│                                                                 │
│ Result: 1 Level 2 community                                    │
│ ┌────────────────────────────────────────────────────┐         │
│ │              Level2_Comm0                          │         │
│ │         [Level1_Comm0, 1, 2]                       │         │
│ │            50 nodes total                          │         │
│ └────────────────────────────────────────────────────┘         │
│                                                                 │
│ Example: Level2_Comm0 = Artificial Intelligence                │
│                                                                 │
│ TERMINATION: Only 1 community at this level → STOP             │
└─────────────────────────────────────────────────────────────────┘
```

---

## Phase 2: Community Report Generation - Detailed Flowchart

```
START: Hierarchical community structure available
│
├─ Step 1: Extract Community Schema
│  │
│  ├─ Action: Build dictionary of all communities across all levels
│  │         community_schema = await knowledge_graph.community_schema()
│  │
│  ├─ Process: For each node in graph:
│  │  │       - Parse node["clusters"] JSON
│  │  │       - For each level/cluster assignment:
│  │  │         - Add node to that community's node list
│  │  │         - Add node's edges to that community's edge list
│  │  │         - Add node's source chunks to community's chunk list
│  │  │
│  │  └─ Code: gdb_networkx.py:140-194
│  │
│  └─ Result: Dictionary of communities
│            {
│              "0": {  # Cluster ID
│                "level": 0,
│                "title": "Cluster 0",
│                "nodes": ["ENTITY1", "ENTITY2", ...],
│                "edges": [["ENTITY1", "ENTITY2"], ...],
│                "chunk_ids": ["chunk-abc", "chunk-def", ...],
│                "occurrence": 0.15,  # Fraction of all chunks
│                "sub_communities": ["5", "7"]  # If higher level
│              },
│              "1": { ... },
│              ...
│            }
│
├─ Step 2: Identify Hierarchy Levels
│  │
│  ├─ Action: Extract unique levels from all communities
│  │         levels = sorted(set([c["level"] for c in communities]))
│  │         # e.g., [0, 1, 2]
│  │
│  ├─ Why: Reports generated level-by-level from highest to lowest
│  │       Higher levels can reference lower-level reports
│  │
│  └─ Code: _op.py:965
│
├─ Step 3: Generate Reports Level-by-Level (Highest to Lowest)
│  │
│  │  FOR EACH level in reversed(levels):  # [2, 1, 0]
│  │  │
│  │  ├─ Step 3a: Select Communities at This Level
│  │  │  │
│  │  │  └─ Action: Filter communities where community["level"] == current_level
│  │  │
│  │  ├─ Step 3b: Generate Report for Each Community
│  │  │  │
│  │  │  │  FOR EACH community at this level:
│  │  │  │  │
│  │  │  │  └─ COMMUNITY REPORT GENERATION PROCESS (see below)
│  │  │  │
│  │  │  └─ NEXT community
│  │  │
│  │  └─ Step 3c: Store Reports for This Level
│  │     │
│  │     ├─ Action: Save all reports to community_reports storage
│  │     │         Each report includes:
│  │     │         - report_string: Markdown formatted text
│  │     │         - report_json: Structured data (title, summary, findings)
│  │     │         - community metadata (level, nodes, edges, etc.)
│  │     │
│  │     └─ Code: _op.py:982-997
│  │
│  └─ NEXT level
│
└─ END: All community reports generated
```

---

## Community Report Generation Process - Single Community

```
START: Generate report for Community X
│
├─ Step 1: Gather Community Data
│  │
│  ├─ Action: Collect nodes and edges belonging to this community
│  │         nodes = community["nodes"]  # e.g., ["ENTITY1", "ENTITY2", ...]
│  │         edges = community["edges"]  # e.g., [["ENTITY1", "ENTITY2"], ...]
│  │
│  └─ Code: _op.py:798-799
│
├─ Step 2: Retrieve Full Node and Edge Details
│  │
│  ├─ Action: Query graph storage for complete information
│  │         nodes_data = await asyncio.gather(*[
│  │             knowledge_graph.get_node(n) for n in nodes
│  │         ])
│  │         # Returns: [{entity_type: "...", description: "...", ...}, ...]
│  │
│  │         edges_data = await asyncio.gather(*[
│  │             knowledge_graph.get_edge(src, tgt) for src, tgt in edges
│  │         ])
│  │         # Returns: [{weight: 1.5, description: "...", ...}, ...]
│  │
│  └─ Code: _op.py:801-806
│
├─ Step 3: Rank and Prepare Node Information
│  │
│  ├─ Action: Create CSV-style table with node details
│  │         For each node:
│  │         - Get node degree (number of connections)
│  │         - Format: [id, entity_name, type, description, degree]
│  │
│  ├─ Ranking: Sort by degree (most connected nodes first)
│  │          High-degree nodes are more important in community
│  │
│  ├─ Truncation: If total tokens exceed limit, keep highest-degree nodes
│  │             truncate_list_by_token_size(..., max_token_size // 2)
│  │
│  └─ Result: CSV table
│            id, entity, type, description, degree
│            0,  NEURAL NETWORK, CONCEPT, A computational model..., 15
│            1,  BACKPROPAGATION, CONCEPT, Training algorithm..., 12
│            ...
│
│  Code: _op.py:807-822
│
├─ Step 4: Rank and Prepare Edge Information
│  │
│  ├─ Action: Create CSV-style table with edge details
│  │         For each edge:
│  │         - Calculate rank = degree(source) + degree(target)
│  │         - Format: [id, source, target, description, weight, rank]
│  │
│  ├─ Ranking: Sort by rank (edges between important nodes first)
│  │
│  ├─ Truncation: If total tokens exceed limit, keep highest-rank edges
│  │             truncate_list_by_token_size(..., max_token_size // 2)
│  │
│  └─ Result: CSV table
│            id, source, target, description, weight, rank
│            0,  NEURAL NETWORK, BACKPROPAGATION, enables training, 1.0, 27
│            ...
│
│  Code: _op.py:823-836
│
├─ Step 5: Check Token Budget and Handle Sub-Communities
│  │
│  ├─ Decision: Did nodes/edges exceed token limit?
│  │           AND does community have sub-communities?
│  │           AND are sub-community reports already generated?
│  │
│  ├─ IF YES:
│  │  │
│  │  ├─ Action: Include sub-community reports instead of raw data
│  │  │         sub_reports = [
│  │  │             already_reports[sub_id] for sub_id in community["sub_communities"]
│  │  │         ]
│  │  │
│  │  ├─ Format: CSV table of sub-community summaries
│  │  │         id, report, rating, importance
│  │  │         0,  "# Title\nSummary...", 7.5, 0.12
│  │  │         1,  "# Title2\nSummary...", 8.2, 0.18
│  │  │
│  │  ├─ Benefit: Provides high-level context without overwhelming detail
│  │  │          Allows LLM to understand sub-structure
│  │  │
│  │  └─ Code: _op.py:844-881
│  │
│  └─ IF NO: Use raw nodes and edges data
│
├─ Step 6: Format Context for LLM
│  │
│  ├─ Action: Combine all information into structured text
│  │         Format:
│  │         -----Reports-----     (if using sub-communities)
│  │         ```csv
│  │         {sub_community_reports}
│  │         ```
│  │         -----Entities-----
│  │         ```csv
│  │         {nodes_table}
│  │         ```
│  │         -----Relationships-----
│  │         ```csv
│  │         {edges_table}
│  │         ```
│  │
│  └─ Code: _op.py:882-895
│
├─ Step 7: Generate Report with LLM
│  │
│  ├─ Prompt: PROMPTS["community_report"]
│  │         Instructs LLM to generate JSON with:
│  │         - title: Short, specific name for community
│  │         - summary: Executive summary of structure and relationships
│  │         - rating: Impact severity score (0-10)
│  │         - rating_explanation: One sentence explaining rating
│  │         - findings: List of 5-10 key insights
│  │           Each finding has:
│  │           - summary: Short insight summary
│  │           - explanation: Detailed multi-paragraph explanation
│  │
│  ├─ Context: Insert formatted CSV tables into prompt
│  │          prompt = community_report_prompt.format(input_text=context)
│  │
│  ├─ LLM Call: response = await best_model_func(prompt, **llm_kwargs)
│  │           With: response_format = {"type": "json_object"}
│  │
│  └─ Code: _op.py:950-951
│
├─ Step 8: Parse and Convert Response
│  │
│  ├─ Action: Parse JSON response from LLM
│  │         data = convert_response_to_json_func(response)
│  │         # Returns: {title: "...", summary: "...", rating: 7.5, ...}
│  │
│  ├─ Convert to String: Format JSON as readable markdown
│  │                    report_string = _community_report_json_to_str(data)
│  │                    # Returns:
│  │                    # # {title}
│  │                    #
│  │                    # {summary}
│  │                    #
│  │                    # ## {finding1.summary}
│  │                    # {finding1.explanation}
│  │                    # ...
│  │
│  └─ Code: _op.py:953, _op.py:898-917
│
├─ Step 9: Combine Report with Community Metadata
│  │
│  ├─ Action: Create complete community record
│  │         community_data = {
│  │             "report_string": report_string,
│  │             "report_json": data,
│  │             "level": community["level"],
│  │             "nodes": community["nodes"],
│  │             "edges": community["edges"],
│  │             "chunk_ids": community["chunk_ids"],
│  │             "occurrence": community["occurrence"],
│  │             "sub_communities": community["sub_communities"]
│  │         }
│  │
│  └─ Code: _op.py:982-995
│
└─ END: Community report complete
         Ready to be used for:
         - Direct retrieval in local search
         - Input to higher-level community reports
         - Synthesis in global search
```

---

## Token Budget Management in Report Generation

```
SCENARIO: Large community exceeds token limits
│
├─ Problem: Community has 100 nodes, 300 edges
│           Context would be ~20,000 tokens
│           LLM limit: 12,000 tokens (max_token_size // 2 per section)
│
├─ Solution 1: Truncate by Importance
│  │
│  ├─ Action: Keep only highest-degree nodes and edges
│  │         truncate_list_by_token_size(nodes, key=lambda x: x[description])
│  │
│  ├─ Result: Context reduced to fit limit
│  │         But: May lose important lower-degree nodes
│  │
│  └─ Code: _op.py:820-822, 834-836
│
└─ Solution 2: Use Sub-Community Reports (Preferred)
   │
   ├─ Condition: Community has sub-communities at lower level
   │            AND lower-level reports already generated
   │            AND truncation would be severe
   │
   ├─ Action: Replace raw nodes/edges with sub-community summaries
   │         Instead of 100 nodes → Include 5 sub-community reports
   │         Each sub-report already summarizes 20 nodes
   │
   ├─ Benefit: Hierarchical abstraction
   │          Higher-level report references lower-level findings
   │          Maintains all information through hierarchy
   │          Fits within token budget
   │
   ├─ Example:
   │  │
   │  │ LEVEL 0 Report (Fine-grained):
   │  │   Title: "Deep Learning Optimization Techniques"
   │  │   Summary: Detailed analysis of specific algorithms
   │  │   Findings: Adam optimizer, learning rate scheduling, etc.
   │  │
   │  │ LEVEL 1 Report (Uses Level 0 as context):
   │  │   Title: "Neural Network Training Methods"
   │  │   Summary: "This community encompasses several optimization
   │  │            approaches, including the Deep Learning Optimization
   │  │            Techniques community which focuses on adaptive methods..."
   │  │   Findings: Broader patterns across sub-communities
   │  │
   │  └─ Code: _op.py:844-881
   │
   └─ Result: Complete, hierarchical understanding without token overflow
```

---

## Data Flow: From Graph to Reports

```
KNOWLEDGE GRAPH
├─ Nodes: ["ENTITY1", "ENTITY2", "ENTITY3", ..., "ENTITY50"]
└─ Edges: [("ENTITY1", "ENTITY2"), ("ENTITY2", "ENTITY3"), ...]

        │ Clustering (Leiden Algorithm)
        ▼

HIERARCHICAL STRUCTURE
├─ Level 0 (8 communities)
│  ├─ Community "0": nodes=[ENTITY1, ENTITY2, ENTITY5], edges=[...]
│  ├─ Community "1": nodes=[ENTITY3, ENTITY4], edges=[...]
│  └─ ...
├─ Level 1 (3 communities)
│  ├─ Community "0": sub_communities=["0", "1", "2"]
│  └─ ...
└─ Level 2 (1 community)
   └─ Community "0": sub_communities=["0", "1", "2"]

        │ Report Generation (Level 2 → Level 0)
        ▼

LEVEL 2 REPORT
├─ Title: "Artificial Intelligence Research"
├─ Summary: "Comprehensive overview of AI field..."
├─ Rating: 9.2
└─ Findings: [5-10 high-level insights referencing Level 1 communities]

        │ (Generated first, used by lower levels)
        ▼

LEVEL 1 REPORTS (3 reports)
├─ Report "0": "Neural Network Methods"
├─ Report "1": "Machine Learning Foundations"
└─ Report "2": "Practical Applications"

        │ (Generated second, used by Level 0)
        ▼

LEVEL 0 REPORTS (8 reports)
├─ Report "0": "Deep Learning Optimization"
├─ Report "1": "Transformer Architecture"
├─ Report "2": "Training Techniques"
└─ ... (5 more detailed reports)

        │ All reports stored
        ▼

USAGE IN QUERIES
├─ Local Search: Find relevant Level 0 communities for specific questions
├─ Global Search: Synthesize information from multiple Level 1/2 communities
└─ Hierarchical Navigation: Drill down from general (Level 2) to specific (Level 0)
```

---

## Summary

The community creation process consists of two main phases:

1. **Graph Clustering** (Hierarchical Leiden Algorithm):
   - Automatically discovers community structure
   - Creates bottom-up hierarchy (Level 0 = finest, Level N = coarsest)
   - Number of communities and levels determined by graph structure
   - Each node belongs to one community per level

2. **Report Generation** (LLM-based Summarization):
   - Processes communities level-by-level (highest to lowest)
   - Extracts and ranks nodes/edges by importance
   - Manages token budgets through truncation or sub-community references
   - Generates structured JSON reports with titles, summaries, and findings
   - Higher-level reports can reference lower-level reports

The hierarchical structure enables both detailed (local) and broad (global) information retrieval during query time.
