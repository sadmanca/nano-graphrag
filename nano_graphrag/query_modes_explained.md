# Query Modes in Nano-GraphRAG: Local vs. Global Search

This document explains how nano-graphrag answers questions using two different search strategies: **Local Search** and **Global Search**.

---

## Quick Comparison

| Aspect | Local Search | Global Search |
|--------|-------------|---------------|
| **Best For** | Specific, detailed questions | Broad, summarization questions |
| **Example Query** | "What algorithm does paper X use?" | "What are the main themes across all papers?" |
| **Starting Point** | Find relevant entities via similarity | Use entire community hierarchy |
| **Scope** | Small neighborhood around relevant entities | Wide view across many communities |
| **Context Used** | Entities + their relationships + community reports + text chunks | High-level community reports only |
| **Speed** | Fast (focused search) | Slower (broad synthesis) |
| **Depth** | Deep (detailed local context) | Shallow (high-level patterns) |

---

## High-Level Overview

```
USER QUERY: "How does backpropagation work?"
    │
    ├─ LOCAL SEARCH MODE
    │  │
    │  ├─ Step 1: Find entities similar to query
    │  │          → ["BACKPROPAGATION", "GRADIENT DESCENT", "NEURAL NETWORK"]
    │  │
    │  ├─ Step 2: Get detailed local context
    │  │          → Communities these entities belong to
    │  │          → Relationships between these entities
    │  │          → Original text chunks mentioning them
    │  │
    │  └─ Step 3: Generate answer from local context
    │             → Focused, detailed response about backpropagation
    │
    └─ GLOBAL SEARCH MODE
       │
       ├─ Step 1: Get all high-level community reports
       │          → Top 512 most important communities across graph
       │
       ├─ Step 2: Ask each report: "What does this say about the query?"
       │          → Multiple perspectives from different parts of graph
       │
       └─ Step 3: Synthesize all perspectives into one answer
                  → Broad overview connecting multiple topics


ANALOGY:
─────────────────────────────────────────────────────────────────
Local Search  = Using index to find specific chapter, read details
Global Search = Reading executive summary of entire book
─────────────────────────────────────────────────────────────────
```

---

## Local Search - Detailed Flowchart

```
START: User asks "How does backpropagation work in neural networks?"
│
├─ STEP 1: Find Relevant Entities
│  │
│  ├─ Action: Query entity vector database
│  │         query_embedding = embed("How does backpropagation work...")
│  │         results = entities_vdb.query(query_embedding, top_k=20)
│  │
│  ├─ Process: Semantic similarity search
│  │          Compare query embedding to all entity embeddings
│  │          Return top K most similar entities
│  │
│  ├─ Result: List of relevant entities with similarity scores
│  │         [
│  │           {"entity_name": "BACKPROPAGATION", "similarity": 0.89},
│  │           {"entity_name": "GRADIENT DESCENT", "similarity": 0.76},
│  │           {"entity_name": "NEURAL NETWORK", "similarity": 0.74},
│  │           {"entity_name": "CHAIN RULE", "similarity": 0.71},
│  │           ...
│  │         ]
│  │
│  └─ Code: _op.py:1235
│
├─ STEP 2: Get Full Entity Details
│  │
│  ├─ Action: Retrieve complete entity information from graph
│  │         node_datas = await asyncio.gather(*[
│  │             knowledge_graph.get_node(entity["entity_name"])
│  │             for entity in results
│  │         ])
│  │
│  ├─ Enrich: Add node degree (connectivity measure)
│  │         node_degrees = await asyncio.gather(*[
│  │             knowledge_graph.node_degree(entity["entity_name"])
│  │         ])
│  │
│  ├─ Result: Full entity profiles
│  │         [
│  │           {
│  │             "entity_name": "BACKPROPAGATION",
│  │             "entity_type": "CONCEPT",
│  │             "description": "Algorithm for training neural networks...",
│  │             "source_id": "chunk-abc<SEP>chunk-def",
│  │             "rank": 15  # node degree
│  │           },
│  │           ...
│  │         ]
│  │
│  └─ Code: _op.py:1243-1255
│
├─ STEP 3: Find Related Communities
│  │
│  ├─ Action: Extract community memberships from entities
│  │         For each entity:
│  │           - Parse clusters: json.loads(entity["clusters"])
│  │           - Collect: [{"level": 0, "cluster": "5"}, ...]
│  │
│  ├─ Filter: Keep only communities at or below query level (default: 2)
│  │         if cluster["level"] <= query_param.level
│  │
│  ├─ Rank: Sort by:
│  │       1. Occurrence count (how many entities point to this community)
│  │       2. Community rating (from community report)
│  │
│  ├─ Retrieve: Get community reports
│  │          community_reports = await community_reports_db.get_by_ids(community_ids)
│  │
│  ├─ Truncate: Fit within token budget
│  │           truncate_list_by_token_size(..., max_token_size=3200)
│  │
│  ├─ Result: List of relevant community reports
│  │         [
│  │           {
│  │             "title": "Neural Network Training Methods",
│  │             "report_string": "# Neural Network Training...\n\nSummary...",
│  │             "rating": 8.5,
│  │             "occurrence": 0.15
│  │           },
│  │           ...
│  │         ]
│  │
│  └─ Code: _op.py:1000-1043
│
├─ STEP 4: Find Related Relationships (Edges)
│  │
│  ├─ Action: Get all edges connected to relevant entities
│  │         all_related_edges = await asyncio.gather(*[
│  │             knowledge_graph.get_node_edges(entity["entity_name"])
│  │         ])
│  │
│  ├─ Enrich: Add edge details
│  │         For each edge (source, target):
│  │           - Get edge data: weight, description
│  │           - Calculate rank: degree(source) + degree(target)
│  │
│  ├─ Rank: Sort by rank (edges between important nodes first)
│  │
│  ├─ Truncate: Fit within token budget
│  │           truncate_list_by_token_size(..., max_token_size=4800)
│  │
│  ├─ Result: List of relevant relationships
│  │         [
│  │           {
│  │             "src_tgt": ("BACKPROPAGATION", "GRADIENT DESCENT"),
│  │             "description": "uses for optimization",
│  │             "weight": 1.0,
│  │             "rank": 28
│  │           },
│  │           ...
│  │         ]
│  │
│  └─ Code: _op.py:1172-1224
│
├─ STEP 5: Find Related Text Chunks (Optional - Default: Disabled)
│  │
│  ├─ Check: if query_param.include_text_chunks == True
│  │
│  ├─ Action: Get original text chunks that mention these entities
│  │         For each entity:
│  │           - Parse source_id: "chunk-abc<SEP>chunk-def"
│  │           - Retrieve chunks: text_chunks_db.get_by_id(chunk_id)
│  │
│  ├─ Rank: Sort by:
│  │       1. Chunk order (earlier chunks first)
│  │       2. Relation count (chunks with more related entities)
│  │
│  ├─ Truncate: Fit within token budget
│  │           truncate_list_by_token_size(..., max_token_size=4000)
│  │
│  ├─ Result: Original text excerpts
│  │         [
│  │           {
│  │             "content": "Backpropagation algorithm computes...",
│  │             "tokens": 150
│  │           },
│  │           ...
│  │         ]
│  │
│  ├─ Note: By default, text chunks are EXCLUDED to reduce token usage
│  │        Set QueryParam(include_text_chunks=True) to enable
│  │
│  └─ Code: _op.py:1046-1169, base.py:17
│
├─ STEP 6: Format Context for LLM
│  │
│  ├─ Action: Combine all information into structured prompt
│  │         Context format:
│  │         -----Reports-----
│  │         ```csv
│  │         id, content
│  │         0,  # Neural Network Training Methods...
│  │         ```
│  │         -----Entities-----
│  │         ```csv
│  │         id, entity, type, description, rank
│  │         0,  BACKPROPAGATION, CONCEPT, Algorithm for..., 15
│  │         ```
│  │         -----Relationships-----
│  │         ```csv
│  │         id, source, target, description, weight, rank
│  │         0,  BACKPROPAGATION, GRADIENT DESCENT, uses..., 1.0, 28
│  │         ```
│  │         -----Sources-----
│  │         ```csv
│  │         id, content
│  │         0,  Original text chunk content...
│  │         ```
│  │
│  └─ Code: _op.py:1276-1348
│
├─ STEP 7: Generate Answer with LLM
│  │
│  ├─ System Prompt: PROMPTS["local_rag_response"]
│  │                "You are a helpful assistant responding to questions
│  │                 about data in the tables provided..."
│  │
│  ├─ Context: Insert the formatted context (from Step 6)
│  │
│  ├─ User Query: "How does backpropagation work in neural networks?"
│  │
│  ├─ LLM Call: response = await best_model_func(
│  │              query,
│  │              system_prompt=sys_prompt
│  │            )
│  │
│  ├─ Result: Detailed answer based on local context
│  │         "Backpropagation is an algorithm for training neural networks
│  │          that works by computing gradients using the chain rule.
│  │          According to the entities found, it uses gradient descent
│  │          for optimization and is fundamental to neural network training.
│  │          The community report on 'Neural Network Training Methods'
│  │          indicates that backpropagation is the primary method for..."
│  │
│  └─ Code: _op.py:1351-1386
│
└─ END: Return answer to user
```

---

## Global Search - Detailed Flowchart

```
START: User asks "What are the main themes in AI research?"
│
├─ STEP 1: Get Community Hierarchy
│  │
│  ├─ Action: Retrieve all communities from graph
│  │         community_schema = await knowledge_graph.community_schema()
│  │
│  ├─ Filter: Keep communities at or below query level (default: 2)
│  │         communities = {k: v for k, v in community_schema.items()
│  │                        if v["level"] <= query_param.level}
│  │
│  ├─ Result: Dictionary of all communities
│  │         {
│  │           "0": {level: 0, nodes: [...], occurrence: 0.05},
│  │           "1": {level: 1, nodes: [...], occurrence: 0.15},
│  │           ...
│  │         }
│  │
│  └─ Code: _op.py:1443-1446
│
├─ STEP 2: Rank and Select Top Communities
│  │
│  ├─ Action: Sort communities by importance
│  │         sorted_communities = sorted(
│  │             communities,
│  │             key=lambda x: x["occurrence"],  # How much of graph they cover
│  │             reverse=True
│  │         )
│  │
│  ├─ Limit: Keep top N communities (default: 512)
│  │        top_communities = sorted_communities[:512]
│  │
│  ├─ Why: Focus on most important communities
│  │       "occurrence" = fraction of total chunks this community covers
│  │       High occurrence = community represents significant portion of data
│  │
│  └─ Code: _op.py:1451-1458
│
├─ STEP 3: Retrieve Community Reports
│  │
│  ├─ Action: Get full reports for selected communities
│  │         community_datas = await community_reports.get_by_ids(
│  │             [community_id for community_id in top_communities]
│  │         )
│  │
│  ├─ Filter: Remove low-quality reports
│  │         Keep only if rating >= query_param.global_min_community_rating
│  │         (default: 0, so keeps all)
│  │
│  ├─ Re-rank: Sort by occurrence AND rating
│  │          sorted(community_datas,
│  │                 key=lambda x: (x["occurrence"], x["rating"]),
│  │                 reverse=True)
│  │
│  ├─ Result: Ordered list of high-quality community reports
│  │         [
│  │           {
│  │             "title": "Neural Network Methods",
│  │             "report_string": "# Neural Network Methods\n\nSummary...",
│  │             "rating": 8.5,
│  │             "occurrence": 0.15
│  │           },
│  │           {
│  │             "title": "Machine Learning Applications",
│  │             "report_string": "# ML Applications...",
│  │             "rating": 7.2,
│  │             "occurrence": 0.12
│  │           },
│  │           ...
│  │         ]
│  │
│  └─ Code: _op.py:1459-1473
│
├─ STEP 4: Group Communities for Processing
│  │
│  ├─ Problem: Too many communities to fit in single LLM context
│  │          512 communities * ~500 tokens each = 256K tokens
│  │          LLM limit: 16K-32K tokens
│  │
│  ├─ Solution: Split into groups that fit token budget
│  │
│  ├─ Action: Create groups of communities
│  │         while communities_remaining:
│  │           this_group = truncate_list_by_token_size(
│  │               communities_remaining,
│  │               key=lambda x: x["report_string"],
│  │               max_token_size=16384
│  │           )
│  │           community_groups.append(this_group)
│  │           communities_remaining = communities_remaining[len(this_group):]
│  │
│  ├─ Result: Multiple groups of communities
│  │         Group 1: [Community 0, 1, 2, ..., 30]  # ~16K tokens
│  │         Group 2: [Community 31, 32, ..., 65]   # ~16K tokens
│  │         Group 3: [Community 66, 67, ..., 95]   # ~16K tokens
│  │         ...
│  │
│  └─ Code: _op.py:1398-1405
│
├─ STEP 5: Map Phase - Extract Key Points from Each Group
│  │
│  │  FOR EACH group of communities:
│  │  │
│  │  ├─ Step 5a: Format Communities as CSV
│  │  │  │
│  │  │  ├─ Action: Create structured table
│  │  │  │         id, content, rating, importance
│  │  │  │         0,  "# Neural Network...\n\nSummary...", 8.5, 0.15
│  │  │  │         1,  "# ML Applications...", 7.2, 0.12
│  │  │  │         ...
│  │  │  │
│  │  │  └─ Code: _op.py:1408-1418
│  │  │
│  │  ├─ Step 5b: Generate LLM Prompt
│  │  │  │
│  │  │  ├─ System Prompt: PROMPTS["global_map_rag_points"]
│  │  │  │                "Generate a list of key points that responds
│  │  │  │                 to the user's question, summarizing all relevant
│  │  │  │                 information in the input data tables..."
│  │  │  │
│  │  │  ├─ Context: CSV table of community reports
│  │  │  │
│  │  │  ├─ Output Format: JSON
│  │  │  │                {
│  │  │  │                  "points": [
│  │  │  │                    {"description": "...", "score": 85},
│  │  │  │                    ...
│  │  │  │                  ]
│  │  │  │                }
│  │  │  │
│  │  │  └─ Code: _op.py:1419-1425
│  │  │
│  │  ├─ Step 5c: Call LLM
│  │  │  │
│  │  │  ├─ Request: response = await best_model_func(
│  │  │  │              query,
│  │  │  │              system_prompt=sys_prompt,
│  │  │  │              response_format={"type": "json_object"}
│  │  │  │            )
│  │  │  │
│  │  │  ├─ Parse: data = convert_response_to_json(response)
│  │  │  │
│  │  │  └─ Result: List of key points for this group
│  │  │            [
│  │  │              {
│  │  │                "description": "Neural networks are extensively used...",
│  │  │                "score": 85
│  │  │              },
│  │  │              {
│  │  │                "description": "Backpropagation is the primary training...",
│  │  │                "score": 78
│  │  │              },
│  │  │              ...
│  │  │            ]
│  │  │
│  │  └─ Code: _op.py:1421-1427
│  │
│  └─ NEXT group
│
├─ STEP 6: Reduce Phase - Combine All Key Points
│  │
│  ├─ Action: Aggregate points from all groups
│  │         all_points = []
│  │         for i, group_points in enumerate(map_results):
│  │             for point in group_points:
│  │                 all_points.append({
│  │                     "analyst": i,          # Which group produced this
│  │                     "answer": point["description"],
│  │                     "score": point["score"]
│  │                 })
│  │
│  ├─ Filter: Remove low-score points
│  │         points = [p for p in points if p["score"] > 0]
│  │
│  ├─ Rank: Sort by importance score (highest first)
│  │       sorted(points, key=lambda x: x["score"], reverse=True)
│  │
│  ├─ Truncate: Fit within token budget
│  │           truncate_list_by_token_size(..., max_token_size=16384)
│  │
│  ├─ Result: Ranked list of insights from entire graph
│  │         [
│  │           {
│  │             "analyst": 0,
│  │             "answer": "Neural networks are extensively researched...",
│  │             "score": 85
│  │           },
│  │           {
│  │             "analyst": 2,
│  │             "answer": "Transfer learning has become prominent...",
│  │             "score": 82
│  │           },
│  │           ...
│  │         ]
│  │
│  └─ Code: _op.py:1478-1500
│
├─ STEP 7: Format Points as Context
│  │
│  ├─ Action: Create structured text for final LLM call
│  │         Format:
│  │         ----Analyst 0----
│  │         Importance Score: 85
│  │         Neural networks are extensively researched...
│  │
│  │         ----Analyst 2----
│  │         Importance Score: 82
│  │         Transfer learning has become prominent...
│  │
│  │         ...
│  │
│  └─ Code: _op.py:1501-1509
│
├─ STEP 8: Generate Final Answer
│  │
│  ├─ System Prompt: PROMPTS["global_reduce_rag_response"]
│  │                "You are a helpful assistant synthesizing perspectives
│  │                 from multiple analysts. Generate a response that
│  │                 summarizes all reports..."
│  │
│  ├─ Context: Formatted key points (from Step 7)
│  │
│  ├─ User Query: "What are the main themes in AI research?"
│  │
│  ├─ LLM Call: response = await best_model_func(
│  │              query,
│  │              system_prompt=sys_prompt
│  │            )
│  │
│  ├─ Result: Comprehensive answer synthesizing all perspectives
│  │         "The main themes in AI research include:
│  │
│  │          1. Neural Network Architectures: Research extensively covers
│  │             deep learning methods, particularly transformer architectures
│  │             and convolutional networks...
│  │
│  │          2. Training Methodologies: Backpropagation remains central,
│  │             with growing interest in optimization techniques and
│  │             transfer learning approaches...
│  │
│  │          3. Applications: Practical deployments span computer vision,
│  │             natural language processing, and reinforcement learning...
│  │
│  │          [Synthesizes information from multiple community reports
│  │           across the entire knowledge graph]"
│  │
│  └─ Code: _op.py:1512-1519
│
└─ END: Return comprehensive answer to user
```

---

## Key Differences Explained

### 1. **Starting Point**

**Local Search:**
```
Query: "How does backpropagation work?"
    ↓
Find similar entities in vector database
    ↓
["BACKPROPAGATION", "GRADIENT DESCENT", "NEURAL NETWORK"]
    ↓
Focus on these entities and their immediate neighborhood
```

**Global Search:**
```
Query: "What are the main themes?"
    ↓
Get ALL high-level communities
    ↓
[512 most important communities across entire graph]
    ↓
Process all communities to find patterns
```

### 2. **Context Gathering**

**Local Search - Deep and Narrow:**
```
For each relevant entity:
├─ Entity details (type, description)
├─ Community reports (that contain these entities)
├─ Direct relationships (edges to/from entities)
├─ Text chunks (original sources mentioning entities)
└─ One-hop neighbors (entities connected to our entities)

Example Context Size: ~10K tokens
- 20 entities
- 2-3 community reports
- 50 relationships
- 10 text chunks
```

**Global Search - Broad and Shallow:**
```
For entire graph:
├─ Top 512 community reports (ranked by importance)
├─ NO individual entities
├─ NO relationships
└─ NO text chunks

Example Context Size: ~256K tokens (split across multiple LLM calls)
- 512 community reports
- Each report ~500 tokens
```

### 3. **LLM Processing Strategy**

**Local Search - Single Pass:**
```
┌─────────────────────────────────────┐
│    Gather Local Context (~10K)     │
└─────────────┬───────────────────────┘
              ↓
        ┌──────────┐
        │   LLM    │  Single call
        └────┬─────┘
             ↓
        Final Answer
```

**Global Search - Map-Reduce:**
```
┌───────────────────────────────────────────────┐
│     512 Community Reports (~256K tokens)     │
└───────┬──────┬──────┬──────┬──────┬──────────┘
        │      │      │      │      │
   ┌────▼──┐ ┌─▼───┐ ┌─▼───┐ ┌─▼───┐ ...
   │ LLM 1 │ │LLM 2│ │LLM 3│ │LLM 4│  MAP phase
   │Extract│ │Extr.│ │Extr.│ │Extr.│  (parallel)
   │points │ │pts. │ │pts. │ │pts. │
   └───┬───┘ └──┬──┘ └──┬──┘ └──┬──┘
       │        │       │       │
       └────┬───┴───┬───┴───┬───┘
            │       │       │
      ┌─────▼───────▼───────▼──────┐
      │   Combine & Rank Points    │
      └──────────────┬──────────────┘
                     ↓
               ┌──────────┐
               │   LLM    │  REDUCE phase
               │Synthesize│
               └────┬─────┘
                    ↓
              Final Answer
```

### 4. **Answer Characteristics**

**Local Search Answer:**
```
✓ Specific and detailed
✓ Cites exact entities and relationships
✓ Includes original source quotes (if text chunks enabled)
✓ Focused on query entities
✗ May miss broader context
✗ Limited to local neighborhood

Example:
"Backpropagation works by computing gradients through the chain rule.
According to the entity BACKPROPAGATION found in the Neural Network
Training Methods community, it uses GRADIENT DESCENT for optimization.
The relationship 'BACKPROPAGATION enables NEURAL NETWORK training'
indicates its fundamental role..."
```

**Global Search Answer:**
```
✓ Comprehensive and broad
✓ Synthesizes multiple perspectives
✓ Identifies patterns across entire graph
✓ Good for themes and summaries
✗ Less specific detail
✗ No citation of exact entities

Example:
"The main themes across all research include: 1) Neural network
architectures with emphasis on transformers and CNNs, 2) Training
methodologies centered on backpropagation and optimization, 3) Transfer
learning and few-shot approaches, 4) Applications in vision and NLP.
These themes emerge from analysis of all major research communities..."
```

---

## When to Use Each Mode

### Use Local Search When:

✓ **Question is specific**: "What is the architecture of ResNet?"
✓ **Need detailed information**: "Explain how attention mechanisms work"
✓ **Asking about particular entities**: "What papers discuss BERT?"
✓ **Want citations/sources**: Need to trace back to original text
✓ **Time-sensitive**: Need faster response

**Example Questions:**
- "How does the transformer attention mechanism work?"
- "What are the hyperparameters for training GPT-3?"
- "Which papers discuss few-shot learning?"
- "What is the relationship between CNNs and image classification?"

### Use Global Search When:

✓ **Question is broad**: "What are the trends in AI?"
✓ **Need high-level overview**: "Summarize main research areas"
✓ **Comparing multiple topics**: "What are the differences between approaches?"
✓ **Looking for themes**: "What are common challenges across papers?"
✓ **Want comprehensive synthesis**: Need big-picture view

**Example Questions:**
- "What are the main themes in machine learning research?"
- "Summarize the key innovations across all papers"
- "What are the emerging trends in AI?"
- "Compare the different approaches to neural network training"

---

## Configuration Parameters

### Local Search Parameters

```python
from nano_graphrag import QueryParam

query_param = QueryParam(
    mode="local",                               # Use local search

    # Entity retrieval
    top_k=20,                                   # Number of entities to retrieve

    # Token budgets for different context types
    local_max_token_for_text_unit=4000,         # Budget for text chunks
    local_max_token_for_local_context=4800,     # Budget for relationships
    local_max_token_for_community_report=3200,  # Budget for community reports

    # Community filtering
    level=2,                                    # Max community level to use
    local_community_single_one=False,           # Use multiple communities (not just 1)

    # Text chunks control
    include_text_chunks=False,                  # Exclude text chunks by default

    # Output format
    response_type="Multiple Paragraphs",        # Desired answer format
    only_need_context=False,                    # Return context or full answer?
)

answer = graph_rag.query("Your question", param=query_param)
```

### Global Search Parameters

```python
from nano_graphrag import QueryParam

query_param = QueryParam(
    mode="global",                              # Use global search

    # Community selection
    global_max_consider_community=512,          # Max communities to consider
    global_min_community_rating=0,              # Min rating threshold (0-10)
    level=2,                                    # Max community level to use

    # Token budgets
    global_max_token_for_community_report=16384, # Budget per group in map phase

    # LLM configuration for map phase
    global_special_community_map_llm_kwargs={
        "response_format": {"type": "json_object"}
    },

    # Output format
    response_type="Multiple Paragraphs",        # Desired answer format
    only_need_context=False,                    # Return context or full answer?
)

answer = graph_rag.query("Your question", param=query_param)
```

---

## Example Query Comparison

### Query: "How does backpropagation work?"

**Local Search Process:**
```
1. Find entities: ["BACKPROPAGATION", "GRADIENT DESCENT", "NEURAL NETWORK"]
2. Get community: "Neural Network Training Methods"
3. Get relationships:
   - BACKPROPAGATION → GRADIENT DESCENT ("uses")
   - BACKPROPAGATION → CHAIN RULE ("implements")
   - BACKPROPAGATION → NEURAL NETWORK ("trains")
4. Generate answer from specific context

Result: Detailed explanation with specific entity relationships
Time: ~2-3 seconds
Tokens: ~10K context
```

**Global Search Process:**
```
1. Get 512 communities across entire graph
2. Map phase: Extract points from each group
   - Group 1 (30 communities): "Backprop is fundamental training method"
   - Group 2 (35 communities): "Used in all deep learning architectures"
   - Group 3 (28 communities): "Relies on gradient descent optimization"
   ...
3. Reduce phase: Synthesize all points into coherent answer

Result: Broad overview of backprop's role across all research
Time: ~10-15 seconds
Tokens: ~256K total (split across multiple calls)
```

---

## Visual Comparison: Context Scope

```
KNOWLEDGE GRAPH (All Entities)
┌──────────────────────────────────────────────────────────────┐
│  ●────●────●        ●────●           ●────●────●             │
│  │    │    │        │    │           │    │    │             │
│  ●    ●    ●────●───●    ●───●       ●────●    ●             │
│  │         │    │   │        │       │         │             │
│  ●    ●────●    ●───●────●   ●       ●    ●────●             │
│       │         │        │   │            │                   │
│  ●────●    ●────●────●   ●───●       ●────●    ●────●        │
│  │    │    │    │    │       │       │    │    │    │        │
│  ●    ●────●    ●────●       ●───●   ●    ●────●    ●        │
│                 │                │                   │        │
│  ●────●    ●────●────●       ●───●   ●────●    ●────●        │
│  │    │    │         │       │       │    │    │             │
│  ●    ●────●    ●────●───●   ●       ●────●    ●             │
│                           │                                   │
└──────────────────────────────────────────────────────────────┘

LOCAL SEARCH SCOPE:
┌──────────────────────────────────────────────────────────────┐
│  ●────●────●        ●────●           ●────●────●             │
│  │ \  │  / │        │    │           │    │    │             │
│  ● ─[●]─ ●────●───●    ●───●       ●────●    ●             │
│  │ /     \    │   │        │       │         │             │
│  ●    ●────●    ●───●────●   ●       ●    ●────●             │
│       │         │        │   │            │                   │
│  ●────●    ●────●────●   ●───●       ●────●    ●────●        │
│  │    │    │    │    │       │       │    │    │    │        │
│  ●    ●────●    ●────●       ●───●   ●    ●────●    ●        │
│                 │                │                   │        │
│  ●────●    ●────●────●       ●───●   ●────●    ●────●        │
│  │    │    │         │       │       │    │    │             │
│  ●    ●────●    ●────●───●   ●       ●────●    ●             │
└──────────────────────────────────────────────────────────────┘
    ↑
    Query entity + immediate neighborhood
    ~20 entities, ~50 edges, 2-3 communities

GLOBAL SEARCH SCOPE:
┌══════════════════════════════════════════════════════════════┐
║  ●────●────●        ●────●           ●────●────●             ║
║  │    │    │        │    │           │    │    │             ║
║  ●    ●    ●────●───●    ●───●       ●────●    ●             ║
║  │         │    │   │        │       │         │             ║
║  ●    ●────●    ●───●────●   ●       ●    ●────●             ║
║       │         │        │   │            │                   ║
║  ●────●    ●────●────●   ●───●       ●────●    ●────●        ║
║  │    │    │    │    │       │       │    │    │    │        ║
║  ●    ●────●    ●────●       ●───●   ●    ●────●    ●        ║
║                 │                │                   │        ║
║  ●────●    ●────●────●       ●───●   ●────●    ●────●        ║
║  │    │    │         │       │       │    │    │             ║
║  ●    ●────●    ●────●───●   ●       ●────●    ●             ║
║                           │                                   ║
└══════════════════════════════════════════════════════════════┘
    ↑
    Entire graph through community reports
    512 communities covering all entities
```

---

## Summary

**Local Search** = Zoomed-in, detailed view
- Find specific entities → Explore their neighborhood → Answer with details
- Best for: "Tell me about X"

**Global Search** = Zoomed-out, bird's-eye view
- Use all communities → Extract themes → Synthesize comprehensive answer
- Best for: "What are the patterns across everything?"

Both modes use the same underlying knowledge graph but access it differently:
- **Local**: Bottom-up (entities → communities)
- **Global**: Top-down (communities → insights)

Choose based on your question's specificity and scope!
