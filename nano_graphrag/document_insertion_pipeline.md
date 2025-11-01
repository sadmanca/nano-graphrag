# Nano-GraphRAG Document Insertion Pipeline

This document provides a comprehensive overview of the document insertion pipeline in nano-graphrag, from raw text input to a fully-indexed knowledge graph with community reports.

---

## Level 1: High-Level Flowchart

The document insertion pipeline consists of the following major steps:

```
1. Document Deduplication & Preparation
   ↓
2. Text Chunking & Tokenization
   ↓
3. Entity & Relationship Extraction
   ↓
4. Graph Clustering (Leiden Algorithm)
   ↓
5. Community Report Generation
   ↓
6. Storage & Indexing
```

---

## Level 2: Detailed Step Explanations

### 1. Document Deduplication & Preparation

The pipeline begins by accepting one or more text documents (as strings). Each document is assigned a unique MD5 hash ID based on its content, using the pattern `doc-{hash}`. This hash-based approach ensures that duplicate documents are automatically detected and skipped, preventing redundant processing. The system checks these hash IDs against the `full_docs` storage to filter out documents that have already been processed. Only new, unique documents proceed to the next stage.

If all documents already exist in storage, the pipeline logs a warning and exits early. This incremental insertion capability allows users to add documents over time without re-processing the entire corpus. The document content is stored with its hash ID in a key-value storage system for later reference.

### 2. Text Chunking & Tokenization

New documents are split into smaller, overlapping text chunks to facilitate more granular entity extraction and retrieval. The chunking process uses the `tiktoken` library to tokenize text based on a specific model's tokenizer (default: GPT-4o). Two chunking strategies are available: token-size-based chunking (default) and separator-based chunking.

Each chunk is created with configurable parameters: `chunk_token_size` (default: 1200 tokens) and `chunk_overlap_token_size` (default: 100 tokens). The overlap ensures that entities or concepts spanning chunk boundaries are not lost. Each chunk is assigned metadata including token count, content text, the parent document ID (`full_doc_id`), and its order within the document (`chunk_order_index`). Like documents, chunks receive MD5 hash IDs (pattern: `chunk-{hash}`) for deduplication. The chunks are stored in the `text_chunks` storage and, if naive RAG is enabled, they are also indexed in the `chunks_vdb` vector database for similarity search.

### 3. Entity & Relationship Extraction

This is the core knowledge extraction phase where the system identifies entities (nodes) and their relationships (edges) from text chunks. Nano-graphrag supports two extraction methods:

#### Standard Extraction Method (`extract_entities`)

The standard method processes each chunk using an LLM with a specialized entity extraction prompt. The prompt instructs the LLM to identify entities with names, types (e.g., person, organization, geo, event), and descriptions, as well as relationships between entities with descriptions and strength scores. The system uses an iterative "gleaning" process (controlled by `entity_extract_max_gleaning`) where it asks the LLM multiple times if any entities were missed, improving recall.

Extracted entities and relationships are accumulated across all chunks. When the same entity appears in multiple chunks, the system merges their descriptions by concatenating them. If the merged description exceeds a token limit (`entity_summary_to_max_tokens`), it's summarized using the cheap LLM model. Entities are stored as nodes in the graph with attributes: `entity_name`, `entity_type`, `description`, and `source_id` (chunk IDs where the entity appears). Relationships are stored as edges with `weight`, `description`, and `source_id`. All entities are also indexed in the `entities_vdb` vector database for fast similarity-based retrieval during queries.

#### GenKG Extraction Method (`extract_entities_genkg`)

The GenKG method provides an alternative, more document-centric approach. Instead of processing chunks individually, it first groups chunks by their parent document ID, reconstructing the full document text. Each document is then summarized using an LLM with a prompt focused on key scientific concepts, methodologies, and findings (limited to 4000 characters).

**Node Extraction**: GenKG extracts nodes representing high-level scientific concepts. Each node is simply a text string (e.g., "Machine Learning", "Neural Networks", "Gradient Descent"). The LLM receives the document summary and is asked to identify the top N (default: 25) most important concepts, methods, and results. The node data structure contains:
- `node`: A string representing the concept (from LLM)
- `source`: The document ID where this concept was found (added by the system)

After normalization for Windows compatibility, nodes are stored with additional metadata:
- `entity_name`: Normalized uppercase version (e.g., "MACHINE LEARNING")
- `entity_type`: Always set to "CONCEPT" (GenKG doesn't distinguish entity types)
- `description`: The original concept text before normalization
- `source_id`: The document ID

**Edge Extraction**: GenKG creates edges by asking the LLM to identify meaningful relationships between the extracted nodes. The edge data structure contains:
- `node1`: Text of the first node
- `node2`: Text of the second node
- `weight`: A float indicating relationship strength (e.g., 1.0 for strong, 0.5 for medium, 0.15 for weak)
- `relation`: A short label describing the relationship type (e.g., "related_to", "enables", "depends_on")

**What Information is Provided to the LLM for Edge Creation?** The edge creation prompt (`genkg.py:244-258`) receives:
1. **All node texts**: A formatted list of all concepts extracted from all documents (e.g., "- Machine Learning\n- Neural Networks\n- Gradient Descent...")
2. **All paper summaries**: The complete text of all document summaries, formatted as "--- Paper: {source} ---\n{summary}" for each document
3. **Instructions**: The LLM is instructed to create "meaningful and non-trivial relationships" and to ensure "no orphan nodes" (i.e., the graph should be connected)

This cross-document context allows the LLM to identify relationships both within a single paper and across multiple papers. For example, if Paper A discusses "Neural Networks" and Paper B discusses "Backpropagation," the LLM can create an edge between them with relation "enables" even though they come from different sources.

**Graph Connectivity Enhancement**: After initial edge extraction, GenKG applies a connectivity algorithm that uses sentence transformers (all-MiniLM-L6-v2) to find semantic similarities between disconnected components. For each isolated subgraph, it computes embeddings for all nodes, finds the most semantically similar node pair between the isolated component and the main graph, and adds a connectivity edge with:
- `weight`: Lower than normal edges (max 0.1 to 0.5 based on similarity score)
- `relation`: "semantic_similarity"

Node names are normalized (uppercased, special characters replaced) for Windows compatibility and consistency with the standard extraction method.

### 4. Graph Clustering (Leiden Algorithm)

After entity extraction, the knowledge graph is clustered into hierarchical communities using the Leiden algorithm. This phase creates a multi-level structure where nodes are grouped into communities at different granularities. The system uses the `hierarchical_leiden` implementation from the graspologic library.

**Connected Component Processing**: Unlike earlier implementations, the current system processes **all connected components** of the graph, not just the largest one. This ensures that entities from all documents are included in the clustering, even if they form separate subgraphs. For each connected component, the graph is stabilized (nodes and edges sorted consistently), node names are normalized (uppercased and HTML-unescaped), and hierarchical clustering is applied.

**How Many Communities Are Created?** The number of communities is **automatically determined by the Leiden algorithm** based on the graph's structure and connectivity patterns. The algorithm is guided by one key parameter:
- `max_graph_cluster_size` (default: 10): Maximum number of nodes allowed in a single community at the finest granularity level

The algorithm works by optimizing modularity - a measure of how well-separated communities are compared to random chance. It iteratively moves nodes between communities to maximize this metric, naturally discovering the optimal community structure. The `max_cluster_size` parameter ensures that communities don't become too large to be meaningful, forcing the algorithm to split larger groups into subcommunities.

**How Are Hierarchical Levels Created?** The `hierarchical_leiden` algorithm creates a **bottom-up hierarchy** automatically:

1. **Level 0 (Finest Granularity)**: The algorithm starts by running Leiden clustering on the full graph, creating initial communities that respect the `max_cluster_size` constraint. These are the smallest, most specific communities. For example, in a scientific paper graph, Level 0 might create communities like "Deep Learning Optimization Techniques" or "Transformer Architecture Components."

2. **Higher Levels (Increasing Granularity)**: The algorithm then creates a "super-graph" where each Level 0 community becomes a single node, and edges between communities are weighted by the number of connections between them. It applies Leiden clustering again to this super-graph, creating Level 1 communities (which are groups of Level 0 communities). This process repeats recursively:
   - Level 1: Groups of Level 0 communities (e.g., "Deep Learning Methods")
   - Level 2: Groups of Level 1 communities (e.g., "Machine Learning")
   - Level N: Continues until the entire graph forms a single community or no further meaningful groupings exist

3. **Automatic Termination**: The algorithm stops creating new levels when either:
   - The super-graph becomes too small to meaningfully cluster further
   - All nodes would end up in a single community at the next level
   - No improvement in modularity can be achieved

**Level Numbering**: In the implementation, level numbers increase with hierarchy size. Level 0 contains the most fine-grained communities (smallest, most specific), while higher levels contain coarser communities (larger, more general). Each node is assigned to **one community at each level**, creating a nested hierarchy where Level N+1 communities are always composed of groups of Level N communities.

**Cluster ID Management**: To ensure unique cluster IDs across different connected components, the system uses an offset mechanism. After clustering each component, it records the maximum cluster ID used and adds it as an offset before clustering the next component. This prevents ID collisions between components.

**Storage**: Each node receives cluster assignments stored as JSON in its `clusters` attribute:
```json
[
  {"level": 0, "cluster": "5"},
  {"level": 1, "cluster": "2"},
  {"level": 2, "cluster": "0"}
]
```

The hierarchical structure is later used to support sub-community relationships in community reports, where higher-level reports can reference and summarize lower-level community findings.

### 5. Community Report Generation

Once clustering is complete, the system generates natural language reports for each community. This process operates level-by-level, starting from the highest (most general) level and moving to lower levels. This order is important because lower-level reports can reference their parent communities' reports.

For each community, the system gathers its nodes, edges, and (if available) sub-community reports. This information is formatted as CSV-style tables and sent to the best LLM model with the `community_report` prompt. The prompt instructs the LLM to generate a structured JSON report containing: a title (short but specific), an executive summary, an impact severity rating (0-10), a rating explanation, and 5-10 detailed findings with summaries and explanations.

The context sent to the LLM is carefully managed to fit within token limits. Nodes are sorted by degree (connectivity) and truncated if necessary. Edges are sorted by rank (combined degree of source and target nodes). If a community's context exceeds the token limit and it has sub-communities, the system includes the sub-community reports instead of the raw nodes and edges. Generated reports are stored in the `community_reports` storage with both the string representation and the JSON structure. These reports enable both local search (finding relevant communities for specific queries) and global search (synthesizing information across multiple communities).

### 6. Storage & Indexing

The final phase commits all processed data to persistent storage. Nano-graphrag uses three types of storage: key-value storage for documents, chunks, and reports; vector databases for embeddings; and graph storage for the knowledge graph structure.

The system invokes `index_done_callback()` on all storage instances, which triggers backend-specific operations like writing graph data to GraphML files (for NetworkX) or committing transactions (for databases). The `full_docs` storage receives the original documents, `text_chunks` stores the chunked text, `community_reports` holds the generated reports, `entities_vdb` indexes entity embeddings for similarity search, and `chunk_entity_relation_graph` persists the graph structure with all nodes, edges, and cluster assignments.

If GenKG extraction is enabled with visualization, additional outputs are generated: an HTML interactive graph visualization and a `.dashkg.json` file containing structured graph data. These files use the normalized entity names and relationships created during extraction. All storage operations are asynchronous and executed concurrently for performance. The pipeline completes when all data has been successfully persisted, making it available for subsequent queries.

---

## Level 3: Code-Level Implementation Details

### 1. Document Deduplication & Preparation

**Entry Point**: `graphrag.py:280` - `async def ainsert(self, string_or_strings)`

The insertion process begins by converting a single string or list of strings into a dictionary where keys are MD5 hashes and values contain the document content:

```python
new_docs = {
    compute_mdhash_id(c.strip(), prefix="doc-"): {"content": c.strip()}
    for c in string_or_strings
}
```

**Hash Function**: `_utils.py:compute_mdhash_id()` - Generates MD5 hash with a prefix (e.g., "doc-abc123...")

**Deduplication Check**: `graphrag.py:290` - `await self.full_docs.filter_keys(list(new_docs.keys()))`
- `filter_keys()` queries the `BaseKVStorage` implementation to find keys that don't already exist
- Returns only the new document keys to be processed
- `full_docs` is typically a `JsonKVStorage` instance from `_storage/kv_json.py`

**Early Exit**: If `len(new_docs) == 0`, the pipeline logs a warning and returns without further processing.

### 2. Text Chunking & Tokenization

**Chunking Function**: `_op.py:101` - `def get_chunks(new_docs, chunk_func=chunking_by_token_size, **chunk_func_params)`

**Process Flow**:
1. Extract document keys and content from `new_docs` dictionary
2. Initialize tiktoken encoder: `ENCODER = tiktoken.encoding_for_model("gpt-4o")` (`_op.py:108`)
3. Encode all documents in batch: `tokens = ENCODER.encode_batch(docs, num_threads=16)` (`_op.py:109`)
4. Call chunking function with tokens, keys, and encoder

**Token-Based Chunking**: `_op.py:32` - `def chunking_by_token_size()`
- Iterates through tokens with a sliding window: `for start in range(0, len(tokens), max_token_size - overlap_token_size)`
- Creates overlapping chunks: `chunk_token.append(tokens[start : start + max_token_size])`
- Decodes chunks back to text: `chunk_token = tiktoken_model.decode_batch(chunk_token)`

**Separator-Based Chunking**: `_op.py:65` - `def chunking_by_seperators()`
- Uses `SeparatorSplitter` from `_splitter.py`
- Splits on configured separators: `PROMPTS["default_text_separator"]` (includes newlines, periods, spaces, etc.)
- Maintains token size constraints while respecting natural text boundaries

**Chunk Schema**: `base.py:34` - `TextChunkSchema`
```python
TextChunkSchema = TypedDict(
    "TextChunkSchema",
    {"tokens": int, "content": str, "full_doc_id": str, "chunk_order_index": int},
)
```

**Chunk Deduplication**: `_op.py:117` - Each chunk gets an MD5 hash: `chunk_id = compute_mdhash_id(chunk["content"], prefix="chunk-")`

**Storage Operations**:
- `graphrag.py:306` - Check for existing chunks: `await self.text_chunks.filter_keys(list(inserting_chunks.keys()))`
- `graphrag.py:318` - Index chunks in vector DB if naive RAG enabled: `await self.chunks_vdb.upsert(inserting_chunks)`

**Key Configuration Parameters** (from `graphrag.py:74-76`):
- `chunk_token_size: int = 1200` - Maximum tokens per chunk
- `chunk_overlap_token_size: int = 100` - Overlap between consecutive chunks
- `tiktoken_model_name: str = "gpt-4o"` - Tokenizer model

### 3. Entity & Relationship Extraction

The system supports two extraction methods selected by `entity_extraction_func` parameter.

#### Standard Extraction Method

**Function**: `_op.py:304` - `async def extract_entities(chunks, knwoledge_graph_inst, entity_vdb, global_config)`

**Prompt Configuration**: `prompt.py:195` - `PROMPTS["entity_extraction"]`
- Instructs LLM to extract entities with format: `("entity"<|>name<|>type<|>description)`
- Instructs LLM to extract relationships with format: `("relationship"<|>source<|>target<|>description<|>strength)`
- Delimiter configuration from `prompt.py:324-327`:
  - `DEFAULT_TUPLE_DELIMITER = "<|>"`
  - `DEFAULT_RECORD_DELIMITER = "##"`
  - `DEFAULT_COMPLETION_DELIMITER = "<|COMPLETE|>"`
  - `DEFAULT_ENTITY_TYPES = ["organization", "person", "geo", "event"]`

**Processing Loop**: `_op.py:329` - `async def _process_single_content(chunk_key_dp)`
1. Format prompt with chunk content: `entity_extract_prompt.format(**context_base, input_text=content)` (`_op.py:334`)
2. Send to LLM: `final_result = await use_llm_func(hint_prompt)` (`_op.py:335`)
3. Iterative gleaning loop (`_op.py:338-351`):
   - Send continue prompt: `PROMPTS["entiti_continue_extraction"]`
   - Ask if more entities exist: `PROMPTS["entiti_if_loop_extraction"]`
   - Break if LLM responds with anything other than "yes"
4. Parse response using delimiters (`_op.py:353-356`)
5. Extract entities and relationships using regex: `record = re.search(r"\((.*)\)", record)` (`_op.py:361`)

**Entity Parsing**: `_op.py:162` - `async def _handle_single_entity_extraction(record_attributes, chunk_key)`
- Validates format: `len(record_attributes) < 4 or record_attributes[0] != '"entity"'`
- Cleans and uppercases entity name: `entity_name = clean_str(record_attributes[1].upper())` (`_op.py:169`)
- Returns dict: `{"entity_name": name, "entity_type": type, "description": desc, "source_id": chunk_key}`

**Relationship Parsing**: `_op.py:183` - `async def _handle_single_relationship_extraction(record_attributes, chunk_key)`
- Validates format: `len(record_attributes) < 5 or record_attributes[0] != '"relationship"'`
- Extracts weight: `float(record_attributes[-1]) if is_float_regex(record_attributes[-1]) else 1.0` (`_op.py:195`)
- Returns dict: `{"src_id": source, "tgt_id": target, "weight": weight, "description": desc, "source_id": chunk_key}`

**Entity Merging**: `_op.py:206` - `async def _merge_nodes_then_upsert(entity_name, nodes_data, knwoledge_graph_inst, global_config)`
1. Get existing node: `already_node = await knwoledge_graph_inst.get_node(entity_name)` (`_op.py:216`)
2. Merge entity types by frequency: `sorted(Counter([dp["entity_type"] for dp in nodes_data] + already_entitiy_types).items(), key=lambda x: x[1], reverse=True)[0][0]` (`_op.py:224-230`)
3. Concatenate descriptions: `description = GRAPH_FIELD_SEP.join(sorted(set([dp["description"] for dp in nodes_data] + already_description)))` (`_op.py:231-233`)
4. Summarize if too long: `await _handle_entity_relation_summary(entity_name, description, global_config)` (`_op.py:237`)

**Summary Function**: `_op.py:135` - `async def _handle_entity_relation_summary(entity_or_relation_name, description, global_config)`
- Check token count: `tokens = encode_string_by_tiktoken(description, model_name=tiktoken_model_name)` (`_op.py:145`)
- If exceeds `summary_max_tokens` (default: 500), truncate and summarize: `await use_llm_func(use_prompt, max_tokens=summary_max_tokens)` (`_op.py:158`)

**Edge Merging**: `_op.py:253` - `async def _merge_edges_then_upsert(src_id, tgt_id, edges_data, knwoledge_graph_inst, global_config)`
- Get existing edge: `await knwoledge_graph_inst.get_edge(src_id, tgt_id)` (`_op.py:265`)
- Sum weights: `weight = sum([dp["weight"] for dp in edges_data] + already_weights)` (`_op.py:275`)
- Concatenate descriptions with `GRAPH_FIELD_SEP` separator (`_op.py:276-278`)
- Auto-create nodes if they don't exist (`_op.py:282-291`)

**Vector Database Indexing**: `_op.py:423` - Store entities in vector DB for similarity search
```python
data_for_vdb = {
    compute_mdhash_id(dp["entity_name"], prefix="ent-"): {
        "content": dp["entity_name"] + dp["description"],
        "entity_name": dp["entity_name"],
    }
    for dp in all_entities_data
}
await entity_vdb.upsert(data_for_vdb)
```

**Configuration Parameters**:
- `entity_extract_max_gleaning: int = 1` - Number of follow-up extraction rounds (`graphrag.py:79`)
- `entity_summary_to_max_tokens: int = 500` - Max tokens before summarizing descriptions (`graphrag.py:80`)
- `best_model_func: callable` - LLM for extraction (default: GPT-4o)
- `cheap_model_func: callable` - LLM for summarization (default: GPT-4o-mini)

#### GenKG Extraction Method

**Function**: `_op.py:435` - `async def extract_entities_genkg(chunks, knwoledge_graph_inst, entity_vdb, global_config)`

**Configuration Parameters**:
- `use_genkg_extraction: bool = False` - Enable GenKG extraction (`graphrag.py:133`)
- `genkg_node_limit: int = 25` - Max nodes per document (`graphrag.py:134`)
- `genkg_llm_provider: str = "gemini"` - LLM provider for GenKG (`graphrag.py:135`)
- `genkg_model_name: str = "gemini-2.5-flash"` - Model name (`graphrag.py:136`)
- `genkg_create_visualization: bool = False` - Generate HTML/JSON outputs (`graphrag.py:137`)

**GenKG Integration**: `_op.py:446-473` - Import GenKG module
- Tries multiple paths to locate `genkg.py` in the project structure
- Imports `GenerateKG` class from `genkg.py:118`

**GenKG Class**: `genkg.py:118` - `class GenerateKG`
- Initializes LLM provider abstraction: `genkg.py:119` - `def __init__(self, llm_provider="gemini", model_name=None)`
- Uses `LLMProvider` class for model abstraction (`genkg.py:37`)

**Document Grouping**: `_op.py:486-491`
```python
for chunk_key, chunk_data in ordered_chunks:
    doc_id = chunk_data.get("full_doc_id", chunk_key)
    if doc_id not in papers_dict:
        papers_dict[doc_id] = ""
    papers_dict[doc_id] += chunk_data["content"] + "\n\n"
```

**Paper Summarization**: `_op.py:500` - `summary = genkg.summarize_paper(doc_content, doc_id)`
- Implementation: `genkg.py:132` - `def summarize_paper(self, paper_text, paper_source, max_chars=4000)`
- Prompt (`genkg.py:155-168`): Focuses on:
  1. Main research objectives and questions
  2. Key methodologies and approaches used
  3. Important findings and results
  4. Core scientific concepts and terminology
  5. Novel contributions to the field
- Enforces character limit: `summary[:max_chars]` (`genkg.py:174`)

**Node Extraction**: `_op.py:504` - `nodes_with_source = genkg.gemini_create_nodes(summary, node_limit, doc_id)`
- Implementation: `genkg.py:183` - `def gemini_create_nodes(self, paper_summary, node_limit, paper_source)`
- Uses structured generation with `KGNode` schema: `genkg.py:109`
  ```python
  class KGNode(BaseModel):
      node: str  # Just a string, e.g., "Neural Networks"
  ```
- Prompt (`genkg.py:206-208`):
  ```
  From the following research paper summary, extract the top {node_limit} most important
  high-level scientific concepts, methods, and results. Focus on concepts that are central
  to the paper's contribution.

  Paper Summary:
  """
  {paper_summary}
  """
  ```
- LLM returns JSON array: `[{"node": "Concept 1"}, {"node": "Concept 2"}, ...]`
- System parses and creates tuples: `(concept_text, paper_source)`
- Returns list of `(concept, paper_source)` tuples (`genkg.py:222`)

**Edge Extraction**: `_op.py:507` - `edges = genkg.create_edges_by_gemini(nodes_with_source, {doc_id: summary})`
- Implementation: `genkg.py:228` - `def create_edges_by_gemini(self, nodes_with_source, summarized_papers)`
- Uses structured generation with `KGEdge` schema: `genkg.py:112`
  ```python
  class KGEdge(BaseModel):
      node1: str      # First node name
      node2: str      # Second node name
      weight: float   # Relationship strength (0.0 to 1.0+)
      relation: str   # Relationship type label
  ```
- Constructs context (`genkg.py:235-242`):
  - **Node list**: All node texts formatted as bullet points (`node_list_str`)
  - **Paper context**: All paper summaries formatted as "--- Paper: {path} ---\n{summary}"
- Prompt (`genkg.py:244-258`):
  ```
  Given the following list of scientific concepts/nodes from a research paper knowledge
  graph, provide the most meaningful edges between them based on their relationships.

  For each edge, return an object with:
  - node1: the first node
  - node2: the second node
  - weight: a float indicating the strength of the relationship (e.g. 1.0 for strong,
            0.5 for medium, 0.15 for weak, etc.)
  - relation: a short label like 'related_to', 'enables', 'depends_on', etc.

  Only return meaningful and non-trivial relationships. There should be no orphan nodes,
  it should be a connected graph. Format as a JSON list.

  Nodes:
  {node_list_str}
  {context_str}  # Contains all paper summaries
  ```
- LLM returns JSON array: `[{"node1": "X", "node2": "Y", "weight": 0.8, "relation": "enables"}, ...]`
- System maps node names back to `(node_text, source)` tuples (`genkg.py:267`)
- Returns list of `(node1_with_source, node2_with_source, {"weight": w, "relation": r})` tuples (`genkg.py:275`)

**Node Normalization**: `_op.py:519-528` - Windows compatibility processing
```python
clean_node_text = (node_text.strip()
                 .replace('(', ' ')
                 .replace(')', ' ')
                 .replace('-', ' ')
                 .replace('/', ' ')
                 .replace('&', 'AND'))
clean_node_text = ' '.join(clean_node_text.split()).upper()
```
This ensures compatibility with nano-graphrag's uppercase entity naming convention and avoids issues with special characters in file systems.

**Entity Data Structure**: `_op.py:533-538`
```python
entity_data = {
    "entity_name": clean_node_text,  # Normalized uppercase
    "entity_type": "CONCEPT",  # GenKG doesn't provide types
    "description": node_text.strip(),  # Original for description
    "source_id": source,
}
```

**Edge Normalization**: `_op.py:547-570` - Same normalization applied to edge endpoints

**Graph Connectivity**: `_op.py:608-676` - Ensures graph is fully connected
- Calls `genkg.ensure_graph_connectivity(nodes_with_source, edges_for_connectivity)` (`_op.py:645`)
- Implementation: `genkg.py:287` - `def ensure_graph_connectivity(self, nodes_with_source, edges)`
- Process:
  1. Build temporary NetworkX graph (`genkg.py:292-303`)
  2. Find connected components: `nx.connected_components(temp_graph)` (`genkg.py:306`)
  3. If multiple components exist, load sentence transformer: `SentenceTransformer('all-MiniLM-L6-v2')` (`genkg.py:316`)
  4. For each isolated component (`genkg.py:325-367`):
     - Compute embeddings for component nodes and main component nodes
     - Find best semantic match: `util.pytorch_cos_sim(component_embeddings, main_embeddings)` (`genkg.py:342`)
     - Create connectivity edge with lower weight: `{"weight": max(0.1, best_similarity * 0.5), "relation": "semantic_similarity"}` (`genkg.py:361-364`)

**Visualization Data Storage**: `_op.py:717-743` - Save data for later visualization generation
- Stores nodes with source, processed edges, and papers_dict in JSON file
- File path: `os.path.join(global_config["working_dir"], "_genkg_viz_data.json")` (`_op.py:741`)
- Actual visualization generation happens later in `graphrag.py:380` - `async def _generate_genkg_visualizations()`

**Final Merging**: `_op.py:684-695` - Uses same merge functions as standard method
- Calls `_merge_nodes_then_upsert()` for all entities (`_op.py:684-688`)
- Calls `_merge_edges_then_upsert()` for all relationships (`_op.py:690-694`)

### 4. Graph Clustering (Leiden Algorithm)

**Trigger**: `graphrag.py:337` - `await self.chunk_entity_relation_graph.clustering(self.graph_cluster_algorithm)`

**Storage Class**: `_storage/gdb_networkx.py:19` - `class NetworkXStorage(BaseGraphStorage)`
- Uses NetworkX graph: `self._graph = preloaded_graph or nx.Graph()` (`gdb_networkx.py:88`)
- Graph persisted to GraphML file: `self._graphml_xml_file` (`gdb_networkx.py:80-82`)

**Clustering Dispatcher**: `gdb_networkx.py:135` - `async def clustering(self, algorithm: str)`
```python
self._clustering_algorithms = {
    "leiden": self._leiden_clustering,
}
await self._clustering_algorithms[algorithm]()
```

**Leiden Implementation**: `gdb_networkx.py:200` - `async def _leiden_clustering(self)`

**Component Processing**: `gdb_networkx.py:210-214`
```python
connected_components = list(nx.connected_components(self._graph))
logger.info(f"Processing {len(connected_components)} connected components for clustering")

for comp_idx, component_nodes in enumerate(connected_components):
    component_subgraph = self._graph.subgraph(component_nodes).copy()
```

**Graph Stabilization**: `gdb_networkx.py:218-222`
- Applies `_stabilize_graph()` to ensure deterministic node/edge ordering
- Node mapping: `node_mapping = {node: html.unescape(node.upper().strip()) for node in component_graph.nodes()}` (`gdb_networkx.py:221`)
- Relabels nodes: `nx.relabel_nodes(component_graph, node_mapping)` (`gdb_networkx.py:222`)

**Hierarchical Leiden**: `gdb_networkx.py:232-237`
```python
from graspologic.partition import hierarchical_leiden

community_mapping = hierarchical_leiden(
    component_graph,
    max_cluster_size=self.global_config["max_graph_cluster_size"],
    random_seed=self.global_config["graph_cluster_seed"],
)
```

**How `hierarchical_leiden` Works**:
- Returns a list of partition objects, where each partition represents one node's assignment to a community at a specific level
- Each partition object has attributes:
  - `partition.node`: The node ID
  - `partition.level`: The hierarchy level (0, 1, 2, ...) where 0 is finest granularity
  - `partition.cluster`: The cluster ID at that level
- Number of levels created is **automatic** - the algorithm creates as many levels as needed based on the graph structure
- Number of communities per level is **automatic** - determined by optimizing modularity while respecting `max_cluster_size`
- The algorithm creates the hierarchy by:
  1. Running Leiden on the full graph to create Level 0 communities (finest granularity, respecting `max_cluster_size`)
  2. Creating a super-graph where each Level 0 community becomes a node
  3. Running Leiden on the super-graph to create Level 1 communities
  4. Repeating this process recursively until no further meaningful groupings can be made
- Example output: If a graph has 50 nodes with `max_cluster_size=10`, the algorithm might create:
  - Level 0: 8 communities (each ≤10 nodes)
  - Level 1: 3 communities (groups of Level 0 communities)
  - Level 2: 1 community (the entire graph)

**Cluster Assignment**: `gdb_networkx.py:240-255`
- Processes each partition from `hierarchical_leiden` output
- Extracts level and cluster ID: `level_key = partition.level`, `cluster_id = f"{partition.cluster + cluster_offset}"`
- Maps back to original node IDs (reversing the `node_mapping`)
- Stores assignments: `node_communities[original_node_id].append({"level": level_key, "cluster": cluster_id})`
- Each node gets one assignment per level created by the algorithm

**Cluster Offset**: `gdb_networkx.py:257-260` - Ensures unique cluster IDs across components
```python
if community_mapping:
    max_cluster_in_component = max(p.cluster for p in community_mapping)
    cluster_offset += max_cluster_in_component + 1
```

**Storage**: `gdb_networkx.py:270` - `self._cluster_data_to_subgraphs(node_communities)`
- Implementation: `gdb_networkx.py:196-198`
  ```python
  def _cluster_data_to_subgraphs(self, cluster_data: dict[str, list[dict[str, str]]]):
      for node_id, clusters in cluster_data.items():
          self._graph.nodes[node_id]["clusters"] = json.dumps(clusters)
  ```

**Configuration Parameters**:
- `graph_cluster_algorithm: str = "leiden"` - Clustering algorithm (`graphrag.py:83`)
- `max_graph_cluster_size: int = 10` - Maximum nodes per cluster (`graphrag.py:84`)
- `graph_cluster_seed: int = 0xDEADBEEF` - Random seed for reproducibility (`graphrag.py:85`)

**Community Schema Retrieval**: `gdb_networkx.py:140` - `async def community_schema()`
- Iterates through all nodes: `for node_id, node_data in self._graph.nodes(data=True)`
- Parses clusters: `clusters = json.loads(node_data["clusters"])` (`gdb_networkx.py:157`)
- Builds community data structure: `SingleCommunitySchema` from `base.py:39`
  ```python
  SingleCommunitySchema = TypedDict(
      "SingleCommunitySchema",
      {
          "level": int,
          "title": str,
          "edges": list[list[str, str]],
          "nodes": list[str],
          "chunk_ids": list[str],
          "occurrence": float,
          "sub_communities": list[str],
      },
  )
  ```
- Computes sub-communities: `gdb_networkx.py:176-186` - Based on node subset relationships between levels

### 5. Community Report Generation

**Entry Point**: `graphrag.py:340` - `await generate_community_report(self.community_reports, self.chunk_entity_relation_graph, asdict(self))`

**Function**: `_op.py:920` - `async def generate_community_report(community_report_kv, knwoledge_graph_inst, global_config)`

**Prompt**: `prompt.py:64` - `PROMPTS["community_report"]`
- Requests JSON with fields: `title`, `summary`, `rating`, `rating_explanation`, `findings`
- Each finding has: `summary` and `explanation`
- Example output: `prompt.py:126-149`

**Community Schema Loading**: `_op.py:933` - `communities_schema = await knwoledge_graph_inst.community_schema()`
- Returns dict of `{cluster_id: SingleCommunitySchema}`
- Keys and values extracted: `community_keys, community_values` (`_op.py:934-936`)

**Level-by-Level Processing**: `_op.py:965-995`
```python
levels = sorted(set([c["level"] for c in community_values]), reverse=True)
logger.info(f"Generating by levels: {levels}")

for level in levels:
    this_level_community_keys, this_level_community_values = zip(
        *[
            (k, v)
            for k, v in zip(community_keys, community_values)
            if v["level"] == level
        ]
    )
```

**Report Generation**: `_op.py:939` - `async def _form_single_community_report(community, already_reports)`

**Context Packing**: `_op.py:943` - `describe = await _pack_single_community_describe(...)`
- Implementation: `_op.py:791` - `async def _pack_single_community_describe(knwoledge_graph_inst, community, max_token_size, already_reports, global_config)`

**Node Data Collection**: `_op.py:801-806`
```python
nodes_data = await asyncio.gather(
    *[knwoledge_graph_inst.get_node(n) for n in nodes_in_order]
)
edges_data = await asyncio.gather(
    *[knwoledge_graph_inst.get_edge(src, tgt) for src, tgt in edges_in_order]
)
```

**Node List Formation**: `_op.py:807-818`
```python
node_fields = ["id", "entity", "type", "description", "degree"]
nodes_list_data = [
    [
        i,
        node_name,
        node_data.get("entity_type", "UNKNOWN"),
        node_data.get("description", "UNKNOWN"),
        await knwoledge_graph_inst.node_degree(node_name),
    ]
    for i, (node_name, node_data) in enumerate(zip(nodes_in_order, nodes_data))
]
```

**Ranking and Truncation**: `_op.py:819-822`
- Sort by degree: `nodes_list_data = sorted(nodes_list_data, key=lambda x: x[-1], reverse=True)`
- Truncate by token size: `truncate_list_by_token_size(nodes_list_data, key=lambda x: x[3], max_token_size=max_token_size // 2)`

**Edge List Formation**: `_op.py:823-836` - Similar process for edges with rank-based sorting

**Sub-Community Handling**: `_op.py:844-881`
- If context exceeds token limit and sub-communities exist: `need_to_use_sub_communities = (truncated and len(community["sub_communities"]) and len(already_reports))` (`_op.py:844-846`)
- Calls `_pack_single_community_by_sub_communities()` (`_op.py:748`)
- Includes sub-community reports instead of raw node/edge data
- Structure: `_op.py:765`
  ```python
  sub_fields = ["id", "report", "rating", "importance"]
  sub_communities_describe = list_of_list_to_csv(
      [sub_fields]
      + [
          [
              i,
              c["report_string"],
              c["report_json"].get("rating", -1),
              c["occurrence"],
          ]
          for i, c in enumerate(may_trun_all_sub_communities)
      ]
  )
  ```

**CSV Formatting**: `_op.py:882-895`
```python
nodes_describe = list_of_list_to_csv([node_fields] + nodes_may_truncate_list_data)
edges_describe = list_of_list_to_csv([edge_fields] + edges_may_truncate_list_data)
return f"""-----Reports-----
```csv
{report_describe}
```
-----Entities-----
```csv
{nodes_describe}
```
-----Relationships-----
```csv
{edges_describe}
```"""
```

**LLM Call**: `_op.py:950-951`
```python
prompt = community_report_prompt.format(input_text=describe)
response = await use_llm_func(prompt, **llm_extra_kwargs)
```

**JSON Parsing**: `_op.py:953` - `data = use_string_json_convert_func(response)`
- Default converter: `convert_response_to_json` from `_utils.py`

**Report Storage**: `_op.py:982-995`
```python
community_datas.update(
    {
        k: {
            "report_string": _community_report_json_to_str(r),
            "report_json": r,
            **v,
        }
        for k, r, v in zip(
            this_level_community_keys,
            this_level_communities_reports,
            this_level_community_values,
        )
    }
)
```

**String Formatting**: `_op.py:898` - `def _community_report_json_to_str(parsed_output)`
- Converts JSON to markdown format with title, summary, and findings sections
- Structure: `_op.py:917`
  ```python
  return f"# {title}\n\n{summary}\n\n{report_sections}"
  ```

**Final Commit**: `_op.py:997` - `await community_report_kv.upsert(community_datas)`

**Configuration Parameters**:
- `best_model_func: callable` - LLM for report generation (`graphrag.py:115`)
- `best_model_max_token_size: int = 32768` - Context limit for LLM (`graphrag.py:116`)
- `special_community_report_llm_kwargs: dict = {"response_format": {"type": "json_object"}}` - Force JSON output (`graphrag.py:102-104`)
- `convert_response_to_json_func: callable = convert_response_to_json` - JSON parsing function (`graphrag.py:143`)

### 6. Storage & Indexing

**Callback System**: The pipeline uses async callbacks for transactional storage operations.

**Start Callback**: `graphrag.py:354` - `async def _insert_start(self)`
- Called at the beginning of insertion
- Implementation: `graphrag.py:354-362`
  ```python
  tasks = []
  for storage_inst in [
      self.chunk_entity_relation_graph,
  ]:
      if storage_inst is None:
          continue
      tasks.append(cast(StorageNameSpace, storage_inst).index_start_callback())
  await asyncio.gather(*tasks)
  ```

**Done Callback**: `graphrag.py:364` - `async def _insert_done(self)`
- Called after all processing completes (in finally block)
- Commits data to all storages
- Implementation: `graphrag.py:364-378`
  ```python
  tasks = []
  for storage_inst in [
      self.full_docs,
      self.text_chunks,
      self.llm_response_cache,
      self.community_reports,
      self.entities_vdb,
      self.chunks_vdb,
      self.chunk_entity_relation_graph,
  ]:
      if storage_inst is None:
          continue
      tasks.append(cast(StorageNameSpace, storage_inst).index_done_callback())
  await asyncio.gather(*tasks)
  ```

**Storage Types**:

1. **Key-Value Storage**: `BaseKVStorage` from `base.py:95`
   - Default implementation: `JsonKVStorage` from `_storage/kv_json.py`
   - Stores data as JSON files in working directory
   - Used for: `full_docs`, `text_chunks`, `llm_response_cache`, `community_reports`
   - Interface methods: `get_by_id()`, `get_by_ids()`, `upsert()`, `filter_keys()`, `drop()`

2. **Vector Storage**: `BaseVectorStorage` from `base.py:80`
   - Default implementation: `NanoVectorDBStorage` from `_storage/vdb_nanovectordb.py`
   - Alternative: `HNSWVectorStorage` from `_storage/vdb_hnswlib.py`
   - Used for: `entities_vdb` (entity embeddings), `chunks_vdb` (chunk embeddings for naive RAG)
   - Interface methods: `query(query: str, top_k: int)`, `upsert(data: dict[str, dict])`
   - Embedding function: Configured via `embedding_func` parameter (default: OpenAI's text-embedding-3-small)

3. **Graph Storage**: `BaseGraphStorage` from `base.py:119`
   - Default implementation: `NetworkXStorage` from `_storage/gdb_networkx.py`
   - Alternative: `Neo4jStorage` from `_storage/gdb_neo4j.py`
   - Used for: `chunk_entity_relation_graph`
   - Interface methods: `upsert_node()`, `upsert_edge()`, `get_node()`, `get_edge()`, `clustering()`, `community_schema()`
   - Persistence: GraphML file written in `index_done_callback()` (`gdb_networkx.py:96-97`)

**NetworkX GraphML Persistence**: `gdb_networkx.py:27`
```python
@staticmethod
def write_nx_graph(graph: nx.Graph, file_name):
    logger.info(
        f"Writing graph with {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges"
    )
    nx.write_graphml(graph, file_name)
```

**Final Storage Operations**: `graphrag.py:349-350`
```python
await self.full_docs.upsert(new_docs)
await self.text_chunks.upsert(inserting_chunks)
```

**GenKG Visualization Generation**: `graphrag.py:344-346`
```python
if self.use_genkg_extraction and self.genkg_create_visualization:
    await self._generate_genkg_visualizations(inserting_chunks, new_docs)
```

**Visualization Function**: `graphrag.py:380` - `async def _generate_genkg_visualizations(self, inserting_chunks, new_docs)`

**Process**:
1. Load stored visualization data: `graphrag.py:385-395`
   ```python
   import json
   viz_data_path = os.path.join(self.working_dir, "_genkg_viz_data.json")
   with open(viz_data_path, 'r', encoding='utf-8') as f:
       genkg_viz_data = json.load(f)
   ```

2. Create NetworkX graph from nano-graphrag processed data: `graphrag.py:428-451`
   ```python
   import networkx as nx
   knowledge_graph = nx.Graph()

   for node_text, source in nodes_with_source:
       knowledge_graph.add_node(node_text,
                               source=source,
                               color=paper_colors.get(source, "#808080"),
                               title=f"Source: {source}")

   for edge_data in edges_data:
       src_id = edge_data.get("src_id")
       tgt_id = edge_data.get("tgt_id")
       weight = edge_data.get("weight", 1.0)
       relation = edge_data.get("description", "related_to")

       if src_id and tgt_id and src_id in knowledge_graph.nodes and tgt_id in knowledge_graph.nodes:
           knowledge_graph.add_edge(src_id, tgt_id, weight=weight, relation=relation)
   ```

3. Export to dashkg.json: `graphrag.py:457-458`
   ```python
   output_json_path = os.path.join(self.working_dir, "output.dashkg.json")
   genkg.export_graph_to_dashkg_json(knowledge_graph, output_json_path)
   ```
   - Implementation: `genkg.py:576` - `def export_graph_to_dashkg_json(self, graph, output_path)`
   - Format: JSON with `metadata`, `nodes`, and `edges` arrays
   - Metadata includes: `total_nodes`, `total_edges`, `sources`, `connected_components`, `graph_density`, `average_degree`

4. Create HTML visualization: `graphrag.py:461-462`
   ```python
   html_path = os.path.join(self.working_dir, "output.html")
   genkg.advanced_graph_to_html(knowledge_graph, html_path, display=False)
   ```
   - Implementation: `genkg.py:451` - `def advanced_graph_to_html(self, graph, path, display)`
   - Uses pyvis for interactive visualization
   - Includes: node colors by source, size by degree, tooltips, relationship labels, legend, statistics

**Configuration Parameters**:
- `working_dir: str` - Directory for all storage files (`graphrag.py:56-57`)
- `key_string_value_json_storage_cls: Type[BaseKVStorage] = JsonKVStorage` - KV storage class (`graphrag.py:126`)
- `vector_db_storage_cls: Type[BaseVectorStorage] = NanoVectorDBStorage` - Vector DB class (`graphrag.py:127`)
- `graph_storage_cls: Type[BaseGraphStorage] = NetworkXStorage` - Graph storage class (`graphrag.py:129`)
- `enable_llm_cache: bool = True` - Cache LLM responses (`graphrag.py:130`)
- `genkg_output_path: Optional[str] = None` - Custom path for visualization files (`graphrag.py:138`)

---

## Summary

The nano-graphrag document insertion pipeline transforms raw text documents into a rich, queryable knowledge graph through six major phases:

1. **Deduplication**: MD5-based hashing prevents duplicate processing
2. **Chunking**: Overlapping text chunks with configurable size and tokenization
3. **Extraction**: Two methods available - standard LLM prompting or GenKG's document-centric approach with connectivity enhancement
4. **Clustering**: Hierarchical Leiden algorithm on all connected components creates multi-level communities
5. **Reporting**: LLM-generated structured reports for each community with context management
6. **Storage**: Persistent storage across multiple backends (JSON, vector DB, graph DB) with optional visualizations

The pipeline is fully asynchronous, supports incremental updates, and provides extensive configuration options for customization. Entity normalization, graph connectivity, and robust error handling ensure high-quality knowledge graph construction from diverse document sources.
