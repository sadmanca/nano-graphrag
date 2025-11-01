# Nano-GraphRAG Pipeline Diagrams

This document contains Mermaid diagrams visualizing the main pipelines in nano-graphrag.

---

## 1. Document Insertion Pipeline

```mermaid
flowchart TD
    Start([User provides documents]) --> Hash[Compute MD5 Hash IDs<br/>doc-abc123...]
    Hash --> Dedup{Already<br/>exists?}
    Dedup -->|Yes| Skip[Skip document]
    Dedup -->|No| Chunk[Text Chunking & Tokenization]

    Chunk --> ChunkProcess[Split into overlapping chunks<br/>Default: 1200 tokens, 100 overlap<br/>Assign chunk-xyz... IDs]
    ChunkProcess --> ChunkDedup{Chunk<br/>exists?}
    ChunkDedup -->|Yes| SkipChunk[Skip chunk]
    ChunkDedup -->|No| ExtractChoice{Extraction<br/>Method?}

    ExtractChoice -->|Standard| StandardExtract[Standard Entity Extraction<br/>Process each chunk with LLM<br/>Extract entities & relationships]
    ExtractChoice -->|GenKG| GenKGExtract[GenKG Entity Extraction<br/>Summarize full document<br/>Extract high-level concepts]

    StandardExtract --> Merge[Merge & Deduplicate Entities<br/>Concatenate descriptions<br/>Sum relationship weights]
    GenKGExtract --> GenKGNorm[Normalize Node Names<br/>Uppercase, remove special chars<br/>Ensure connectivity]
    GenKGNorm --> Merge

    Merge --> VectorDB[Index Entities in Vector DB<br/>For similarity search during queries]
    VectorDB --> Cluster[Graph Clustering<br/>Hierarchical Leiden Algorithm]

    Cluster --> ClusterProcess[Process all connected components<br/>Create hierarchical communities<br/>Level 0 → Level N]
    ClusterProcess --> Report[Community Report Generation<br/>Generate natural language reports<br/>Process highest level first]

    Report --> Store[Storage & Indexing<br/>Persist to disk/database<br/>GraphML, JSON, Vector DB]
    Store --> GenKGViz{GenKG<br/>visualization?}
    GenKGViz -->|Yes| CreateViz[Create HTML & JSON outputs<br/>Interactive graph visualization]
    GenKGViz -->|No| End
    CreateViz --> End([Pipeline Complete<br/>Ready for queries])

    Skip --> End
    SkipChunk --> End

    style Start fill:#e1f5ff
    style End fill:#c8e6c9
    style Cluster fill:#fff9c4
    style Report fill:#fff9c4
    style StandardExtract fill:#ffe0b2
    style GenKGExtract fill:#ffe0b2
    style Store fill:#f3e5f5
```

---

## 2. Community Creation Pipeline (Detailed)

### Phase 1: Graph Clustering

```mermaid
flowchart TD
    Start([Knowledge Graph Available<br/>Nodes + Edges]) --> FindComp[Find Connected Components<br/>nx.connected_components]
    FindComp --> CompCount{Multiple<br/>components?}

    CompCount -->|Yes| ProcessComp[Process Each Component Separately]
    CompCount -->|No| ProcessComp

    ProcessComp --> Stabilize[Stabilize Graph<br/>Sort nodes & edges<br/>Normalize names to uppercase]

    Stabilize --> HierLeiden[Hierarchical Leiden Clustering<br/>Input: Component graph<br/>Config: max_cluster_size=10]

    HierLeiden --> Level0[Level 0: Base Communities<br/>Run Leiden on graph<br/>Optimize modularity<br/>Respect max_cluster_size]

    Level0 --> CreateSuper[Create Super-Graph<br/>Each Level 0 community → super-node<br/>Edge weights = connection counts]

    CreateSuper --> Level1[Level 1: Group Communities<br/>Run Leiden on super-graph<br/>Groups of Level 0 communities]

    Level1 --> CheckTerm{Termination?<br/>Only 1 community<br/>OR no improvement<br/>OR graph too small}

    CheckTerm -->|No| CreateSuper2[Create Level N+1 Super-Graph]
    CreateSuper2 --> LevelN[Level N+1: Higher Grouping<br/>Continue recursion]
    LevelN --> CheckTerm

    CheckTerm -->|Yes| AssignIDs[Assign Cluster IDs with Offset<br/>Ensure uniqueness across components]

    AssignIDs --> StoreJSON[Store in Node Attributes<br/>node clusters = JSON array<br/>level: 0, cluster: 5, etc.]

    StoreJSON --> MoreComp{More<br/>components?}
    MoreComp -->|Yes| ProcessComp
    MoreComp -->|No| BuildSchema[Build Community Schema<br/>Aggregate nodes, edges, chunks<br/>Per community]

    BuildSchema --> EndPhase1([Clustering Complete<br/>Hierarchical structure created])

    style Start fill:#e1f5ff
    style EndPhase1 fill:#fff9c4
    style HierLeiden fill:#ffe0b2
    style Level0 fill:#ffccbc
    style Level1 fill:#ffccbc
    style LevelN fill:#ffccbc
```

### Phase 2: Report Generation

```mermaid
flowchart TD
    Start([Hierarchical Communities Available]) --> GetSchema[Get Community Schema<br/>All communities with metadata]

    GetSchema --> ExtractLevels[Extract Unique Levels<br/>e.g., 0, 1, 2]

    ExtractLevels --> SortLevels[Sort Levels: Highest to Lowest<br/>e.g., 2, 1, 0]

    SortLevels --> LoopLevels{For each<br/>level}

    LoopLevels --> FilterComm[Filter Communities at Current Level]

    FilterComm --> LoopComm{For each<br/>community}

    LoopComm --> GetNodes[Gather Community Data<br/>Nodes list, Edges list<br/>Sub-communities if any]

    GetNodes --> FetchDetails[Fetch Full Details from Graph<br/>Node: type, description, degree<br/>Edge: weight, description, rank]

    FetchDetails --> RankNodes[Rank by Importance<br/>Nodes: Sort by degree<br/>Edges: Sort by rank]

    RankNodes --> CheckTokens{Exceeds<br/>token limit?}

    CheckTokens -->|Yes, has subs| UseSubs[Use Sub-Community Reports<br/>Include summaries instead of raw data]
    CheckTokens -->|Yes, no subs| Truncate[Truncate to Fit<br/>Keep highest-ranked items]
    CheckTokens -->|No| FormatCSV[Format as CSV Tables<br/>Entities table<br/>Relationships table<br/>Reports table if using subs]

    UseSubs --> FormatCSV
    Truncate --> FormatCSV

    FormatCSV --> Prompt[Create LLM Prompt<br/>community_report prompt<br/>Insert CSV context]

    Prompt --> CallLLM[Call Best Model LLM<br/>Request JSON format<br/>title, summary, rating, findings]

    CallLLM --> ParseJSON[Parse JSON Response<br/>Extract structured data]

    ParseJSON --> ConvertMD[Convert to Markdown<br/>Format as readable report]

    ConvertMD --> StoreBoth[Store Both Formats<br/>report_string markdown<br/>report_json structured]

    StoreBoth --> NextComm[Next Community]
    NextComm --> LoopComm

    LoopComm -->|Done| SaveLevel[Save All Reports for Level<br/>Upsert to community_reports storage]

    SaveLevel --> NextLevel[Next Level]
    NextLevel --> LoopLevels

    LoopLevels -->|Done| End([All Community Reports Generated<br/>Ready for query answering])

    style Start fill:#e1f5ff
    style End fill:#c8e6c9
    style CallLLM fill:#ffe0b2
    style StoreBoth fill:#f3e5f5
    style UseSubs fill:#fff9c4
```

---

## 3. Local Search Pipeline

```mermaid
flowchart TD
    Start([User Query:<br/>How does backpropagation work?]) --> Embed[Embed Query<br/>Convert to vector representation]

    Embed --> SearchVDB[Search Entity Vector DB<br/>Find top K similar entities<br/>Default: top_k=20]

    SearchVDB --> ResultEntities[Retrieved Entities<br/>BACKPROPAGATION 0.89<br/>GRADIENT DESCENT 0.76<br/>NEURAL NETWORK 0.74<br/>...]

    ResultEntities --> GetDetails[Get Full Entity Details<br/>Fetch from graph storage<br/>Get node degree connectivity]

    GetDetails --> GetCommunities[Find Related Communities<br/>Parse clusters from entities<br/>Filter by level ≤ query level]

    GetCommunities --> RankComm[Rank Communities<br/>Sort by occurrence count<br/>Sort by rating]

    RankComm --> TruncComm[Truncate to Token Budget<br/>Max: 3200 tokens<br/>Keep highest-ranked]

    TruncComm --> GetEdges[Find Related Relationships<br/>Get edges connected to entities<br/>Add edge weight & description]

    GetEdges --> RankEdges[Rank Relationships<br/>Sort by rank degree src + degree tgt<br/>Truncate to 4800 tokens]

    RankEdges --> ChunksCheck{Include<br/>text chunks?}

    ChunksCheck -->|Yes| GetChunks[Get Original Text Chunks<br/>Parse source_id from entities<br/>Fetch from text_chunks storage]
    ChunksCheck -->|No| FormatContext

    GetChunks --> RankChunks[Rank Text Chunks<br/>Sort by chunk order<br/>Truncate to 4000 tokens]

    RankChunks --> FormatContext[Format Context as CSV Tables<br/>Reports table<br/>Entities table<br/>Relationships table<br/>Sources table if included]

    FormatContext --> BuildPrompt[Build LLM Prompt<br/>local_rag_response system prompt<br/>Insert context tables<br/>Add user query]

    BuildPrompt --> CallLLM[Call Best Model LLM<br/>Single pass generation<br/>No special format required]

    CallLLM --> Answer[Generate Detailed Answer<br/>Based on local context<br/>Cites specific entities/relationships]

    Answer --> End([Return Answer to User<br/>Focused, detailed response])

    style Start fill:#e1f5ff
    style End fill:#c8e6c9
    style SearchVDB fill:#ffe0b2
    style CallLLM fill:#ffe0b2
    style GetCommunities fill:#fff9c4
    style GetEdges fill:#fff9c4
    style GetChunks fill:#f3e5f5
```

---

## 4. Global Search Pipeline

```mermaid
flowchart TD
    Start([User Query:<br/>What are the main themes?]) --> GetAll[Get All Communities<br/>Retrieve community schema<br/>From graph storage]

    GetAll --> FilterLevel[Filter by Level<br/>Keep communities where<br/>level ≤ query level default 2]

    FilterLevel --> RankOccur[Rank by Importance<br/>Sort by occurrence<br/>How much of graph they cover]

    RankOccur --> SelectTop[Select Top N Communities<br/>Default: top 512<br/>Filter by min rating ≥ 0]

    SelectTop --> GetReports[Retrieve Community Reports<br/>Fetch full report_string<br/>From community_reports storage]

    GetReports --> GroupComm[Group Communities by Token Budget<br/>Split into groups that fit<br/>Max: 16384 tokens per group]

    GroupComm --> GroupList[Groups Created<br/>Group 1: Communities 0-30<br/>Group 2: Communities 31-65<br/>Group 3: Communities 66-95<br/>...]

    GroupList --> MapPhase[MAP PHASE: Process Each Group]

    MapPhase --> LoopGroups{For each<br/>group}

    LoopGroups --> FormatGroup[Format Group as CSV<br/>id, content, rating, importance<br/>All community reports in group]

    FormatGroup --> MapPrompt[Create Map Prompt<br/>global_map_rag_points system prompt<br/>Request JSON key points<br/>Each with description & score]

    MapPrompt --> MapLLM[Call Best Model LLM<br/>Extract key points from group<br/>Return JSON array]

    MapLLM --> ParsePoints[Parse JSON Response<br/>Extract points with scores]

    ParsePoints --> NextGroup[Next Group]
    NextGroup --> LoopGroups

    LoopGroups -->|Done| ReducePhase[REDUCE PHASE: Combine All Points]

    ReducePhase --> Aggregate[Aggregate All Points<br/>From all groups<br/>Label with analyst ID]

    Aggregate --> FilterScore[Filter Low-Score Points<br/>Keep only score > 0]

    FilterScore --> RankPoints[Rank All Points<br/>Sort by importance score<br/>Highest first]

    RankPoints --> TruncPoints[Truncate to Token Budget<br/>Max: 16384 tokens<br/>Keep highest-scored]

    TruncPoints --> FormatAnalysts[Format as Analyst Reports<br/>----Analyst 0----<br/>Importance Score: 85<br/>Description...<br/>...]

    FormatAnalysts --> ReducePrompt[Create Reduce Prompt<br/>global_reduce_rag_response<br/>Synthesize all perspectives<br/>Insert analyst reports]

    ReducePrompt --> ReduceLLM[Call Best Model LLM<br/>Generate comprehensive synthesis<br/>Combine multiple viewpoints]

    ReduceLLM --> FinalAnswer[Generate Comprehensive Answer<br/>Synthesizes patterns across graph<br/>Multiple perspectives integrated]

    FinalAnswer --> End([Return Answer to User<br/>Broad, thematic response])

    style Start fill:#e1f5ff
    style End fill:#c8e6c9
    style MapPhase fill:#fff9c4
    style ReducePhase fill:#fff9c4
    style MapLLM fill:#ffe0b2
    style ReduceLLM fill:#ffe0b2
    style GroupComm fill:#f3e5f5
```

---

## 5. Comparison: Local vs. Global Search Flow

```mermaid
graph TB
    subgraph Local["LOCAL SEARCH - Deep & Narrow"]
        L1[User Query] --> L2[Find Similar Entities<br/>Vector search]
        L2 --> L3[Get Entity Neighborhood<br/>Communities + Relationships + Chunks]
        L3 --> L4[Single LLM Call<br/>~10K tokens context]
        L4 --> L5[Detailed Answer<br/>Cites specific entities]
    end

    subgraph Global["GLOBAL SEARCH - Broad & Shallow"]
        G1[User Query] --> G2[Get All Communities<br/>Top 512 by importance]
        G2 --> G3[MAP: Process in Groups<br/>Multiple parallel LLM calls]
        G3 --> G4[Extract Key Points<br/>From each group]
        G4 --> G5[REDUCE: Synthesize<br/>Single LLM call]
        G5 --> G6[Comprehensive Answer<br/>Synthesizes patterns]
    end

    Query([Same User Query]) -.->|Specific question| L1
    Query -.->|Broad question| G1

    style Local fill:#e3f2fd
    style Global fill:#fff3e0
    style Query fill:#f3e5f5
```

---

## 6. Entity Extraction Methods Comparison

```mermaid
flowchart LR
    subgraph Standard["Standard Extraction"]
        S1[Text Chunks] --> S2[Process Each Chunk<br/>Independently]
        S2 --> S3[LLM Extraction<br/>Per chunk]
        S3 --> S4[Entities with Types<br/>person, org, geo, event]
        S4 --> S5[Relationships with<br/>Descriptions & Weights]
        S5 --> S6[Merge Across Chunks<br/>Same entity = concatenate]
        S6 --> S7[Final Graph<br/>Granular entities]
    end

    subgraph GenKG["GenKG Extraction"]
        G1[Text Chunks] --> G2[Group by Document<br/>Reconstruct full text]
        G2 --> G3[Summarize Document<br/>4000 chars]
        G3 --> G4[Extract Concepts<br/>High-level only]
        G4 --> G5[All as CONCEPT type<br/>No type distinction]
        G5 --> G6[Create Edges with Context<br/>All nodes + all summaries]
        G6 --> G7[Ensure Connectivity<br/>Semantic similarity]
        G7 --> G8[Final Graph<br/>High-level concepts]
    end

    Docs([Input Documents]) --> S1
    Docs --> G1

    style Standard fill:#e3f2fd
    style GenKG fill:#f3e5f5
    style Docs fill:#c8e6c9
```

---

## 7. Hierarchical Leiden Clustering (Detailed)

```mermaid
flowchart TD
    Start([Graph: 50 nodes<br/>max_cluster_size=10]) --> L0Start[Level 0: Run Leiden]

    L0Start --> L0Process[Optimize Modularity<br/>Move nodes between communities<br/>Respect max_cluster_size ≤ 10]

    L0Process --> L0Result[Level 0 Result:<br/>8 communities<br/>Comm0: 4 nodes<br/>Comm1: 3 nodes<br/>Comm2: 5 nodes<br/>Comm3: 10 nodes<br/>... 8 total]

    L0Result --> Super1[Create Level 1 Super-Graph<br/>8 super-nodes<br/>1 per Level 0 community]

    Super1 --> Super1Edges[Add Super-Edges<br/>Weight = connection count<br/>between communities]

    Super1Edges --> L1Start[Level 1: Run Leiden<br/>On super-graph]

    L1Start --> L1Process[Optimize Modularity<br/>On 8 super-nodes]

    L1Process --> L1Result[Level 1 Result:<br/>3 communities<br/>L1_Comm0: Comm0,1,2<br/>L1_Comm1: Comm3,4,5<br/>L1_Comm2: Comm6,7]

    L1Result --> Super2[Create Level 2 Super-Graph<br/>3 super-nodes<br/>1 per Level 1 community]

    Super2 --> L2Start[Level 2: Run Leiden<br/>On super-graph]

    L2Start --> L2Process[Optimize Modularity<br/>On 3 super-nodes]

    L2Process --> L2Result[Level 2 Result:<br/>1 community<br/>L2_Comm0: All nodes]

    L2Result --> Terminate{Termination Check<br/>Only 1 community?}

    Terminate -->|Yes| Store[Store Assignments<br/>Each node has:<br/>Level 0: cluster X<br/>Level 1: cluster Y<br/>Level 2: cluster Z]

    Terminate -->|No| Continue[Continue to Level 3...]
    Continue --> Super3[Create Level 3 Super-Graph...]

    Store --> End([Hierarchy Complete<br/>3 levels created<br/>L0: 8 comm, L1: 3 comm, L2: 1 comm])

    style Start fill:#e1f5ff
    style End fill:#c8e6c9
    style L0Result fill:#ffccbc
    style L1Result fill:#ffccbc
    style L2Result fill:#ffccbc
    style Super1 fill:#fff9c4
    style Super2 fill:#fff9c4
```

---

## 8. Data Flow: Document to Query

```mermaid
flowchart LR
    subgraph Input["Input Stage"]
        Doc[Documents]
    end

    subgraph Processing["Processing Stage"]
        Chunks[Text Chunks<br/>1200 tokens each]
        Entities[Entities<br/>Nodes + Edges]
        Clusters[Communities<br/>Hierarchical]
        Reports[Community Reports<br/>Natural language]
    end

    subgraph Storage["Storage Stage"]
        VDB[(Vector DB<br/>Entity embeddings)]
        GraphDB[(Graph DB<br/>NetworkX/Neo4j)]
        KV[(Key-Value Store<br/>Chunks, Docs, Reports)]
    end

    subgraph Query["Query Stage"]
        LocalQ[Local Search<br/>Entity-focused]
        GlobalQ[Global Search<br/>Community-focused]
        Answer[Generated Answer]
    end

    Doc --> Chunks
    Chunks --> Entities
    Entities --> Clusters
    Clusters --> Reports

    Entities --> VDB
    Entities --> GraphDB
    Clusters --> GraphDB
    Chunks --> KV
    Reports --> KV
    Doc --> KV

    VDB --> LocalQ
    GraphDB --> LocalQ
    KV --> LocalQ

    GraphDB --> GlobalQ
    KV --> GlobalQ

    LocalQ --> Answer
    GlobalQ --> Answer

    style Input fill:#e1f5ff
    style Processing fill:#fff9c4
    style Storage fill:#f3e5f5
    style Query fill:#c8e6c9
```

---

## 9. Token Budget Management in Local Search

```mermaid
flowchart TD
    Start([Total Budget: 12,000 tokens]) --> Split[Split Budget:<br/>Text Chunks: 4,000<br/>Relationships: 4,800<br/>Community Reports: 3,200]

    Split --> GetData[Fetch All Data:<br/>100 potential entities<br/>300 potential edges<br/>50 potential chunks<br/>5 potential communities]

    GetData --> RankEnt[Rank Entities<br/>By node degree<br/>connectivity]

    RankEnt --> TruncEnt{Entities fit<br/>in budget?}

    TruncEnt -->|No| KeepTop[Keep Top N Entities<br/>Until 4,000 token limit]
    TruncEnt -->|Yes| AllEnt[Keep All Entities]

    KeepTop --> RankEdge[Rank Edges<br/>By rank src + tgt degree]
    AllEnt --> RankEdge

    RankEdge --> TruncEdge{Edges fit<br/>in budget?}

    TruncEdge -->|No| KeepTopEdge[Keep Top N Edges<br/>Until 4,800 token limit]
    TruncEdge -->|Yes| AllEdge[Keep All Edges]

    KeepTopEdge --> RankComm[Rank Communities<br/>By occurrence & rating]
    AllEdge --> RankComm

    RankComm --> TruncComm{Communities fit<br/>in budget?}

    TruncComm -->|No| KeepTopComm[Keep Top N Communities<br/>Until 3,200 token limit]
    TruncComm -->|Yes| AllComm[Keep All Communities]

    KeepTopComm --> CheckChunks{Include<br/>text chunks?}
    AllComm --> CheckChunks

    CheckChunks -->|Yes| RankChunks[Rank Chunks<br/>By order & relation count]
    CheckChunks -->|No| Format

    RankChunks --> TruncChunks{Chunks fit<br/>in budget?}

    TruncChunks -->|No| KeepTopChunks[Keep Top N Chunks<br/>Until 4,000 token limit]
    TruncChunks -->|Yes| AllChunks[Keep All Chunks]

    KeepTopChunks --> Format[Format as Context<br/>CSV tables for LLM]
    AllChunks --> Format

    Format --> Verify{Total tokens<br/>≤ 12,000?}

    Verify -->|Yes| Send[Send to LLM<br/>Generate answer]
    Verify -->|No| Error[Error: Budget exceeded<br/>Should not happen]

    Send --> End([Answer Generated])

    style Start fill:#e1f5ff
    style End fill:#c8e6c9
    style TruncEnt fill:#ffccbc
    style TruncEdge fill:#ffccbc
    style TruncComm fill:#ffccbc
    style TruncChunks fill:#ffccbc
    style Send fill:#ffe0b2
```

---

## 10. Map-Reduce in Global Search (Detailed)

```mermaid
sequenceDiagram
    participant User
    participant System
    participant Group1 as Community Group 1<br/>(30 communities)
    participant Group2 as Community Group 2<br/>(35 communities)
    participant Group3 as Community Group 3<br/>(28 communities)
    participant LLM1 as LLM (Map 1)
    participant LLM2 as LLM (Map 2)
    participant LLM3 as LLM (Map 3)
    participant Reduce as Reduce Phase
    participant LLMFinal as LLM (Final)

    User->>System: Query: What are main themes?
    System->>System: Retrieve 512 communities
    System->>System: Split into groups by token budget

    par Parallel Map Phase
        System->>Group1: Communities 0-29 (~16K tokens)
        System->>Group2: Communities 30-64 (~16K tokens)
        System->>Group3: Communities 65-92 (~16K tokens)

        Group1->>LLM1: Extract key points + scores
        Group2->>LLM2: Extract key points + scores
        Group3->>LLM3: Extract key points + scores

        LLM1->>System: [Point1: 85, Point2: 78, ...]
        LLM2->>System: [Point3: 82, Point4: 75, ...]
        LLM3->>System: [Point5: 80, Point6: 72, ...]
    end

    System->>Reduce: Aggregate all points
    Reduce->>Reduce: Filter score > 0
    Reduce->>Reduce: Rank by importance
    Reduce->>Reduce: Truncate to 16K tokens

    Reduce->>LLMFinal: Synthesize all perspectives<br/>into coherent answer
    LLMFinal->>System: Comprehensive answer
    System->>User: Return synthesized answer

    Note over LLM1,LLM3: Map phase extracts<br/>insights from<br/>different parts<br/>of graph
    Note over LLMFinal: Reduce phase<br/>combines all<br/>insights into<br/>one answer
```

---

## Notes on Diagram Usage

### Viewing Mermaid Diagrams

These diagrams can be viewed in:
- **GitHub**: Automatically renders Mermaid diagrams in `.md` files
- **VS Code**: Install "Markdown Preview Mermaid Support" extension
- **Mermaid Live Editor**: Copy-paste code to https://mermaid.live
- **GitLab**: Native Mermaid support in markdown
- **Notion, Obsidian**: Support Mermaid rendering

### Diagram Legend

- **Light Blue** (`#e1f5ff`): Start/Input nodes
- **Light Green** (`#c8e6c9`): End/Output nodes
- **Light Yellow** (`#fff9c4`): Important processing steps
- **Light Orange** (`#ffe0b2`): LLM operations
- **Light Purple** (`#f3e5f5`): Storage/Database operations
- **Light Red** (`#ffccbc`): Truncation/Filtering operations

### Customization

To modify these diagrams:
1. Copy the Mermaid code block
2. Paste into Mermaid Live Editor
3. Edit the flowchart
4. Copy the updated code back

### Related Documentation

- `document_insertion_pipeline.md`: Detailed text explanation of insertion pipeline
- `community_creation_flowchart.md`: ASCII flowcharts for community creation
- `query_modes_explained.md`: Detailed explanation of local vs. global search
