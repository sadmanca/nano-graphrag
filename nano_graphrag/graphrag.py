import asyncio
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime
from functools import partial
from typing import Callable, Dict, List, Optional, Type, Union, cast

import tiktoken


from ._llm import (
    gpt_4o_complete,
    gpt_4o_mini_complete,
    openai_embedding,
    azure_gpt_4o_complete,
    azure_openai_embedding,
    azure_gpt_4o_mini_complete,
    gemini_2_5_flash_complete,
    gemini_1_5_pro_complete,
    gemini_embedding,
)
from ._op import (
    chunking_by_token_size,
    chunking_large_context,  # New optimized chunking for large context models
    extract_entities,
    extract_entities_genkg,
    generate_community_report,
    get_chunks,
    local_query,
    global_query,
    naive_query,
)
from ._storage import (
    JsonKVStorage,
    NanoVectorDBStorage,
    NetworkXStorage,
)
from ._utils import (
    EmbeddingFunc,
    compute_mdhash_id,
    limit_async_func_call,
    convert_response_to_json,
    always_get_an_event_loop,
    logger,
)
from .base import (
    BaseGraphStorage,
    BaseKVStorage,
    BaseVectorStorage,
    StorageNameSpace,
    QueryParam,
)
from .prompt import GRAPH_FIELD_SEP

# Enhanced KG imports - using new enhanced_kg module
try:  # pragma: no cover - best effort import
    from .enhanced_kg import gemini_create_nodes, create_edges_by_gemini, EnhancedKG
    _HAS_ENHANCED_KG_FUNCS = True
except Exception:  # noqa: E722
    gemini_create_nodes = None  # type: ignore
    create_edges_by_gemini = None  # type: ignore
    EnhancedKG = None  # type: ignore
    _HAS_ENHANCED_KG_FUNCS = False

# Genkg.py fallback for enhanced node/edge generation
try:  # pragma: no cover - genkg.py fallback
    if not _HAS_ENHANCED_KG_FUNCS:
        # Try to import from genkg.py
        import sys
        from pathlib import Path
        genkg_path = Path(__file__).parent.parent
        if str(genkg_path) not in sys.path:
            sys.path.insert(0, str(genkg_path))
        
        from genkg import GenerateKG
        # Create a global genkg instance to use for node/edge creation
        _genkg_instance = GenerateKG(llm_provider="gemini", model_name="gemini-2.5-flash")
        
        # Wrap genkg methods to match expected signatures
        def gemini_create_nodes(content, node_limit, source_id):
            """Wrapper for genkg.gemini_create_nodes"""
            try:
                return _genkg_instance.gemini_create_nodes(content, node_limit, source_id)
            except Exception as e:
                print(f"Error in genkg node creation: {e}")
                return []
        
        def create_edges_by_gemini(nodes_with_source, papers_dict, model_name=None):
            """Wrapper for genkg.create_edges_by_gemini"""
            try:
                # Convert nodes_with_source from (node_id, node_data) format to expected format
                converted_nodes = [(node_data.get("description", node_id), node_data.get("source_id", "")) 
                                 for node_id, node_data in nodes_with_source]
                # genkg.py method expects summarized_papers, but we have papers_dict (full content)
                # For now, just use the papers_dict as-is since it's passed as summarized_papers parameter
                return _genkg_instance.create_edges_by_gemini(converted_nodes, papers_dict)
            except Exception as e:
                print(f"Error in genkg edge creation: {e}")
                return []
        
        _HAS_GEMINI_CREATE_FUNCS = True
        print("Using genkg.py for enhanced node and edge generation")
    else:
        _HAS_GEMINI_CREATE_FUNCS = True
except Exception as e:  # noqa: E722
    print(f"Failed to import genkg.py functions: {e}")
    # Legacy fallback for old gemini functions
    try:  # pragma: no cover - legacy fallback
        from gemini_create_nodes import gemini_create_nodes as legacy_gemini_create_nodes
        from gemini_create_edges import create_edges_by_gemini as legacy_create_edges_by_gemini
        gemini_create_nodes = legacy_gemini_create_nodes
        create_edges_by_gemini = legacy_create_edges_by_gemini
        _HAS_GEMINI_CREATE_FUNCS = True
        print("Using legacy gemini functions")
    except Exception:  # noqa: E722
        _HAS_GEMINI_CREATE_FUNCS = False
        print("No enhanced node/edge generation functions available")


@dataclass
class GraphRAG:
    working_dir: str = field(
        default_factory=lambda: f"./nano_graphrag_cache_{datetime.now().strftime('%Y-%m-%d-%H:%M:%S')}"
    )
    # graph mode
    enable_local: bool = True
    enable_naive_rag: bool = True

    # text chunking
    chunk_func: Callable[
        [
            list[list[int]],
            List[str],
            tiktoken.Encoding,
            Optional[int],
            Optional[int],
        ],
        List[Dict[str, Union[str, int]]],
    ] = chunking_large_context  # Default to optimized chunking for Gemini 2.5 Flash Lite
    chunk_token_size: int = 32768  # Large chunks for 1M context window
    chunk_overlap_token_size: int = 2048  # Proportional overlap for large chunks
    # Tokenizer model reference (legacy config - now uses cl100k_base encoding directly)
    tiktoken_model_name: str = "cl100k_base"  # Direct encoding for compatibility

    # entity extraction
    entity_extract_max_gleaning: int = 0  # Reduced for speed - Gemini 2.5 is more capable
    entity_summary_to_max_tokens: int = 2000  # Increased for better summaries

    # graph clustering
    graph_cluster_algorithm: str = "leiden"
    max_graph_cluster_size: int = 10
    graph_cluster_seed: int = 0xDEADBEEF

    # node embedding
    node_embedding_algorithm: str = "node2vec"
    node2vec_params: dict = field(
        default_factory=lambda: {
            "dimensions": 1536,
            "num_walks": 10,
            "walk_length": 40,
            "num_walks": 10,
            "window_size": 2,
            "iterations": 3,
            "random_seed": 3,
        }
    )

    # community reports
    special_community_report_llm_kwargs: dict = field(
        default_factory=lambda: {"response_format": {"type": "json_object"}}
    )

    # text embedding
    embedding_func: EmbeddingFunc = field(default_factory=lambda: gemini_embedding)
    embedding_batch_num: int = 32
    embedding_func_max_async: int = 16
    query_better_than_threshold: float = 0.2

    # LLM
    using_azure_openai: bool = False
    using_gemini: bool = True
    best_model_func: callable = gemini_2_5_flash_complete
    best_model_max_token_size: int = 32768
    best_model_max_async: int = 16
    cheap_model_func: callable = gemini_2_5_flash_complete
    cheap_model_max_token_size: int = 32768
    cheap_model_max_async: int = 16

    # entity extraction
    entity_extraction_func: callable = extract_entities

    # storage
    key_string_value_json_storage_cls: Type[BaseKVStorage] = JsonKVStorage
    vector_db_storage_cls: Type[BaseVectorStorage] = NanoVectorDBStorage
    vector_db_storage_cls_kwargs: dict = field(default_factory=dict)
    graph_storage_cls: Type[BaseGraphStorage] = NetworkXStorage
    enable_llm_cache: bool = True

    # GenKG integration
    use_genkg_extraction: bool = False
    genkg_node_limit: int = 25
    genkg_llm_provider: str = "gemini"  
    genkg_model_name: str = "gemini-2.5-flash"
    genkg_create_visualization: bool = False
    genkg_output_path: Optional[str] = None

    # extension
    always_create_working_dir: bool = True
    addon_params: dict = field(default_factory=dict)
    convert_response_to_json_func: callable = convert_response_to_json

    def __post_init__(self):
        _print_config = ",\n  ".join([f"{k} = {v}" for k, v in asdict(self).items()])
        logger.debug(f"GraphRAG init with param:\n\n  {_print_config}\n")

        if self.using_azure_openai:
            # If there's no OpenAI API key, use Azure OpenAI
            if self.best_model_func == gpt_4o_complete:
                self.best_model_func = azure_gpt_4o_complete
            if self.cheap_model_func == gpt_4o_mini_complete:
                self.cheap_model_func = azure_gpt_4o_mini_complete
            if self.embedding_func == openai_embedding:
                self.embedding_func = azure_openai_embedding
            logger.info(
                "Switched the default openai funcs to Azure OpenAI if you didn't set any of it"
            )
            
        # Override with Gemini if using_gemini is True
        if self.using_gemini:
            logger.info("Using Gemini for LLM and embeddings")
            self.best_model_func = gemini_2_5_flash_complete
            self.cheap_model_func = gemini_2_5_flash_complete  
            self.embedding_func = gemini_embedding

        if not os.path.exists(self.working_dir) and self.always_create_working_dir:
            logger.info(f"Creating working directory {self.working_dir}")
            os.makedirs(self.working_dir)

        self.full_docs = self.key_string_value_json_storage_cls(
            namespace="full_docs", global_config=asdict(self)
        )

        self.text_chunks = self.key_string_value_json_storage_cls(
            namespace="text_chunks", global_config=asdict(self)
        )

        self.llm_response_cache = (
            self.key_string_value_json_storage_cls(
                namespace="llm_response_cache", global_config=asdict(self)
            )
            if self.enable_llm_cache
            else None
        )

        self.community_reports = self.key_string_value_json_storage_cls(
            namespace="community_reports", global_config=asdict(self)
        )
        self.chunk_entity_relation_graph = self.graph_storage_cls(
            namespace="chunk_entity_relation", global_config=asdict(self)
        )

        self.embedding_func = limit_async_func_call(self.embedding_func_max_async)(
            self.embedding_func
        )
        self.entities_vdb = (
            self.vector_db_storage_cls(
                namespace="entities",
                global_config=asdict(self),
                embedding_func=self.embedding_func,
                meta_fields={"entity_name"},
            )
            if self.enable_local
            else None
        )
        self.chunks_vdb = (
            self.vector_db_storage_cls(
                namespace="chunks",
                global_config=asdict(self),
                embedding_func=self.embedding_func,
            )
            if self.enable_naive_rag
            else None
        )

        # Configure GenKG if enabled
        if self.use_genkg_extraction:
            logger.info("Using GenKG for entity extraction")
            self.entity_extraction_func = extract_entities_genkg
            
            # Set default output path if not provided and visualization is enabled
            if self.genkg_create_visualization and not self.genkg_output_path:
                self.genkg_output_path = os.path.join(self.working_dir, "output.html")
                
        self.best_model_func = limit_async_func_call(self.best_model_max_async)(
            partial(self.best_model_func, hashing_kv=self.llm_response_cache)
        )
        self.cheap_model_func = limit_async_func_call(self.cheap_model_max_async)(
            partial(self.cheap_model_func, hashing_kv=self.llm_response_cache)
        )

    def insert(self, string_or_strings):
        loop = always_get_an_event_loop()
        return loop.run_until_complete(self.ainsert(string_or_strings))

    def query(self, query: str, param: QueryParam = QueryParam()):
        loop = always_get_an_event_loop()
        return loop.run_until_complete(self.aquery(query, param))

    async def aquery(self, query: str, param: QueryParam = QueryParam()):
        if param.mode == "local" and not self.enable_local:
            raise ValueError("enable_local is False, cannot query in local mode")
        if param.mode == "naive" and not self.enable_naive_rag:
            raise ValueError("enable_naive_rag is False, cannot query in naive mode")
        if param.mode == "local":
            response = await local_query(
                query,
                self.chunk_entity_relation_graph,
                self.entities_vdb,
                self.community_reports,
                self.text_chunks,
                param,
                asdict(self),
            )
        elif param.mode == "global":
            response = await global_query(
                query,
                self.chunk_entity_relation_graph,
                self.entities_vdb,
                self.community_reports,
                self.text_chunks,
                param,
                asdict(self),
            )
        elif param.mode == "naive":
            response = await naive_query(
                query,
                self.chunks_vdb,
                self.text_chunks,
                param,
                asdict(self),
            )
        else:
            raise ValueError(f"Unknown mode {param.mode}")
        await self._query_done()
        return response

    async def ainsert(self, string_or_strings):
        await self._insert_start()
        try:
            if isinstance(string_or_strings, str):
                string_or_strings = [string_or_strings]
            # ---------- new docs
            new_docs = {
                compute_mdhash_id(c.strip(), prefix="doc-"): {"content": c.strip()}
                for c in string_or_strings
            }
            _add_doc_keys = await self.full_docs.filter_keys(list(new_docs.keys()))
            new_docs = {k: v for k, v in new_docs.items() if k in _add_doc_keys}
            if not len(new_docs):
                logger.warning(f"All docs are already in the storage")
                return
            logger.info(f"[New Docs] inserting {len(new_docs)} docs")

            # ---------- chunking

            inserting_chunks = get_chunks(
                new_docs=new_docs,
                chunk_func=self.chunk_func,
                overlap_token_size=self.chunk_overlap_token_size,
                max_token_size=self.chunk_token_size,
            )

            _add_chunk_keys = await self.text_chunks.filter_keys(
                list(inserting_chunks.keys())
            )
            inserting_chunks = {
                k: v for k, v in inserting_chunks.items() if k in _add_chunk_keys
            }
            if not len(inserting_chunks):
                logger.warning(f"All chunks are already in the storage")
                return
            logger.info(f"[New Chunks] inserting {len(inserting_chunks)} chunks")
            if self.enable_naive_rag:
                logger.info("Insert chunks for naive RAG")
                await self.chunks_vdb.upsert(inserting_chunks)

            # TODO: no incremental update for communities now, so just drop all
            await self.community_reports.drop()

            # ---------- Knowledge Graph Construction ----------
            # Option A: Gemini-based concept + edge suggestion
            if self.use_gemini_extraction:
                if not _HAS_GEMINI_CREATE_FUNCS:
                    logger.warning("Gemini extraction requested but helper functions not importable. Skipping.")
                else:
                    logger.info("[Gemini Extraction] generating nodes & edges ...")
                    # Build papers dict (doc_id -> full content)
                    papers_dict = {doc_id: dp["content"] for doc_id, dp in new_docs.items()}
                    # Gather nodes per document
                    all_nodes = []  # list of (concept, source_doc_id)
                    for doc_id, dp in new_docs.items():
                        try:
                            nodes_for_doc = gemini_create_nodes(
                                dp["content"], self.gemini_node_limit, doc_id
                            ) if gemini_create_nodes else []
                        except Exception as e:  # pragma: no cover - external API failures
                            logger.warning(f"Gemini node creation failed for {doc_id}: {e}")
                            nodes_for_doc = []
                        all_nodes.extend(nodes_for_doc)
                    # Deduplicate & merge sources
                    merged_nodes: dict[str, dict] = {}
                    for concept, source in all_nodes:
                        concept_up = concept.upper().strip()
                        if not concept_up:
                            continue
                        if concept_up not in merged_nodes:
                            merged_nodes[concept_up] = {
                                "entity_type": '"GEMINI"',
                                "description": concept.strip(),
                                "source_id": source,
                            }
                        else:
                            # merge sources / descriptions
                            if concept.strip() not in merged_nodes[concept_up]["description"].split(GRAPH_FIELD_SEP):
                                merged_nodes[concept_up]["description"] += GRAPH_FIELD_SEP + concept.strip()
                            if source not in merged_nodes[concept_up]["source_id"].split(GRAPH_FIELD_SEP):
                                merged_nodes[concept_up]["source_id"] += GRAPH_FIELD_SEP + source
                    # Upsert nodes
                    for node_id, node_data in merged_nodes.items():
                        try:
                            await self.chunk_entity_relation_graph.upsert_node(node_id, node_data)
                        except Exception as e:  # pragma: no cover
                            logger.warning(f"Failed to upsert Gemini node {node_id}: {e}")
                    # Create edges via Gemini using (concept, source) tuples
                    edges = []
                    try:
                        edges = create_edges_by_gemini(
                            list(merged_nodes.items()),  # but our items are node_id->data; adapt to expected format
                            papers_dict,
                        ) if create_edges_by_gemini else []
                    except Exception as e:  # pragma: no cover
                        logger.warning(f"Gemini edge creation failed: {e}")
                        edges = []
                    # edges expected: [ ((node1, source1),(node2, source2), {attrs}) ]
                    for edge in edges:
                        try:
                            (n1_text, _src1), (n2_text, _src2), attrs = edge
                            n1 = n1_text.upper().strip()
                            n2 = n2_text.upper().strip()
                            if not n1 or not n2 or n1 == n2:
                                continue
                            # Ensure nodes exist (might not if Gemini referenced unseen concept)
                            for nid, raw in [(n1, n1_text), (n2, n2_text)]:
                                if not await self.chunk_entity_relation_graph.has_node(nid):
                                    await self.chunk_entity_relation_graph.upsert_node(
                                        nid,
                                        node_data={
                                            "entity_type": '"GEMINI"',
                                            "description": raw,
                                            "source_id": "",
                                        },
                                    )
                            weight = float(attrs.get("weight", 1.0)) if isinstance(attrs, dict) else 1.0
                            relation = attrs.get("relation", "related_to") if isinstance(attrs, dict) else "related_to"
                            await self.chunk_entity_relation_graph.upsert_edge(
                                n1,
                                n2,
                                edge_data={
                                    "weight": weight,
                                    "description": relation,
                                    "source_id": merged_nodes.get(n1, {}).get("source_id", "") + GRAPH_FIELD_SEP + merged_nodes.get(n2, {}).get("source_id", ""),
                                    "order": 1,
                                },
                            )
                        except Exception as e:  # pragma: no cover
                            logger.warning(f"Failed to upsert Gemini edge: {e}")
                    logger.info(
                        f"Gemini KG added {len(merged_nodes)} nodes and {len(edges)} edges"
                    )
                    # If not combining, skip default extraction
                    if not self.gemini_combine_with_llm_extraction:
                        maybe_new_kg = self.chunk_entity_relation_graph
                    else:
                        logger.info("[Entity Extraction] (combined) ...")
                        maybe_new_kg = await self.entity_extraction_func(
                            inserting_chunks,
                            knwoledge_graph_inst=self.chunk_entity_relation_graph,
                            entity_vdb=self.entities_vdb,
                            global_config=asdict(self),
                        )
                        if maybe_new_kg is None:
                            logger.warning("LLM entity extraction found no additional entities (Gemini results kept)")
                            maybe_new_kg = self.chunk_entity_relation_graph
                    self.chunk_entity_relation_graph = maybe_new_kg
            else:
                # Default path only
                logger.info("[Entity Extraction] ...")
                maybe_new_kg = await self.entity_extraction_func(
                    inserting_chunks,
                    knwoledge_graph_inst=self.chunk_entity_relation_graph,
                    entity_vdb=self.entities_vdb,
                    global_config=asdict(self),
                )
                if maybe_new_kg is None:
                    logger.warning("No new entities found")
                    return
                self.chunk_entity_relation_graph = maybe_new_kg
            # ---------- update clusterings of graph
            logger.info("[Community Report]...")
            await self.chunk_entity_relation_graph.clustering(
                self.graph_cluster_algorithm
            )
            await generate_community_report(
                self.community_reports, self.chunk_entity_relation_graph, asdict(self)
            )

            # ---------- generate GenKG visualizations if enabled
            if self.use_genkg_extraction and self.genkg_create_visualization:
                await self._generate_genkg_visualizations(inserting_chunks, new_docs)

            # ---------- commit upsertings and indexing
            await self.full_docs.upsert(new_docs)
            await self.text_chunks.upsert(inserting_chunks)
        finally:
            await self._insert_done()

    async def _insert_start(self):
        tasks = []
        for storage_inst in [
            self.chunk_entity_relation_graph,
        ]:
            if storage_inst is None:
                continue
            tasks.append(cast(StorageNameSpace, storage_inst).index_start_callback())
        await asyncio.gather(*tasks)

    async def _insert_done(self):
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

    async def _generate_genkg_visualizations(self, inserting_chunks, new_docs):
        """Generate GenKG visualizations using the same normalized nodes as nano-graphrag"""
        # Retrieve stored GenKG visualization data from file
        import json
        import os
        viz_data_path = os.path.join(self.working_dir, "_genkg_viz_data.json")
        if not os.path.exists(viz_data_path):
            logger.warning("No GenKG visualization data found")
            return
            
        try:
            with open(viz_data_path, 'r', encoding='utf-8') as f:
                genkg_viz_data = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load GenKG visualization data: {e}")
            return
        
        try:
            # Import genkg locally
            import sys
            import os
            possible_genkg_paths = [
                os.path.join(self.working_dir, "..", "nano-graphrag", "genkg.py"),
                os.path.join(self.working_dir, "..", "..", "nano-graphrag", "genkg.py"),
                os.path.join(os.path.dirname(__file__), "..", "genkg.py"),
            ]
            
            genkg_found = False
            for genkg_path in possible_genkg_paths:
                if os.path.exists(genkg_path):
                    genkg_dir = os.path.dirname(genkg_path)
                    if genkg_dir not in sys.path:
                        sys.path.insert(0, genkg_dir)
                    genkg_found = True
                    break
                    
            if not genkg_found:
                logger.warning("GenKG not found for visualization")
                return
                
            from .genkg import GenerateKG
            logger.info("[GenKG Visualization] Creating output files from stored entities...")
            
            # Use the ALREADY PROCESSED data from nano-graphrag - no duplicate processing!
            nodes_with_source = genkg_viz_data["nodes_with_source"]  # These are normalized nodes from nano-graphrag
            edges_data = genkg_viz_data.get("edges", [])  # These are the edges already created
            
            # Create NetworkX graph directly from nano-graphrag processed data
            import networkx as nx
            knowledge_graph = nx.Graph()
            
            # Generate paper colors
            all_sources = set(source for _, source in nodes_with_source)
            distinctive_colors = ["#4285F4", "#EA4335", "#FBBC05", "#34A853", "#FF9900", "#146EB4"]
            paper_colors = {source: distinctive_colors[i % len(distinctive_colors)] for i, source in enumerate(all_sources)}
            
            # Add nodes with attributes
            for node_text, source in nodes_with_source:
                knowledge_graph.add_node(node_text, 
                                       source=source, 
                                       color=paper_colors.get(source, "#808080"), 
                                       title=f"Source: {source}")
            
            # Add edges from stored edge data
            for edge_data in edges_data:
                src_id = edge_data.get("src_id")
                tgt_id = edge_data.get("tgt_id") 
                weight = edge_data.get("weight", 1.0)
                relation = edge_data.get("description", "related_to")
                
                if src_id and tgt_id and src_id in knowledge_graph.nodes and tgt_id in knowledge_graph.nodes:
                    knowledge_graph.add_edge(src_id, tgt_id, weight=weight, relation=relation)
            
            # Initialize GenKG only for visualization methods
            genkg = GenerateKG(llm_provider=self.genkg_llm_provider, model_name=self.genkg_model_name)
            
            # Export to output.dashkg.json with normalized names
            output_json_path = os.path.join(self.working_dir, "output.dashkg.json")
            genkg.export_graph_to_dashkg_json(knowledge_graph, output_json_path)
            
            # Create HTML visualization
            html_path = os.path.join(self.working_dir, "output.html")
            genkg.advanced_graph_to_html(knowledge_graph, html_path, display=False)
            
            logger.info(f"[GenKG Visualization] Files generated: {html_path}")
            logger.info(f"Generated visualizations with {len(knowledge_graph.nodes)} nodes, {len(knowledge_graph.edges)} edges")
            
        except Exception as e:
            logger.error(f"Failed to generate GenKG visualizations: {e}")
            import traceback
            traceback.print_exc()

    async def _query_done(self):
        tasks = []
        for storage_inst in [self.llm_response_cache]:
            if storage_inst is None:
                continue
            tasks.append(cast(StorageNameSpace, storage_inst).index_done_callback())
        await asyncio.gather(*tasks)
