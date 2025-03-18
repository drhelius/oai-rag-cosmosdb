import os
import time
import streamlit as st
import pandas as pd
import altair as alt
from streamlit_extras.switch_page_button import switch_page
import plotly.express as px
from datetime import datetime

from app_config import DEFAULT_SETTINGS, UI_CONFIG, METRICS, SEARCH_CONFIG, CHAT_CONFIG
from app_utils import save_uploaded_file, validate_file, clear_temp_files, format_time
from document_processor import DocumentProcessor
from cosmos_db_client import CosmosDBClient
from search_service import SearchService
from chat_service import ChatService

# Initialize app
def init_app():
    # Page configuration
    st.set_page_config(
        page_title="CosmosDB RAG Demo",
        page_icon="📚",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize session state variables if they don't exist
    if "process_history" not in st.session_state:
        st.session_state.process_history = []
    if "settings" not in st.session_state:
        st.session_state.settings = DEFAULT_SETTINGS.copy()
    if "metrics" not in st.session_state:
        st.session_state.metrics = {k: 0 for k in METRICS.keys()}
    if "search_history" not in st.session_state:
        st.session_state.search_history = []
    if "search_settings" not in st.session_state:
        st.session_state.search_settings = SEARCH_CONFIG.copy()
    if "compare_mode" not in st.session_state:
        st.session_state.compare_mode = False
    
    # Add chat-specific session state
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []
    if "chat_settings" not in st.session_state:
        st.session_state.chat_settings = CHAT_CONFIG.copy()
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "chat_metrics" not in st.session_state:
        st.session_state.chat_metrics = {"total_queries": 0, "avg_response_time": 0}
    if "show_chat_settings" not in st.session_state:
        st.session_state.show_chat_settings = False
    if "show_context" not in st.session_state:
        st.session_state.show_context = False
    
    # App title and description
    st.title("📚 CosmosDB RAG Demo")
    st.markdown("""
    This demo showcases the power of Azure CosmosDB with its hybrid search feature for RAG systems.
    Index your documents, search through them, and chat with an AI assistant powered by CosmosDB vector search.
    """)

# Display configuration panel in sidebar
def display_configuration(column):
    with column:
        st.subheader("Configuration")
        with st.expander("Storage Settings", expanded=False):
            st.session_state.settings["blob_container_name"] = st.text_input(
                "Blob Container Name", 
                value=st.session_state.settings["blob_container_name"]
            )
            st.session_state.settings["cosmos_db_name"] = st.text_input(
                "CosmosDB Database", 
                value=st.session_state.settings["cosmos_db_name"]
            )
            st.session_state.settings["cosmos_container_name"] = st.text_input(
                "CosmosDB Container", 
                value=st.session_state.settings["cosmos_container_name"]
            )
        
        with st.expander("Processing Settings", expanded=False):
            st.session_state.settings["tokens_per_chunk"] = st.slider(
                "Chunk Size (tokens)", 
                min_value=100, 
                max_value=1000, 
                value=st.session_state.settings["tokens_per_chunk"],
                step=50
            )
            st.session_state.settings["overlap_tokens"] = st.slider(
                "Overlap Size (tokens)", 
                min_value=0, 
                max_value=300, 
                value=st.session_state.settings["overlap_tokens"],
                step=10
            )
            st.session_state.settings["embedding_model"] = st.selectbox(
                "Embedding Model", 
                ["text_embedding_3_small", "text_embedding_3_large"], 
                index=0 if st.session_state.settings["embedding_model"] == "text_embedding_3_small" else 1,
                key="config_embedding_model"  # Added unique key
            )

# Display metrics panel
def display_metrics(column):
    with column:
        st.subheader("Metrics")
        metrics_cols = st.columns(2)
        
        # Display metrics in two columns
        metrics_keys = list(METRICS.keys())
        for i, key in enumerate(metrics_keys[:4]):  # First 4 metrics
            with metrics_cols[i % 2]:
                if key in ["processing_time", "avg_embedding_time"]:
                    value = format_time(st.session_state.metrics.get(key, 0))
                else:
                    value = st.session_state.metrics.get(key, 0)
                st.metric(METRICS[key], value)
        
        # Time metrics
        for i, key in enumerate(metrics_keys[4:]):  # Last 2 metrics (time metrics)
            with metrics_cols[i % 2]:
                value = format_time(st.session_state.metrics.get(key, 0))
                st.metric(METRICS[key], value)

# Process a single document
def process_single_document(file, file_index, total_files, processor, process_container, batch_metrics):
    with process_container:
        st.subheader(f"Processing {file.name} ({file_index+1}/{total_files})")
        col1, col2 = st.columns([3, 1])
        
        with col1:
            progress_bar = st.progress(0)
            status_text = st.empty()
            details = st.empty()
        
        with col2:
            st.markdown("**Document Info:**")
            doc_info = st.empty()
        
        try:
            # Save uploaded file to temp directory
            file_path = save_uploaded_file(file)
            
            # Process the document
            doc_id, doc_metrics = processor.process_document(
                file_path, 
                progress_bar=progress_bar, 
                status_text=status_text
            )
            
            # Display document metrics
            doc_info.markdown(f"""
            - **Pages:** {doc_metrics['pages']}
            - **Chunks:** {doc_metrics['chunks']}
            - **CosmosDB Items:** {doc_metrics['cosmos_items']}
            - **Processing Time:** {format_time(doc_metrics['processing_time'])}
            - **Document ID:** `{doc_metrics['document_id']}`
            """)
            
            # Add success message with details
            details.success(f"Successfully processed and indexed {file.name}")
            
            # Add to batch metrics
            batch_metrics["documents"].append({
                "filename": file.name,
                "document_id": doc_metrics["document_id"],
                "pages": doc_metrics["pages"],
                "chunks": doc_metrics["chunks"],
                "processing_time": doc_metrics["processing_time"],
                "cosmos_items": doc_metrics["cosmos_items"]
            })
            
            # Add to session state history
            st.session_state.process_history.append({
                "timestamp": datetime.now().strftime("%H:%M:%S"),
                "filename": file.name,
                "document_id": doc_metrics["document_id"],
                "pages": doc_metrics["pages"],
                "chunks": doc_metrics["chunks"],
                "processing_time": doc_metrics["processing_time"],
                "cosmos_items": doc_metrics["cosmos_items"]
            })
            
        except Exception as e:
            st.error(f"Error processing {file.name}: {str(e)}")
            return False
            
        return True

# Process batch of documents
def process_document_batch(uploaded_files):
    num_files = len(uploaded_files)
    max_files = UI_CONFIG["max_files"]
    
    if num_files > max_files:
        st.warning(f"Too many files selected. Maximum allowed is {max_files}.")
        return
        
    st.info(f"{num_files} file{'s' if num_files > 1 else ''} selected for processing.")
    
    if st.button("Process Documents", key="process_btn", type="primary"):
        process_container = st.container()
        
        # Initialize document processor
        processor = DocumentProcessor(st.session_state.settings)
        
        # Keep track of this batch's metrics for display
        batch_metrics = {
            "start_time": time.time(),
            "documents": []
        }
        
        # Process each file
        for i, file in enumerate(uploaded_files):
            process_single_document(file, i, num_files, processor, process_container, batch_metrics)
        
        # Update session state metrics
        batch_processing_time = time.time() - batch_metrics["start_time"]
        st.session_state.metrics["document_count"] += len(batch_metrics["documents"])
        st.session_state.metrics["total_pages"] += sum(doc["pages"] for doc in batch_metrics["documents"])
        st.session_state.metrics["total_chunks"] += sum(doc["chunks"] for doc in batch_metrics["documents"])
        st.session_state.metrics["cosmos_db_items"] += sum(doc["cosmos_items"] for doc in batch_metrics["documents"])
        st.session_state.metrics["processing_time"] += batch_processing_time
        
        # Calculate average embedding time
        if processor.metrics["avg_embedding_time"] > 0:
            st.session_state.metrics["avg_embedding_time"] = processor.metrics["avg_embedding_time"]
        
        # Summary of batch processing
        st.subheader("Batch Processing Complete")
        st.success(f"Successfully processed {len(batch_metrics['documents'])} documents in {format_time(batch_processing_time)}")
        
        # Clear temporary files
        clear_temp_files()
        
        # Force a rerun to update the metrics display
        st.rerun()

# Display document upload section
def display_document_upload(column):
    with column:
        st.subheader("Upload Documents")
        st.markdown("Upload PDF documents to be processed and indexed in CosmosDB.")
        
        # File uploader
        uploaded_files = st.file_uploader(
            "Drag and drop PDF files here",
            type=UI_CONFIG["allowed_file_types"],
            accept_multiple_files=True,
            key="document_uploader"
        )
        
        # Process files button
        if uploaded_files:
            process_document_batch(uploaded_files)

# Display processing history and visualizations
def display_processing_history(column):
    with column:
        if st.session_state.process_history:
            with st.expander("Processing History", expanded=True):
                history_df = pd.DataFrame(st.session_state.process_history)
                
                if not history_df.empty:
                    # Reformat the time columns
                    if 'processing_time' in history_df.columns:
                        history_df['processing_time'] = history_df['processing_time'].apply(lambda x: f"{x:.2f}s")
                    
                    # Display as a table
                    st.dataframe(
                        history_df,
                        column_config={
                            "timestamp": "Time",
                            "filename": "Document",
                            "document_id": "ID",
                            "pages": "Pages",
                            "chunks": "Chunks",
                            "processing_time": "Processing Time",
                            "cosmos_items": "CosmosDB Items"
                        },
                        use_container_width=True,
                        hide_index=True
                    )
                    
                    display_visualizations(history_df)
                    
                    # Clear history button at the bottom
                    if st.button("Clear History"):
                        st.session_state.process_history = []
                        st.rerun()

# Display visualizations of processing history
def display_visualizations(history_df):
    # Visualization section below the history table
    st.subheader("Document Processing Visualizations")
    
    # Plot chunk distribution
    if 'chunks' in history_df.columns:
        st.subheader("Chunks per Document")
        fig = px.bar(history_df, x='filename', y='chunks', color='chunks',
                 labels={'chunks': 'Number of Chunks', 'filename': 'Document'},
                 height=250)
        fig.update_layout(margin=dict(l=20, r=20, t=30, b=50))
        st.plotly_chart(fig, use_container_width=True)
    
    # Plot processing time
    if 'processing_time' in history_df.columns:
        # Convert processing_time back to numeric for plotting
        if isinstance(history_df['processing_time'][0], str):
            history_df['processing_time_num'] = history_df['processing_time'].str.replace('s', '').astype(float)
        else:
            history_df['processing_time_num'] = history_df['processing_time']
            
        st.subheader("Processing Time")
        fig = px.bar(history_df, x='filename', y='processing_time_num', color='processing_time_num',
                 labels={'processing_time_num': 'Seconds', 'filename': 'Document'},
                 height=250)
        fig.update_layout(margin=dict(l=20, r=20, t=30, b=50))
        st.plotly_chart(fig, use_container_width=True)

# Document search tab
def display_search_tab():
    st.title("🔍 Search Documents")
    st.markdown("""
    Search through your indexed documents using different search methods: 
    full-text search, vector similarity search, or hybrid search.
    """)
    
    # Initialize CosmosDB client for search
    cosmos_db = CosmosDBClient(
        st.session_state.settings["cosmos_db_name"],
        st.session_state.settings["cosmos_container_name"]
    )
    
    # Initialize search service
    search_service = SearchService(
        cosmos_db,
        embedding_model=st.session_state.search_settings["embedding_model"]
    )
    
    # Create columns for search interface
    col1, col2 = st.columns([2, 1])
    
    with col2:
        # Search settings
        with st.expander("Search Settings", expanded=True):
            st.session_state.search_settings["top_k"] = st.slider(
                "Number of results", 
                min_value=1, 
                max_value=20, 
                value=st.session_state.search_settings["top_k"],
                key="search_top_k"  # Added unique key
            )
            
            st.session_state.search_settings["min_similarity"] = st.slider(
                "Minimum similarity score", 
                min_value=0.0, 
                max_value=1.0, 
                value=st.session_state.search_settings["min_similarity"],
                step=0.05,
                key="search_min_similarity"  # Added unique key
            )
            
            st.session_state.search_settings["embedding_model"] = st.selectbox(
                "Embedding Model", 
                ["text_embedding_3_small", "text_embedding_3_large"], 
                index=0 if st.session_state.search_settings["embedding_model"] == "text_embedding_3_small" else 1,
                key="search_embedding_model"  # Added unique key
            )
            
            st.session_state.compare_mode = st.toggle(
                "Compare Search Methods", 
                value=st.session_state.compare_mode,
                key="search_compare_mode"  # Added unique key
            )
            
        # Source document filter
        with st.expander("Filter by Document", expanded=True):
            # Get unique source documents
            try:
                source_docs = cosmos_db.get_unique_document_sources()
                if source_docs:
                    selected_sources = st.multiselect(
                        "Filter by source document",
                        options=source_docs,
                        default=[],
                        key="search_doc_filter"  # Added unique key
                    )
                    if selected_sources:
                        st.info(f"Filtering by {len(selected_sources)} documents")
                else:
                    st.info("No documents indexed yet")
            except Exception as e:
                st.error(f"Error loading document sources: {e}")
        
        # Search metrics display
        with st.expander("Search Statistics", expanded=True):
            if st.session_state.search_history:
                latest_search = st.session_state.search_history[-1]
                col1_metrics, col2_metrics = st.columns(2)
                
                with col1_metrics:
                    st.metric("Last Query Time", f"{latest_search['metrics']['query_time_ms']:.2f}ms")
                    st.metric("Results Found", latest_search['metrics']['results_count'])
                
                with col2_metrics:
                    st.metric("Search Type", latest_search['metrics']['search_type'])
                    if "embedding_dimensions" in latest_search['metrics']:
                        st.metric("Embedding Size", latest_search['metrics']['embedding_dimensions'])
                        
                if len(st.session_state.search_history) > 1:
                    # Show comparison chart of query times
                    search_df = pd.DataFrame([
                        {
                            "query": s["query"][:15] + "..." if len(s["query"]) > 15 else s["query"],
                            "time_ms": s["metrics"]["query_time_ms"],
                            "type": s["metrics"]["search_type"],
                            "count": s["metrics"]["results_count"]
                        }
                        for s in st.session_state.search_history[-5:]  # Last 5 searches
                    ])
                    
                    st.subheader("Recent Search Performance")
                    fig = px.bar(search_df, x="query", y="time_ms", color="type",
                             labels={"time_ms": "Query Time (ms)", "query": "Query"},
                             height=200)
                    fig.update_layout(margin=dict(l=5, r=5, t=5, b=5))
                    st.plotly_chart(fig, use_container_width=True)
    
    with col1:
        # Search input
        query = st.text_input("Search Query", key="search_query")
        
        # Method selector and search button row
        col_method, col_button = st.columns([3, 1])
        
        with col_method:
            if not st.session_state.compare_mode:
                search_method = st.radio(
                    "Search Method",
                    options=["Full-Text Search", "Vector Search", "Hybrid Search"],
                    horizontal=True,
                    key="search_method"  # Added unique key
                )
        
        with col_button:
            search_clicked = st.button(
                "🔍 Search", 
                type="primary", 
                use_container_width=True,
                disabled=not query,
                key="search_button"  # Added unique key
            )
        
        # Perform search when button clicked
        if search_clicked and query:
            with st.spinner("Searching..."):
                if st.session_state.compare_mode:
                    # Compare all search methods
                    comparison_results = search_service.compare_search_methods(
                        query,
                        top_k=st.session_state.search_settings["top_k"],
                        min_similarity=st.session_state.search_settings["min_similarity"]
                    )
                    
                    # Calculate overlap metrics
                    overlap_vector_text = search_service.calculate_result_overlap(
                        comparison_results["vector"]["results"],
                        comparison_results["text"]["results"]
                    )
                    
                    overlap_vector_hybrid = search_service.calculate_result_overlap(
                        comparison_results["vector"]["results"],
                        comparison_results["hybrid"]["results"]
                    )
                    
                    overlap_text_hybrid = search_service.calculate_result_overlap(
                        comparison_results["text"]["results"],
                        comparison_results["hybrid"]["results"]
                    )
                    
                    # Add to search history
                    for method in ["vector", "text", "hybrid"]:
                        st.session_state.search_history.append({
                            "query": query,
                            "method": method,
                            "results": comparison_results[method]["results"],
                            "metrics": comparison_results[method]["metrics"],
                            "timestamp": datetime.now().strftime("%H:%M:%S")
                        })
                    
                    # Display results in tabs
                    search_tabs = st.tabs(["Vector Search", "Full-Text Search", "Hybrid Search", "Comparison"])
                    
                    # Vector Search Results
                    with search_tabs[0]:
                        display_search_results(
                            comparison_results["vector"]["results"],
                            comparison_results["vector"]["metrics"]
                        )
                    
                    # Text Search Results
                    with search_tabs[1]:
                        display_search_results(
                            comparison_results["text"]["results"],
                            comparison_results["text"]["metrics"]
                        )
                    
                    # Hybrid Search Results
                    with search_tabs[2]:
                        display_search_results(
                            comparison_results["hybrid"]["results"],
                            comparison_results["hybrid"]["metrics"]
                        )
                    
                    # Comparison tab
                    with search_tabs[3]:
                        st.subheader("Search Method Comparison")
                        
                        # Performance metrics comparison
                        metrics_df = pd.DataFrame([
                            {
                                "Method": "Vector Search",
                                "Query Time (ms)": comparison_results["vector"]["metrics"]["query_time_ms"],
                                "Results Count": comparison_results["vector"]["metrics"]["results_count"]
                            },
                            {
                                "Method": "Text Search",
                                "Query Time (ms)": comparison_results["text"]["metrics"]["query_time_ms"],
                                "Results Count": comparison_results["text"]["metrics"]["results_count"]
                            },
                            {
                                "Method": "Hybrid Search",
                                "Query Time (ms)": comparison_results["hybrid"]["metrics"]["query_time_ms"],
                                "Results Count": comparison_results["hybrid"]["metrics"]["results_count"]
                            }
                        ])
                        
                        # Performance chart
                        st.subheader("Performance Comparison")
                        fig = px.bar(
                            metrics_df, 
                            x="Method", 
                            y="Query Time (ms)",
                            color="Method",
                            height=300
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Results overlap metrics
                        st.subheader("Results Overlap Analysis")
                        col_overlap1, col_overlap2, col_overlap3 = st.columns(3)
                        
                        with col_overlap1:
                            st.metric("Vector-Text Overlap", f"{overlap_vector_text['overlap_percentage']:.1f}%")
                            st.caption(f"Jaccard Similarity: {overlap_vector_text['jaccard_similarity']:.3f}")
                            
                        with col_overlap2:
                            st.metric("Vector-Hybrid Overlap", f"{overlap_vector_hybrid['overlap_percentage']:.1f}%")
                            st.caption(f"Jaccard Similarity: {overlap_vector_hybrid['jaccard_similarity']:.3f}")
                            
                        with col_overlap3:
                            st.metric("Text-Hybrid Overlap", f"{overlap_text_hybrid['overlap_percentage']:.1f}%")
                            st.caption(f"Jaccard Similarity: {overlap_text_hybrid['jaccard_similarity']:.3f}")
                        
                        # Common results across all methods
                        vector_ids = set(doc["id"] for doc in comparison_results["vector"]["results"])
                        text_ids = set(doc["id"] for doc in comparison_results["text"]["results"])
                        hybrid_ids = set(doc["id"] for doc in comparison_results["hybrid"]["results"])
                        
                        common_ids = vector_ids.intersection(text_ids).intersection(hybrid_ids)
                        
                        st.metric("Common results across all methods", len(common_ids))
                        
                else:
                    # Single search method
                    if search_method == "Full-Text Search":
                        results, metrics = search_service.text_search(
                            query,
                            top_k=st.session_state.search_settings["top_k"]
                        )
                    elif search_method == "Vector Search":
                        results, metrics = search_service.vector_search(
                            query,
                            top_k=st.session_state.search_settings["top_k"],
                            min_similarity=st.session_state.search_settings["min_similarity"]
                        )
                    else:  # Hybrid Search
                        results, metrics = search_service.hybrid_search(
                            query,
                            top_k=st.session_state.search_settings["top_k"],
                            min_similarity=st.session_state.search_settings["min_similarity"]
                        )
                    
                    # Add to search history
                    st.session_state.search_history.append({
                        "query": query,
                        "method": search_method.lower().replace("-", "_").replace(" ", "_"),
                        "results": results,
                        "metrics": metrics,
                        "timestamp": datetime.now().strftime("%H:%M:%S")
                    })
                    
                    # Display results
                    display_search_results(results, metrics)

def display_search_results(results, metrics):
    """Display search results in a well-formatted way."""
    st.subheader(f"Search Results ({metrics['search_type']})")
    st.caption(f"Found {len(results)} results in {metrics['query_time_ms']:.2f}ms")
    
    if not results:
        st.info("No results found. Try modifying your search query or adjusting the search settings.")
        return
    
    # Add a score distribution chart
    if len(results) > 1:
        scores = [result.get("searchScore", 0) for result in results]
        score_df = pd.DataFrame({
            "Position": list(range(1, len(scores) + 1)),
            "Score": scores
        })
        
        # Make a small score distribution chart
        chart = alt.Chart(score_df).mark_bar().encode(
            x=alt.X('Position:O', axis=alt.Axis(title='Result Position')),
            y=alt.Y('Score:Q', axis=alt.Axis(title='Score'))
        ).properties(
            height=200
        )
        st.altair_chart(chart, use_container_width=True)
    
    # Display results
    for i, result in enumerate(results):
        with st.expander(f"Result {i+1}: {result['id']}"):
            st.markdown(f"**Score:** {result.get('searchScore', 0):.4f}")
            st.markdown(f"**Document ID:** {result.get('sourceDocumentId', 'N/A')}")
            
            # Show metadata
            if 'metadata' in result and result['metadata']:
                st.markdown("**Metadata:**")
                for key, value in result['metadata'].items():
                    st.markdown(f"- **{key}:** {value}")
            
            # Show technical details without using an expander (to avoid nesting)
            st.markdown("**Technical Details:**")
            st.json(result)

# LLM chat tab with RAG  
def display_chat_tab():
    st.title("💬 Chat with Your Documents")
    st.markdown("""
    Ask questions about your documents and get AI-powered answers using the power of CosmosDB's search capabilities.
    The AI will use document content as context to provide accurate responses.
    """)
    
    # Initialize CosmosDB client
    cosmos_db = CosmosDBClient(
        st.session_state.settings["cosmos_db_name"],
        st.session_state.settings["cosmos_container_name"]
    )
    
    # Initialize chat service
    chat_service = ChatService(
        cosmos_db,
        model_id=st.session_state.chat_settings["llm_model"],
        embedding_model=st.session_state.chat_settings["embedding_model"]
    )
    
    # Create columns for chat interface
    col1, col2 = st.columns([3, 1])
    
    # Right column - Settings and metrics
    with col2:
        # Chat settings
        with st.expander("Chat Settings", expanded=True):
            st.session_state.chat_settings["search_type"] = st.selectbox(
                "Context Search Method",
                ["hybrid", "vector", "text"],
                index=["hybrid", "vector", "text"].index(st.session_state.chat_settings["search_type"]),
                help="Method used to retrieve document context for the AI",
                key="chat_search_type"
            )
            
            st.session_state.chat_settings["top_k"] = st.slider(
                "Number of context chunks",
                min_value=1,
                max_value=10,
                value=st.session_state.chat_settings["top_k"],
                help="How many document chunks to retrieve for context",
                key="chat_top_k"
            )
            
            st.session_state.chat_settings["min_similarity"] = st.slider(
                "Minimum similarity score",
                min_value=0.0,
                max_value=1.0,
                value=st.session_state.chat_settings["min_similarity"],
                step=0.05,
                help="Minimum similarity threshold for vector search (not used in text search)",
                key="chat_min_similarity"
            )
            
            st.session_state.chat_settings["llm_model"] = st.selectbox(
                "LLM Model",
                ["gpt4o_1", "gpt35_turbo"],
                index=0 if st.session_state.chat_settings["llm_model"] == "gpt4o_1" else 1,
                help="Large Language Model used for generating responses",
                key="chat_llm_model"
            )
            
            st.session_state.chat_settings["temperature"] = st.slider(
                "Temperature",
                min_value=0.0,
                max_value=1.0,
                value=st.session_state.chat_settings["temperature"],
                step=0.1,
                help="Controls randomness of responses (lower = more deterministic)",
                key="chat_temperature"
            )
            
            st.session_state.show_context = st.toggle(
                "Show retrieved context",
                value=st.session_state.show_context,
                help="Display the document context used for generating responses",
                key="chat_show_context"
            )
        
        # Chat metrics
        with st.expander("Chat Analytics", expanded=True):
            if st.session_state.chat_history:
                col1_metrics, col2_metrics = st.columns(2)
                
                with col1_metrics:
                    st.metric("Total Queries", st.session_state.chat_metrics["total_queries"])
                    st.metric("Avg. Response Time", f"{st.session_state.chat_metrics['avg_response_time']:.2f}ms")
                
                with col2_metrics:
                    last_query = st.session_state.chat_history[-1]
                    st.metric("Last Query Time", f"{last_query['metrics'].get('elapsed_time_ms', 0):.2f}ms")
                    st.metric("Context Chunks", last_query['metrics'].get('results_retrieved', 0))
                
                # Search type distribution
                search_types = [chat["metrics"]["search_type"] for chat in st.session_state.chat_history]
                search_type_counts = {t: search_types.count(t) for t in set(search_types)}
                
                search_df = pd.DataFrame({
                    "Search Type": list(search_type_counts.keys()),
                    "Count": list(search_type_counts.values())
                })
                
                st.subheader("Search Methods Used")
                fig = px.pie(
                    search_df, 
                    values='Count', 
                    names='Search Type', 
                    color='Search Type',
                    hole=0.4, 
                    height=200
                )
                fig.update_layout(margin=dict(l=5, r=5, t=5, b=5))
                st.plotly_chart(fig, use_container_width=True)
                
                # Response time trends
                if len(st.session_state.chat_history) > 1:
                    response_times = [
                        {
                            "Query": i+1, 
                            "Time (ms)": chat["metrics"].get("elapsed_time_ms", 0),
                            "Type": chat["metrics"]["search_type"]
                        } 
                        for i, chat in enumerate(st.session_state.chat_history[-10:])  # Last 10 queries
                    ]
                    response_df = pd.DataFrame(response_times)
                    
                    st.subheader("Response Time Trends")
                    fig = px.line(
                        response_df, 
                        x="Query", 
                        y="Time (ms)", 
                        color="Type", 
                        markers=True,
                        height=200
                    )
                    fig.update_layout(margin=dict(l=5, r=5, t=5, b=5))
                    st.plotly_chart(fig, use_container_width=True)
    
    # Left column - Chat interface
    with col1:
        # Display chat messages
        chat_container = st.container()
        
        with chat_container:
            # Display chat history
            for message in st.session_state.chat_messages:
                with st.chat_message(message["role"]):
                    st.write(message["content"])
            
            # Add placeholder for streaming response
            response_placeholder = st.empty()
        
        # Chat input and buttons
        chat_input_container = st.container()
        
        with chat_input_container:
            # User input row with buttons
            col_input, col_buttons = st.columns([4, 1])
            
            with col_input:
                user_input = st.chat_input("Ask a question about your documents", key="chat_user_input")
            
            with col_buttons:
                clear_chat = st.button("🗑️ Clear Chat", key="clear_chat_button")
                if clear_chat:
                    st.session_state.chat_messages = []
                    st.rerun()
        
        # Process user input
        if user_input:
            # Add user message to chat
            st.session_state.chat_messages.append({"role": "user", "content": user_input})
            
            # Display user message
            with st.chat_message("user"):
                st.write(user_input)
            
            # Check if documents are available
            try:
                doc_sources = cosmos_db.get_unique_document_sources()
                if not doc_sources:
                    with st.chat_message("assistant"):
                        st.write("I don't have any documents indexed yet. Please upload and process some documents in the 'Document Loading & Indexing' tab first.")
                    st.session_state.chat_messages.append({"role": "assistant", "content": "I don't have any documents indexed yet. Please upload and process some documents in the 'Document Loading & Indexing' tab first."})
                    return
            except Exception as e:
                with st.chat_message("assistant"):
                    st.write(f"Error connecting to document database: {str(e)}. Please check your settings and try again.")
                st.session_state.chat_messages.append({"role": "assistant", "content": f"Error connecting to document database: {str(e)}. Please check your settings and try again."})
                return
            
            # Format messages for OpenAI API
            formatted_messages = [
                {"role": msg["role"], "content": msg["content"]}
                for msg in st.session_state.chat_messages
                if msg["role"] in ["user", "assistant", "system"]
            ]
            
            # Display assistant response with streaming
            with st.chat_message("assistant"):
                full_response = ""
                message_placeholder = st.empty()
                
                # Start spinner during initial context retrieval
                with st.spinner("Retrieving document context..."):
                    # Process streaming response
                    context = None
                    metrics = None
                    
                    # Use the chat service to generate a response with RAG
                    for content_chunk, result in chat_service.chat_with_rag(
                        query=user_input,
                        messages=formatted_messages,
                        search_type=st.session_state.chat_settings["search_type"],
                        top_k=st.session_state.chat_settings["top_k"],
                        min_similarity=st.session_state.chat_settings["min_similarity"]
                    ):
                        if content_chunk is not None:
                            # This is a content chunk
                            full_response += content_chunk
                            message_placeholder.markdown(full_response + "▌")
                        elif isinstance(result, dict) and "context_retrieval_time_ms" in result:
                            # This is metrics information
                            metrics = result
                            
                            # Update chat metrics
                            st.session_state.chat_metrics["total_queries"] += 1
                            current_avg = st.session_state.chat_metrics["avg_response_time"]
                            query_count = st.session_state.chat_metrics["total_queries"]
                            new_time = metrics.get("elapsed_time_ms", 0)
                            
                            # Compute running average
                            st.session_state.chat_metrics["avg_response_time"] = (
                                (current_avg * (query_count - 1) + new_time) / query_count
                            )
                            
                            # Store in chat history
                            st.session_state.chat_history.append({
                                "query": user_input,
                                "response": full_response,
                                "metrics": metrics,
                                "timestamp": datetime.now().strftime("%H:%M:%S")
                            })
                        elif result is not None:
                            # This is an error message
                            message_placeholder.error(f"Error: {result}")
                            full_response = f"Error: {result}"
                
                # Show final response without cursor
                message_placeholder.markdown(full_response)
                
                # Display context if enabled
                if st.session_state.show_context and metrics:
                    with st.expander("📄 Context Used", expanded=False):
                        st.caption(f"Retrieved {metrics.get('results_retrieved', 0)} document chunks using {metrics.get('search_type', 'unknown')} search")
                        st.info(f"Query execution time: {metrics.get('context_retrieval_time_ms', 0):.2f}ms")
                        
                        # Show search results visualization if available
                        if "search_metrics" in metrics and "results_count" in metrics["search_metrics"]:
                            search_metrics = metrics["search_metrics"]
                            st.caption(f"Found {search_metrics['results_count']} results in {search_metrics['query_time_ms']:.2f}ms")
                
            # Add assistant response to chat history
            st.session_state.chat_messages.append({"role": "assistant", "content": full_response})

# Main function
def main():
    # Initialize the app
    init_app()
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs([
        "📄 Document Loading & Indexing", 
        "🔍 Document Search",
        "💬 LLM Chat with RAG"
    ])
    
    # Document Loading & Indexing Tab
    with tab1:
        col1, col2 = st.columns([3, 1])
        
        # Right column - Configuration and metrics
        display_configuration(col2)
        display_metrics(col2)
        
        # Left column - Document upload and processing
        display_document_upload(col1)
        display_processing_history(col1)
    
    # Document Search Tab
    with tab2:
        display_search_tab()
        
    # LLM Chat with RAG Tab
    with tab3:
        display_chat_tab()

if __name__ == "__main__":
    main()

