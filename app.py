import os
import time
import streamlit as st
import pandas as pd
import altair as alt
from streamlit_extras.switch_page_button import switch_page
import plotly.express as px
from datetime import datetime

from app_config import DEFAULT_SETTINGS, UI_CONFIG, METRICS
from app_utils import save_uploaded_file, validate_file, clear_temp_files, format_time
from document_processor import DocumentProcessor

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
                index=0 if st.session_state.settings["embedding_model"] == "text_embedding_3_small" else 1
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

# Document search tab (placeholder)
def display_search_tab():
    st.info("Document search functionality coming soon!")

# LLM chat tab (placeholder)  
def display_chat_tab():
    st.info("LLM chat functionality coming soon!")

# Main function
def main():
    # Initialize the app
    init_app()
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs([
        "📄 Document Loading & Indexing", 
        "🔍 Document Search (Coming Soon)",
        "💬 LLM Chat (Coming Soon)"
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
    
    # Document Search Tab (placeholder)
    with tab2:
        display_search_tab()
        
    # LLM Chat Tab (placeholder)
    with tab3:
        display_chat_tab()

if __name__ == "__main__":
    main()

