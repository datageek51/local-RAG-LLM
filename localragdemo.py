import streamlit as st
import PyPDF2
from Utilities import * 
import os
import logging
from datetime import datetime
os.environ["ANONYMIZED_TELEMETRY"] = "False"

# Configure logging
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"app_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()  # Also print to console
    ]
)

logger = logging.getLogger(__name__)
logger.info(f"Application started. Log file: {log_file}")

import ollama
try:
    import chromadb
except Exception:
    chromadb = None
from chat_page import chat_page  # import the chat interface
from audio_page import audio_processing_page  # import the audio processing page


# Note: previously there was a compatibility wrapper for rerun here.
# To avoid an extra wrapper, call Streamlit's rerun APIs inline where needed
# and handle the AttributeError at the call site.

def show_debug_panel():
    """Shows a collapsible debug panel with system information and app state"""
    with st.sidebar.expander("🔧 Debug Panel", expanded=False):
        st.markdown("### System Info")
        if chromadb is not None:
            try:
                version = getattr(chromadb, '__version__', 'unknown')
            except Exception:
                version = 'unknown'
            st.text(f"ChromaDB Version: {version}")
        else:
            st.text("ChromaDB: not installed")
        st.text(f"Ollama Available: {hasattr(ollama, 'chat')}")
        
        st.markdown("### Logging")
        st.text(f"Log file: {log_file}")
        if st.button("View Recent Logs (last 50 lines)"):
            try:
                with open(log_file, 'r') as f:
                    lines = f.readlines()
                    recent_lines = lines[-50:] if len(lines) > 50 else lines
                    st.text_area("Recent Log Entries", ''.join(recent_lines), height=300)
            except Exception as e:
                st.error(f"Could not read log file: {e}")
        
        st.markdown("### Session State")
        st.json(dict(st.session_state))
        
        st.markdown("### Environment")
        st.text(f"ANONYMIZED_TELEMETRY: {os.environ.get('ANONYMIZED_TELEMETRY', 'Not Set')}")
        
        if st.button("Clear Session State"):
            logger.info("Session state cleared by user")
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            try:
                st.experimental_rerun()
            except Exception:
                try:
                    st.rerun()
                except Exception:
                    raise RuntimeError('Streamlit rerun API not available. Please upgrade Streamlit.')


# Mock document classifier (not currently used but preserved for future use)
def mock_document_classifier(text):
    classes = ['Category X', 'Category Y', 'Category Z']
    # Simple mock without numpy dependency
    from random import choice
    return choice(classes)

# Document processing page
def document_processing_page():
    st.title("📄 Document ML Processing")

    # add a key so we can clear the uploader programmatically via session_state
    uploaded_doc = st.file_uploader("Upload Document File (PDF)", type=['pdf'], key='uploaded_doc')

    # Option to split PDF into smaller files
    split_pdf_option = st.checkbox("Split PDF into smaller files", value=False)
    
    if split_pdf_option:
        pages_per_file = st.number_input("Pages per file:", min_value=1, max_value=1000, value=10, step=1)
    
    if split_pdf_option and uploaded_doc is not None:
        st.info(f"📄 Split mode: The PDF will be split into multiple files with {pages_per_file} pages each before processing.")
        if st.button("Split PDF and Save Files"):
            with st.spinner('Splitting PDF...'):
                from PyPDF2 import PdfReader, PdfWriter
                import os
                
                try:
                    reader = PdfReader(uploaded_doc)
                    total_pages = len(reader.pages)
                    
                    # Calculate number of splits
                    num_splits = (total_pages + pages_per_file - 1) // pages_per_file
                    
                    # Get original filename without extension
                    original_filename = uploaded_doc.name
                    base_name = os.path.splitext(original_filename)[0]
                    
                    # Create output directory if it doesn't exist
                    output_dir = "split_pdfs"
                    os.makedirs(output_dir, exist_ok=True)
                    
                    split_files = []
                    
                    # Split the PDF
                    for split_num in range(1, num_splits + 1):
                        writer = PdfWriter()
                        
                        # Calculate page range for this split
                        start_page = (split_num - 1) * pages_per_file
                        end_page = min(split_num * pages_per_file, total_pages)
                        
                        # Add pages to this split
                        for page_num in range(start_page, end_page):
                            writer.add_page(reader.pages[page_num])
                        
                        # Generate output filename
                        output_filename = f"{base_name}-{split_num}.pdf"
                        output_path = os.path.join(output_dir, output_filename)
                        
                        # Write the split PDF
                        with open(output_path, 'wb') as output_file:
                            writer.write(output_file)
                        
                        split_files.append(output_filename)
                    
                    st.success(f"✅ Successfully split {total_pages} pages into {num_splits} PDF files!")
                    st.markdown(f"**Files saved in `{output_dir}/` folder:**")
                    for file in split_files:
                        st.text(f"  • {file}")
                    
                except Exception as e:
                    logger.error(f"Failed to split PDF: {str(e)}", exc_info=True)
                    st.error(f"Failed to split PDF: {e}")
        
        st.markdown("---")

    # Metadata inputs for this document (applied to all chunks extracted from the uploaded file)
    st.markdown("**Document metadata (applies to all extracted chunks):**")
    doc_source = st.text_input("Document Source", value="")
    doc_category = st.text_input("Document Category", value="")

    # Collection selection: list existing collections and allow creating a new one
    st.markdown("**Target ChromaDB Collection:**")
    collection_names = []
    try:
        import chromadb
        from chromadb.config import Settings
        client = chromadb.PersistentClient(path=CHROMA_PATH, settings=Settings(anonymized_telemetry=False))
        cols = client.list_collections()
        # extract names safely
        for c in cols:
            try:
                name = c.name if hasattr(c, 'name') else str(c)
            except Exception:
                name = str(c)
            collection_names.append(name)
    except Exception:
        # chromadb not available or client couldn't connect
        collection_names = []

    new_collection_label = "-- Create new collection --"
    options = collection_names.copy()
    options.insert(0, new_collection_label)

    # show select + refresh button side-by-side
    sel_col, ref_col = st.columns([8, 1])
    with sel_col:
        selected_collection = st.selectbox("Choose collection (existing or create new):", options, index=0)
    with ref_col:
        if st.button("🔄", help="Refresh collections list", key="refresh_collections_doc_page"):
            # simple approach: rerun the page which will re-query collections
            try:
                st.rerun()
            except AttributeError:
                try:
                    st.experimental_rerun()
                except Exception:
                    raise RuntimeError('Streamlit rerun API not available. Please upgrade Streamlit.')

    new_collection_name = None
    chosen_collection_name = None
    is_new = selected_collection == new_collection_label
    if is_new:
        new_collection_name = st.text_input("New collection name:", value="")
        # validate new name immediately
        if new_collection_name:
            chosen_collection_name = new_collection_name.strip()
        else:
            chosen_collection_name = None
    else:
        chosen_collection_name = selected_collection

    # Validate the new collection name if provided
    def _validate_collection_name(name: str, existing: list) -> tuple[bool, str]:
        if not name or not name.strip():
            return False, "Collection name cannot be empty"
        if len(name) < 1 or len(name) > 64:
            return False, "Collection name must be between 1 and 64 characters"
        # allow letters, digits, hyphen, underscore
        import re as _re
        if not _re.match(r'^[A-Za-z0-9_\-]+$', name):
            return False, "Use only letters, numbers, underscores and hyphens"
        if name in existing:
            return False, "A collection with that name already exists"
        return True, ""

    collection_name_valid = True
    collection_name_msg = ""
    if is_new:
        ok, msg = _validate_collection_name(chosen_collection_name or "", collection_names)
        collection_name_valid = ok
        collection_name_msg = msg
        if not ok and chosen_collection_name:
            st.error(f"Invalid collection name: {msg}")
        elif not chosen_collection_name:
            st.info("Enter a name for the new collection before processing")

    # allow processing only when the user clicks the button
    if uploaded_doc is not None:
        # disable processing button if creating a new collection and name invalid/missing
        disable_process = False
        if is_new and (not chosen_collection_name or not collection_name_valid):
            disable_process = True

        if st.button("Extract Text and Process Document", disabled=disable_process):
            with st.spinner('Extracting text from PDF...'):
                from PyPDF2 import PdfReader
                
                logger.info(f"Starting text extraction for file: {uploaded_doc.name}")
                logger.debug(f"File size: {uploaded_doc.size} bytes")
                logger.debug(f"Collection name: {chosen_collection_name}")
                logger.debug(f"Metadata - Source: {doc_source}, Category: {doc_category}")
                
                try:
                    # Reset file pointer to beginning
                    uploaded_doc.seek(0)
                    logger.debug("File pointer reset to beginning before reading")
                    
                    reader = PdfReader(uploaded_doc)
                    total_pages = len(reader.pages)
                    logger.info(f"PDF loaded. Total pages: {total_pages}")

                    # Collect chunks from all pages
                    documents = []
                    for page_idx, page in enumerate(reader.pages):
                        logger.debug(f"Processing page {page_idx + 1}/{total_pages}")
                        try:
                            page_text = page.extract_text()
                            text_length = len(page_text) if page_text else 0
                            logger.debug(f"Page {page_idx + 1}: extracted {text_length} characters")
                            
                            if page_text:
                                # Log first 100 chars of extracted text for debugging
                                logger.debug(f"Page {page_idx + 1} text preview: {page_text[:100]}...")
                                
                                # Chunk each page and extend the documents list
                                chunks = fixed_size_chunking(page_text, 2000)
                                logger.debug(f"Page {page_idx + 1}: created {len(chunks)} chunks")
                                documents.extend(chunks)
                            else:
                                logger.warning(f"Page {page_idx + 1}: No text extracted (empty page or image-only)")
                        except Exception as page_error:
                            logger.error(f"Error processing page {page_idx + 1}: {str(page_error)}", exc_info=True)

                    logger.info(f"Text extraction complete. Total chunks created: {len(documents)}")
                    
                    if not documents:
                        logger.warning("No text could be extracted from any page of the PDF")
                        st.warning("No text could be extracted from the uploaded PDF.")
                        st.info("💡 This may happen if the PDF contains only images or scanned content. Try a text-based PDF.")
                    else:
                        st.info(f"Extracted {len(documents)} text chunks. Processing with ChromaDB...")
                        logger.info(f"Starting ChromaDB processing for {len(documents)} chunks")
                        
                        # Progress bar for batch processing
                        progress_bar = st.progress(0)
                        status_text = st.empty()

                        def update_progress(current: int, total: int):
                            progress = float(current) / float(total) if total > 0 else 0
                            progress_bar.progress(progress)
                            status_text.text(f"Processing chunks: {current}/{total}")
                            logger.debug(f"Progress: {current}/{total} chunks processed")

                        try:
                            # Initialize or get ChromaDB collection with progress tracking
                            # Create a metadata dict to attach to each chunk
                            doc_meta = {}
                            if doc_source:
                                doc_meta['source'] = doc_source
                            if doc_category:
                                doc_meta['category'] = doc_category
                            
                            logger.debug(f"Document metadata: {doc_meta}")

                            collection = initialize_chroma_db(
                                documents=documents,
                                batch_size=10,  # process 10 docs at a time
                                progress_callback=update_progress,
                                document_metadata=doc_meta if doc_meta else None,
                                collection_name=chosen_collection_name
                            )
                            logger.info(f"ChromaDB processing complete. Collection: {chosen_collection_name}")
                            
                            progress_bar.progress(1.0)  # ensure bar shows complete
                            status_text.text(f"Successfully processed all {len(documents)} chunks!")
                            
                            # Store collection in session state for chat page
                            st.session_state['chroma_collection'] = collection
                            st.session_state['doc_processed'] = True
                            logger.debug("Session state updated with collection and doc_processed flag")

                            # Preview the first few chunks
                            with st.expander("Document chunks preview (first 5)", expanded=False):
                                for i, chunk in enumerate(documents[:5]):
                                    st.text(f"Chunk {i}:")
                                    st.text(chunk[:2500] + "..." if len(chunk) > 2500 else chunk)
                            
                            # Add a button to navigate to chat
                            col1, col2 = st.columns(2)
                            with col1:
                                if st.button("Go to Chat"):
                                        # trigger a rerun so user can select Chat from the sidebar
                                        try:
                                            st.experimental_rerun()
                                        except Exception:
                                            try:
                                                st.rerun()
                                            except Exception:
                                                raise RuntimeError('Streamlit rerun API not available. Please upgrade Streamlit.')
                            with col2:
                                if st.button("Clear & Upload Another"):
                                    st.session_state['uploaded_doc'] = None
                                    st.session_state['doc_processed'] = False
                                    try:
                                        st.experimental_rerun()
                                    except Exception:
                                        try:
                                            st.rerun()
                                        except Exception:
                                            raise RuntimeError('Streamlit rerun API not available. Please upgrade Streamlit.')

                        except Exception as e:
                            logger.error(f"Failed to process documents: {str(e)}", exc_info=True)
                            st.error(f"Failed to process documents: {e}")
                            if st.button("Clear & Try Again"):
                                logger.info("User clicked 'Clear & Try Again'")
                                st.session_state['uploaded_doc'] = None
                                st.session_state['doc_processed'] = False
                                try:
                                    st.experimental_rerun()
                                except Exception:
                                    try:
                                        st.rerun()
                                    except Exception:
                                        raise RuntimeError('Streamlit rerun API not available. Please upgrade Streamlit.')
                
                except Exception as outer_e:
                    logger.error(f"Unexpected error during PDF processing: {str(outer_e)}", exc_info=True)
                    st.error(f"Unexpected error: {outer_e}")
                    st.info("Check the log file for detailed error information.")

# Main screen navigation
def main():
    st.sidebar.title("📁 Navigation")
    choice = st.sidebar.radio("Select Processing Type:", [
        "🏠 Home", 
        "📄 Document Processing",
        "🎤 Audio Processing",
        "💬 Chat"
    ])
    
    # Add debug panel to sidebar
    show_debug_panel()

    if choice == "🏠 Home":
        st.title("🖥️ AI Based Chat with Documents App")
        st.write("""
        ### Welcome!

        Choose from the sidebar:
        - **Document Processing**: Add documents to the knowledge base.
        - **Audio Processing**: Transcribe and add audio files to the knowledge base.
        - **Chat**: Ask questions about your processed documents and audio.
        """)
    elif choice == "📄 Document Processing" or choice == "Document Processing":
        document_processing_page()
    elif choice == "🎤 Audio Processing" or choice == "Audio Processing":
        audio_processing_page()
    elif choice == "💬 Chat" or choice == "Chat":
        # Pass the ChromaDB collection if documents have been processed
        collection = st.session_state.get('chroma_collection')
        chat_page(collection=collection)

# Main app entry point
if __name__ == "__main__":
    # Initialize session state for collection if needed
    if 'chroma_collection' not in st.session_state:
        st.session_state['chroma_collection'] = None

    # Call main() which handles navigation
    main()