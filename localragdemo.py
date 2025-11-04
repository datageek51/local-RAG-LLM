import streamlit as st
import PyPDF2
from Utilities import * 
import os
os.environ["ANONYMIZED_TELEMETRY"] = "False"

import ollama
try:
    import chromadb
except Exception:
    chromadb = None
from chat_page import chat_page  # import the chat interface


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
        
        st.markdown("### Session State")
        st.json(dict(st.session_state))
        
        st.markdown("### Environment")
        st.text(f"ANONYMIZED_TELEMETRY: {os.environ.get('ANONYMIZED_TELEMETRY', 'Not Set')}")
        
        if st.button("Clear Session State"):
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
                reader = PdfReader(uploaded_doc)

                # Collect chunks from all pages
                documents = []
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        # Chunk each page and extend the documents list
                        chunks = fixed_size_chunking(page_text, 200)
                        documents.extend(chunks)

                if not documents:
                    st.warning("No text could be extracted from the uploaded PDF.")
                else:
                    st.info(f"Extracted {len(documents)} text chunks. Processing with ChromaDB...")
                    
                    # Progress bar for batch processing
                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    def update_progress(current: int, total: int):
                        progress = float(current) / float(total) if total > 0 else 0
                        progress_bar.progress(progress)
                        status_text.text(f"Processing chunks: {current}/{total}")

                    try:
                        # Initialize or get ChromaDB collection with progress tracking
                        # Create a metadata dict to attach to each chunk
                        doc_meta = {}
                        if doc_source:
                            doc_meta['source'] = doc_source
                        if doc_category:
                            doc_meta['category'] = doc_category

                        collection = initialize_chroma_db(
                            documents=documents,
                            batch_size=10,  # process 10 docs at a time
                            progress_callback=update_progress,
                            document_metadata=doc_meta if doc_meta else None,
                            collection_name=chosen_collection_name
                        )
                        progress_bar.progress(1.0)  # ensure bar shows complete
                        status_text.text(f"Successfully processed all {len(documents)} chunks!")
                        
                        # Store collection in session state for chat page
                        st.session_state['chroma_collection'] = collection
                        st.session_state['doc_processed'] = True

                        # Preview the first few chunks
                        with st.expander("Document chunks preview (first 5)", expanded=False):
                            for i, chunk in enumerate(documents[:5]):
                                st.text(f"Chunk {i}:")
                                st.text(chunk[:500] + "..." if len(chunk) > 500 else chunk)
                        
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
                        st.error(f"Failed to process documents: {e}")
                        if st.button("Clear & Try Again"):
                            st.session_state['uploaded_doc'] = None
                            st.session_state['doc_processed'] = False
                            try:
                                st.experimental_rerun()
                            except Exception:
                                try:
                                    st.rerun()
                                except Exception:
                                    raise RuntimeError('Streamlit rerun API not available. Please upgrade Streamlit.')

# Main screen navigation
def main():
    st.sidebar.title("📁 Navigation")
    choice = st.sidebar.radio("Select Processing Type:", [
        "🏠 Home", 
        "📄 Document Processing",
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
        - **Chat**: Ask questions about your processed documents.
        """)
    elif choice == "📄 Document Processing" or choice == "Document Processing":
        document_processing_page()
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