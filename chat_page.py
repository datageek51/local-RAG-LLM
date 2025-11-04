import streamlit as st
from typing import List, Dict, Any, Tuple
import ollama
from Utilities import CHROMA_PATH, COLLECTION_NAME, EMBEDDING_MODEL, LLM_MODEL


# Removed compatibility wrapper to avoid an extra function; calls to Streamlit rerun
# are handled inline at the call sites with try/except to support older versions.

def get_collection_metadata(collection) -> Dict:
    """Get metadata about a collection including size, creation time, etc."""
    try:
        if collection is None:
            return {"error": "No collection provided"}

        # count may raise if collection invalid
        count = collection.count()

        # Get a sample document to check ID format
        sample = collection.get(limit=1)
        has_content_ids = False
        try:
            ids = sample.get('ids', [])
            has_content_ids = any(isinstance(i, str) and i.startswith('d_') for i in ids)
        except Exception:
            has_content_ids = False

        # Safely get embedding function name
        emb_name = "unknown"
        try:
            emb_fn = getattr(collection, '_embedding_function', None)
            if emb_fn is not None:
                # Ollama-style embedding wrapper may have a .model attribute
                if hasattr(emb_fn, 'model'):
                    emb_name = f"ollama:{getattr(emb_fn, 'model') }"
                elif hasattr(emb_fn, 'name'):
                    name_attr = getattr(emb_fn, 'name')
                    emb_name = name_attr if isinstance(name_attr, str) else 'default'
                else:
                    emb_name = str(type(emb_fn))
        except Exception:
            emb_name = 'unknown'

        metadata = {
            "name": getattr(collection, 'name', 'unknown'),
            "count": count,
            "uses_content_hash": has_content_ids,
            "embedding_function": emb_name,
        }
        return metadata
    except Exception as e:
        # If collection is bad, try to return partial info
        return {
            "name": getattr(collection, 'name', 'unknown') if collection is not None else 'unknown',
            "error": str(e)
        }

def get_available_collections():
    """Get list of available ChromaDB collections with metadata."""
    try:
        import chromadb
        from chromadb.config import Settings
        from Utilities import CHROMA_PATH
        
        client = chromadb.PersistentClient(
            path=CHROMA_PATH,
            settings=Settings(anonymized_telemetry=False)
        )
        collections = client.list_collections()
        
        # Enhance with metadata
        collection_info = []
        for col in collections:
            metadata = get_collection_metadata(col)
            collection_info.append((col, metadata))
            
        return client, collection_info
    except Exception as e:
        st.error(f"Failed to connect to ChromaDB: {e}")
        return None, []

def delete_collection(client, name: str) -> bool:
    """Safely delete a collection. Returns True if successful."""
    try:
        client.delete_collection(name)
        return True
    except Exception as e:
        st.error(f"Failed to delete collection {name}: {e}")
        return False
import ollama
import os
from Utilities import CHROMA_PATH, COLLECTION_NAME, EMBEDDING_MODEL
def chat_page(collection=None):
    """
    Displays a chat interface that uses RAG with the provided ChromaDB collection.
    If no collection is provided, tries to list available collections for selection.
    """
    # Set page config to reduce memory usage
    st.set_page_config(layout="wide", initial_sidebar_state="collapsed")
    st.title("💬 Chat with Documents")
    
    # Get available collections
    client, collections = get_available_collections()
    
    if not collections:
        st.warning("No document collections found. Please upload and process documents in the Document Processing page first.")
        if st.button("Go to Document Processing"):
            # streamlit's switch_page only works for files in the pages/ directory.
            # Instead, just rerun the app which will return the user to the main flow.
            try:
                st.experimental_rerun()
            except Exception:
                try:
                    st.rerun()
                except Exception:
                    raise RuntimeError('Streamlit rerun API not available. Please upgrade Streamlit.')
        return
    
    # Collection management sidebar
    with st.sidebar:
        st.title("📚 Collections")
        
        # Refresh button
        col1, col2 = st.columns([4, 1])
        with col1:
            st.subheader("Available Collections")
        with col2:
            if st.button("🔄", help="Refresh collections list", key="refresh_collections_chat_page"):
                try:
                    st.rerun()
                except AttributeError:
                    try:
                        st.experimental_rerun()
                    except Exception:
                        raise RuntimeError('Streamlit rerun API not available. Please upgrade Streamlit.')
        
        # Model Settings Section
        st.divider()
        st.subheader("⚙️ Model Settings")
        
        # CPU-friendly model options (sorted by size)
        cpu_friendly_llm_models = [
            "tinyllama",
            "phi",
            "gemma:2b",
            "qwen2:1.5b",
            "llama3.2:1b",
            "llama3.2:3b",
            "gemma2:2b",
            "mistral",
            "llama3.1:8b"
        ]
        
        # Initialize session state for model selection if not present
        if 'selected_llm_model' not in st.session_state:
            st.session_state.selected_llm_model = os.environ.get('LLM_MODEL', 'gemma:2b')
        
        # LLM Model selector
        st.session_state.selected_llm_model = st.selectbox(
            "💬 Chat Model (LLM)",
            options=cpu_friendly_llm_models,
            index=cpu_friendly_llm_models.index(st.session_state.selected_llm_model) 
                  if st.session_state.selected_llm_model in cpu_friendly_llm_models 
                  else 2,  # default to gemma:2b
            help="Smaller models use less memory and are better for CPU-only systems",
            key="llm_model_selector"
        )
        
        # Show model size hint
        model_sizes = {
            "tinyllama": "~600MB",
            "phi": "~1.6GB",
            "gemma:2b": "~1.6GB",
            "qwen2:1.5b": "~900MB",
            "llama3.2:1b": "~1.3GB",
            "llama3.2:3b": "~2GB",
            "gemma2:2b": "~1.6GB",
            "mistral": "~4GB",
            "llama3.1:8b": "~4.7GB"
        }
        size_hint = model_sizes.get(st.session_state.selected_llm_model, "Unknown size")
        st.caption(f"📊 Approximate size: {size_hint}")
        
        # Info about model selection
        with st.expander("ℹ️ Model Selection Tips", expanded=False):
            st.markdown("""
            **Recommended for low memory:**
            - tinyllama, phi, qwen2:1.5b
            
            **Balanced performance:**
            - gemma:2b, llama3.2:3b
            
            **Better quality (needs more RAM):**
            - mistral, llama3.1:8b
            
            💡 If you get GPU/memory errors, try a smaller model!
            """)
        
        # Collection selector if we don't have a specific collection
        active_collection = collection
        if active_collection is None:
            # Create selection options with metadata
            options = []
            for col, meta in collections:
                doc_count = meta.get('count', '?')
                label = f"{col.name} ({doc_count} docs)"
                options.append((col, label))
            
            if options:
                selected_idx = st.selectbox(
                    "Select a collection:",
                    range(len(options)),
                    format_func=lambda i: options[i][1]
                )
                active_collection = options[selected_idx][0]
                
                # Collection actions
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("🗑️ Delete", help="Delete this collection"):
                        if st.session_state.get('confirm_delete') == active_collection.name:
                            # Second click - do the deletion
                            if delete_collection(client, active_collection.name):
                                st.success(f"Deleted collection {active_collection.name}")
                                st.session_state.pop('confirm_delete', None)
                                active_collection = None
                                try:
                                    st.experimental_rerun()
                                except Exception:
                                    try:
                                        st.rerun()
                                    except Exception:
                                        raise RuntimeError('Streamlit rerun API not available. Please upgrade Streamlit.')
                        else:
                            # First click - ask for confirmation
                            st.session_state['confirm_delete'] = active_collection.name
                            st.warning("Click delete again to confirm")
                
                with col2:
                    if st.button("📊 Details", help="Show collection details"):
                        st.session_state['show_details'] = not st.session_state.get('show_details', False)
                
                # Show collection details if requested
                if st.session_state.get('show_details'):
                    meta = next(m for _, m in collections if m['name'] == active_collection.name)
                    st.write("Collection Details:")
                    st.json(meta)
            
    if active_collection is None:
        st.error("Please select a collection to start chatting.")
        return

    # Initialize chat history in session state if not present
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    # Show active collection info in main area
    meta = get_collection_metadata(active_collection)
    current_llm = st.session_state.get('selected_llm_model', 'gemma:2b')
    st.info(
        f"💬 Chatting with collection: **{meta['name']}**\n\n"
        f"📑 {meta['count']} documents | "
        f"🔍 Using {meta['embedding_function']} embeddings | "
        f"🤖 Chat Model: **{current_llm}** | "
        f"{'✅' if meta.get('uses_content_hash') else '❌'} Content-hash IDs"
    )

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask a question about your documents..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get relevant context via ChromaDB
        with st.spinner("Searching documents..."):
            try:
                # ensure we have a proper collection object (some chroma APIs return metadata dicts)
                if not hasattr(active_collection, 'query') and client is not None:
                    try:
                        # try to fetch a real collection object by name
                        active_collection = client.get_collection(active_collection.name)
                    except Exception:
                        pass

                # use the active_collection selected in the UI (may differ from the optional function param)
                results = active_collection.query(
                    query_texts=[prompt],
                    n_results=3  # adjust based on needs
                )

                # results may contain 'documents' or 'metadatas' depending on chroma version
                docs = []
                if isinstance(results, dict):
                    docs = results.get('documents') or results.get('results') or []
                # normalize to a list of strings
                context_items = []
                if docs:
                    # docs is often a list-of-lists (one per query)
                    first = docs[0]
                    if isinstance(first, list):
                        context_items = [str(d) for d in first if d]
                    else:
                        context_items = [str(first)]

                context = "\n---\n".join(context_items) if context_items else ""
            except Exception as e:
                st.error(f"Failed to query documents: {e}")
                return

        # Generate response with Ollama using RAG context
        with st.spinner("Generating response..."):
            try:
                rag_prompt = f"""
You are a helpful but concise assistant. Keep responses short and focused.
Only use information from the context to answer. If unsure, say "I don't have enough context."
Avoid long explanations - be direct and brief.

Context:
{context}

User question: {prompt}

Answer:"""
                
                # Use the model selected in the UI, or fall back to environment/default
                model_name = st.session_state.get('selected_llm_model', os.environ.get('LLM_MODEL', LLM_MODEL))
                
                # Build a list of models to try: primary + optional fallbacks from env + conservative defaults
                fallbacks_env = os.environ.get("LLM_FALLBACKS", "")
                fallback_list = [m.strip() for m in fallbacks_env.split(",") if m.strip()]
                # Conservative built-in fallbacks (small models that work well on CPU)
                conservative_defaults = ["tinyllama", "phi", "qwen2:1.5b"]
                
                # Try smallest models first when falling back
                models_to_try = [model_name]
                if model_name not in (fallback_list + conservative_defaults):
                    models_to_try.extend(sorted(fallback_list + conservative_defaults, 
                                              key=lambda x: 'phi' in x or 'tiny' in x or '2b' in x, 
                                              reverse=True))

                answer = None
                last_exc = None
                memory_indicators = ["out of memory", "insufficient memory", "cuda out of memory", "allocate", "memoryerror"]

                for m in models_to_try:
                    try:
                        response = ollama.chat(
                            model=m,
                            messages=[{"role": "user", "content": rag_prompt}]
                        )

                        # Robustly extract the assistant content from various Ollama client response shapes
                        if isinstance(response, dict):
                            answer = response.get('message', {}).get('content') or response.get('content') or str(response)
                        else:
                            try:
                                answer = response['message']['content']
                            except Exception:
                                answer = str(response)

                        # On success, append and display
                        st.session_state.messages.append({"role": "assistant", "content": answer})
                        with st.chat_message("assistant"):
                            st.markdown(answer)

                        # Stop trying further models
                        last_exc = None
                        break

                    except Exception as e:
                        last_exc = e
                        msg = str(e).lower()
                        # If the error looks memory-related, try next fallback model
                        if any(ind in msg for ind in memory_indicators):
                            st.warning(
                                f"⚠️ Model '{m}' failed due to memory. "
                                f"Try selecting a smaller model from the sidebar (tinyllama, phi, or qwen2:1.5b recommended)."
                            )
                            continue
                        else:
                            # Non-memory error - show and stop
                            st.error(f"Failed to generate response: {e}")
                            break

                # If no answer produced, provide the earlier context fallback for memory failures
                if answer is None:
                    if last_exc is not None and any(ind in str(last_exc).lower() for ind in memory_indicators):
                        st.error(
                            "❌ Failed to generate model response due to insufficient memory.\n\n"
                            "**💡 Try these solutions:**\n"
                            "- Select a smaller model from the sidebar (recommended: tinyllama, phi, or qwen2:1.5b)\n"
                            "- Close other applications to free up memory\n"
                            "- Make sure Ollama is using CPU mode"
                        )
                        if context:
                            fallback_answer = (
                                "I couldn't generate a model response due to memory limits. "
                                "Here are the most relevant document snippets I found:\n\n" + context
                            )
                        else:
                            fallback_answer = "I couldn't generate a model response due to memory limits and no context was available."

                        st.session_state.messages.append({"role": "assistant", "content": fallback_answer})
                        with st.chat_message("assistant"):
                            st.markdown(fallback_answer)
                    elif last_exc is not None:
                        # Some other error occurred and was already shown above; provide a short message
                        st.error(f"Failed to generate response: {last_exc}")
                    else:
                        st.error("Failed to generate a model response for unknown reasons.")
            
            except Exception as e:
                st.error(f"Failed to generate response: {e}")

    # Add a clear button
    if st.session_state.messages and st.button("Clear Chat History"):
        st.session_state.messages = []
        try:
            st.experimental_rerun()
        except Exception:
            try:
                st.rerun()
            except Exception:
                raise RuntimeError('Streamlit rerun API not available. Please upgrade Streamlit.')