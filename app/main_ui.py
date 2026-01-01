"""Streamlit UI for NYC Tax Law RAG System.

This module provides a web interface for the NYC Tax Law AI system
with system internals visibility and database management.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Add project root to Python path
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st


def get_qdrant_config() -> tuple[str | None, str | None]:
    """Get Qdrant configuration from secrets or environment.

    Returns:
        Tuple of (qdrant_url, qdrant_api_key) or (None, None) if not configured.
    """
    # Try Streamlit secrets first (for Streamlit Cloud)
    try:
        qdrant_url = st.secrets.get("QDRANT_URL")
        qdrant_api_key = st.secrets.get("QDRANT_API_KEY")
        if qdrant_url and qdrant_api_key:
            return qdrant_url, qdrant_api_key
    except Exception:
        pass

    # Fall back to environment variables
    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")
    if qdrant_url and qdrant_api_key:
        return qdrant_url, qdrant_api_key

    return None, None

# Page configuration - must be first Streamlit command
st.set_page_config(
    page_title="NYC Tax Law AI",
    page_icon="⚖️",
    layout="wide",
)


@st.cache_resource
def load_vectorstore():
    """Load the vector store (Qdrant Cloud or local ChromaDB)."""
    from app.ingest import VectorStoreManager

    qdrant_url, qdrant_api_key = get_qdrant_config()

    if qdrant_url and qdrant_api_key:
        # Cloud mode - use Qdrant
        manager = VectorStoreManager(
            use_qdrant=True,
            qdrant_url=qdrant_url,
            qdrant_api_key=qdrant_api_key,
        )
    else:
        # Local mode - use ChromaDB
        manager = VectorStoreManager()

    return manager.load_vector_db()


@st.cache_resource
def get_advisor(
    _vectorstore,
    reranker_enabled: bool,
    reranker_threshold: float,
    confidence_threshold: float,
):
    """Initialize TaxAdvisor with vectorstore.

    Args:
        _vectorstore: ChromaDB vector store instance.
        reranker_enabled: Whether to enable re-ranking.
        reranker_threshold: Minimum score for documents to pass.
        confidence_threshold: Minimum best score for high-confidence answers.

    Returns:
        Configured TaxAdvisor instance.
    """
    from app.generator import TaxAdvisor

    return TaxAdvisor(
        vectorstore=_vectorstore,
        docstore_path="data/docstore.json",
        generation_model="o3-mini",
        fallback_model="gpt-5.2-pro",
        reranker_enabled=reranker_enabled,
        reranker_threshold=reranker_threshold,
        confidence_threshold=confidence_threshold,
    )


def check_data_files() -> tuple[bool, list[str]]:
    """Check if required data files exist.

    Returns:
        Tuple of (all_exist, missing_files).
    """
    required_files = [
        Path("data/tax_law.txt"),
        Path("data/docstore.json"),
        Path("vectorstore"),  # ChromaDB vector store directory
    ]
    missing = [str(f) for f in required_files if not f.exists()]
    return len(missing) == 0, missing


def rebuild_database() -> None:
    """Rebuild the entire database from source files."""
    from app.ingest import DocumentEnricher, LegalParser, VectorStoreManager

    progress = st.progress(0, text="Starting database rebuild...")

    # Step 1: Parse documents
    progress.progress(10, text="Parsing legal documents...")
    parser = LegalParser()
    sections = parser.parse_file("data/tax_law.txt")
    st.sidebar.info(f"Parsed {len(sections)} sections")

    # Step 2: Enrich documents
    progress.progress(30, text="Enriching documents with AI metadata...")
    enricher = DocumentEnricher()
    enriched = enricher.enrich_batch(sections, "data/docstore.json")
    st.sidebar.info(f"Enriched {len(enriched)} documents")

    # Step 3: Create vector store
    progress.progress(70, text="Creating vector embeddings...")
    manager = VectorStoreManager()
    manager.create_vector_db("data/docstore.json")

    progress.progress(100, text="Database rebuild complete!")


def render_sidebar() -> tuple[bool, float, float]:
    """Render the sidebar with system internals and config.

    Returns:
        Tuple of (reranker_enabled, reranker_threshold, confidence_threshold).
    """
    st.sidebar.title("System Internals")

    # Show last query's HyDE expansion
    if "last_response" in st.session_state and st.session_state.last_response:
        response = st.session_state.last_response

        with st.sidebar.expander("Expanded Query (HyDE)", expanded=False):
            st.text(response.debug_info.hyde_query or "No HyDE expansion")

        with st.sidebar.expander("Retrieved Sections", expanded=False):
            if response.sources:
                for source in response.sources:
                    st.markdown(
                        f"**§ {source.section_id}** - Score: {source.relevance_score:.1f}/10"
                    )
                    st.caption(source.summary or "No summary")
                    st.divider()
            else:
                st.info("No sections retrieved")

        # Debug info with confidence metrics
        with st.sidebar.expander("Pipeline Stats", expanded=False):
            debug = response.debug_info
            st.metric("Documents Retrieved", debug.documents_retrieved)
            st.metric("After Re-ranking", debug.documents_after_rerank)
            st.metric("Best Score", f"{debug.best_score:.1f}/10")
            st.metric("Avg Score", f"{debug.avg_score:.1f}/10")
            st.metric("Processing Time", f"{debug.processing_time_ms:.0f}ms")
            st.caption(f"Model: {debug.model_used}")
            if debug.fallback_model:
                st.caption(f"Fallback Model: {debug.fallback_model}")
            st.caption(f"Confidence: {response.confidence_level.upper()}")

    st.sidebar.divider()

    # Configuration section
    st.sidebar.subheader("Configuration")

    reranker_enabled = st.sidebar.checkbox(
        "Enable Re-ranker",
        value=False,
        help="Toggle re-ranking guardrail for A/B testing",
    )

    reranker_threshold = st.sidebar.slider(
        "Re-ranker Threshold",
        min_value=0.0,
        max_value=10.0,
        value=7.0,
        step=0.5,
        help="Minimum relevance score (0-10) for documents to pass",
        disabled=not reranker_enabled,
    )

    confidence_threshold = st.sidebar.slider(
        "Confidence Threshold",
        min_value=7.0,
        max_value=10.0,
        value=8.0,
        step=0.5,
        help="Below this score, use fallback model for general advice",
        disabled=not reranker_enabled,
    )

    st.sidebar.divider()

    # Database management
    st.sidebar.subheader("Database Management")

    if st.sidebar.button("Re-build Database", type="secondary"):
        with st.spinner("Rebuilding database... This may take a while."):
            try:
                rebuild_database()
                st.cache_resource.clear()
                st.sidebar.success("Database rebuilt successfully!")
                st.rerun()
            except Exception as e:
                st.sidebar.error(f"Rebuild failed: {str(e)}")

    return reranker_enabled, reranker_threshold, confidence_threshold


def render_main_area(
    reranker_enabled: bool,
    reranker_threshold: float,
    confidence_threshold: float,
) -> None:
    """Render the main chat area.

    Args:
        reranker_enabled: Whether re-ranking is enabled.
        reranker_threshold: Minimum score for documents to pass.
        confidence_threshold: Minimum best score for high-confidence answers.
    """
    st.title("NYC Tax Law AI (Advanced RAG)")
    st.caption(
        "Professional tax guidance for NYC tax preparers with strict legal citations"
    )

    # Check for required files
    files_exist, missing = check_data_files()

    if not files_exist:
        st.error("Required data files are missing:")
        for f in missing:
            st.write(f"- `{f}`")
        st.info(
            "Click 'Re-build Database' in the sidebar after placing `data/tax_law.txt`"
        )
        return

    # Load vectorstore and advisor
    try:
        vectorstore = load_vectorstore()
        advisor = get_advisor(
            vectorstore, reranker_enabled, reranker_threshold, confidence_threshold
        )
    except Exception as e:
        st.error(f"Failed to initialize system: {str(e)}")
        st.info("Try clicking 'Re-build Database' in the sidebar")
        return

    # Query input
    user_query = st.text_area(
        "Ask a question about NYC Tax Law:",
        placeholder="e.g., What are the property tax rates for commercial buildings?",
        height=100,
    )

    col1, col2 = st.columns([1, 5])
    with col1:
        ask_button = st.button("Ask Question", type="primary", disabled=not user_query)
    with col2:
        if st.button("Clear History"):
            advisor.clear_history()
            st.session_state.last_response = None
            st.rerun()

    # Process query
    if ask_button and user_query:
        with st.spinner("Analyzing NYC Tax Laws..."):
            try:
                response = advisor.answer_question(user_query)
                st.session_state.last_response = response

                # Add to conversation history
                advisor.add_to_history(user_query, response.answer)
            except Exception as e:
                st.error(f"Error processing question: {str(e)}")
                return

    # Display response
    if "last_response" in st.session_state and st.session_state.last_response:
        response = st.session_state.last_response

        st.divider()

        # Three-tier display based on confidence level
        if response.confidence_level == "none" or response.debug_info.documents_after_rerank == 0:
            # NO CONFIDENCE - Yellow warning
            st.warning("No Relevant Documents Found")
            st.markdown(response.answer)
        elif response.confidence_level == "low" or response.is_fallback:
            # LOW CONFIDENCE - Orange/Yellow with fallback indicator
            st.warning("General Guidance (Low Confidence)")
            st.markdown(response.answer)

            # Show sources as "Potentially Related Sections"
            if response.sources:
                with st.expander(
                    f"Potentially Related Sections ({len(response.sources)})",
                    expanded=False,
                ):
                    for source in response.sources:
                        st.markdown(f"**§ {source.section_id}** ({source.relevance_score:.1f}/10)")
                        st.caption(source.summary or "No summary available")
                        st.divider()
        else:
            # HIGH CONFIDENCE - Green success
            st.success("Answer Generated Successfully")
            st.markdown(response.answer)

            # Show sources as "Sources Used"
            if response.sources:
                with st.expander(
                    f"Sources Used ({len(response.sources)})", expanded=False
                ):
                    for source in response.sources:
                        st.markdown(f"### § {source.section_id}")
                        st.caption(f"Relevance Score: {source.relevance_score:.1f}/10")
                        if source.summary:
                            st.write(f"**Summary:** {source.summary}")
                        if source.text_preview:
                            st.text(f"Preview: {source.text_preview}...")
                        st.divider()

        # Warnings expander
        if response.warnings:
            with st.expander(f"Warnings ({len(response.warnings)})", expanded=True):
                for warning in response.warnings:
                    st.warning(warning)


def main() -> None:
    """Main entry point for the Streamlit app."""
    # Initialize session state
    if "last_response" not in st.session_state:
        st.session_state.last_response = None

    # Render sidebar and get config
    reranker_enabled, reranker_threshold, confidence_threshold = render_sidebar()

    # Render main area
    render_main_area(reranker_enabled, reranker_threshold, confidence_threshold)


if __name__ == "__main__":
    main()
