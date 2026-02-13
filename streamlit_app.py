"""AI-Powered Knowledge Graph Generator - Desktop Application.

Inspired by ISC SANS Diary #32712 and Robert McDermott's ai-knowledge-graph project.
Takes unstructured text and generates interactive knowledge graphs using LLMs.
"""

from __future__ import annotations

import streamlit as st
import streamlit.components.v1 as components

from knowledge_graph.chunking import chunk_text
from knowledge_graph.llm import (
    extract_triplets,
    infer_relationships,
    standardize_entities,
)
from knowledge_graph.graph import (
    build_graph,
    compute_node_sizes,
    detect_communities,
    get_graph_stats,
    visualize_graph,
)

st.set_page_config(
    page_title="AI Knowledge Graph Generator",
    page_icon="🔗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -- Sidebar Configuration --
with st.sidebar:
    st.header("LLM Configuration")
    st.caption("Any OpenAI-compatible endpoint: Ollama, LM Studio, OpenAI, vLLM, etc.")

    base_url = st.text_input(
        "API Base URL",
        value="http://localhost:11434/v1",
        help="For Ollama: http://localhost:11434/v1 | For OpenAI: https://api.openai.com/v1",
    )
    model = st.text_input(
        "Model",
        value="gemma3",
        help="Model name (e.g., gemma3, llama3, gpt-4o-mini)",
    )
    api_key = st.text_input(
        "API Key (optional)",
        type="password",
        help="Required for OpenAI, optional for local Ollama",
    )

    st.divider()
    st.header("Chunking Settings")
    chunk_size = st.slider("Words per chunk", 50, 500, 200, step=25)
    chunk_overlap = st.slider("Overlap (words)", 0, 100, 20, step=5)

    st.divider()
    st.header("Processing Options")
    do_standardize = st.checkbox("Entity Standardization", value=True, help="Use LLM to unify duplicate entity names")
    do_inference = st.checkbox("Relationship Inference", value=True, help="Use LLM to infer new relationships between disconnected communities")

    st.divider()
    st.header("Display")
    dark_mode = st.checkbox("Dark Mode", value=True)
    graph_height = st.slider("Graph Height (px)", 400, 1000, 700, step=50)

# -- Main Area --
st.title("AI-Powered Knowledge Graph Generator")
st.caption(
    "Extract knowledge from unstructured text as interactive graphs. "
    "Based on [ISC SANS Diary #32712](https://isc.sans.edu/diary/32712) "
    "and [Robert McDermott's ai-knowledge-graph](https://github.com/robert-mcdermott/ai-knowledge-graph)."
)

# Input section
tab_paste, tab_upload = st.tabs(["Paste Text", "Upload File"])

with tab_paste:
    input_text = st.text_area(
        "Enter or paste your text below:",
        height=250,
        placeholder=(
            "Paste any unstructured text here — news articles, threat intel reports, "
            "research papers, etc. The LLM will extract entities and relationships."
        ),
    )

with tab_upload:
    uploaded_file = st.file_uploader(
        "Upload a plain text file (.txt)",
        type=["txt"],
    )
    if uploaded_file is not None:
        input_text = uploaded_file.read().decode("utf-8", errors="replace")
        st.text_area("File content preview:", value=input_text[:2000], height=200, disabled=True)

# Determine which text to use
text = input_text if "input_text" in dir() and input_text else ""

# Generate button
col1, col2 = st.columns([1, 4])
with col1:
    generate = st.button("Generate Graph", type="primary", disabled=not text.strip())
with col2:
    if not text.strip():
        st.info("Paste text or upload a file to get started.")

# Session state for results
if "triplets" not in st.session_state:
    st.session_state.triplets = []
if "graph_html" not in st.session_state:
    st.session_state.graph_html = ""
if "stats" not in st.session_state:
    st.session_state.stats = {}

if generate and text.strip():
    all_triplets: list[dict] = []

    # Step 1: Chunking
    with st.status("Processing text...", expanded=True) as status:
        st.write("Chunking text...")
        chunks = chunk_text(text, chunk_size=chunk_size, overlap=chunk_overlap)
        st.write(f"Created **{len(chunks)}** text chunks.")

        # Step 2: Entity & relationship extraction
        st.write("Extracting knowledge triplets from each chunk...")
        progress = st.progress(0)
        for i, chunk in enumerate(chunks):
            try:
                triplets = extract_triplets(chunk, base_url, model, api_key)
                all_triplets.extend(triplets)
            except Exception as e:
                st.warning(f"Chunk {i+1}/{len(chunks)} failed: {e}")
            progress.progress((i + 1) / len(chunks))
        st.write(f"Extracted **{len(all_triplets)}** triplets.")

        # Step 3: Entity standardization
        if do_standardize and all_triplets:
            st.write("Standardizing entity names...")
            try:
                all_triplets = standardize_entities(all_triplets, base_url, model, api_key)
                st.write("Entity standardization complete.")
            except Exception as e:
                st.warning(f"Standardization failed: {e}")

        # Step 4: Build graph
        st.write("Building knowledge graph...")
        G = build_graph(all_triplets)
        communities = detect_communities(G)

        # Step 5: Relationship inference
        if do_inference and len(set(communities.values())) > 1:
            st.write("Inferring relationships between communities...")
            community_sets: list[set[str]] = []
            comm_map: dict[int, set[str]] = {}
            for node, cid in communities.items():
                comm_map.setdefault(cid, set()).add(node)
            community_sets = list(comm_map.values())

            try:
                inferred = infer_relationships(all_triplets, community_sets, base_url, model, api_key)
                if inferred:
                    all_triplets.extend(inferred)
                    G = build_graph(all_triplets)
                    communities = detect_communities(G)
                    st.write(f"Inferred **{len(inferred)}** additional relationships.")
            except Exception as e:
                st.warning(f"Inference failed: {e}")

        # Step 6: Compute sizes and visualize
        st.write("Generating visualization...")
        sizes = compute_node_sizes(G)
        stats = get_graph_stats(G, communities)
        html = visualize_graph(
            G, communities, sizes,
            dark_mode=dark_mode,
            height=f"{graph_height}px",
        )

        st.session_state.triplets = all_triplets
        st.session_state.graph_html = html
        st.session_state.stats = stats

        status.update(label="Processing complete!", state="complete")

# Display results
if st.session_state.graph_html:
    st.divider()

    # Stats row
    stats = st.session_state.stats
    if stats:
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Nodes", stats.get("nodes", 0))
        c2.metric("Edges (total)", stats.get("edges", 0))
        c3.metric("Extracted", stats.get("extracted_edges", 0))
        c4.metric("Inferred", stats.get("inferred_edges", 0))
        c5.metric("Communities", stats.get("communities", 0))

    # Interactive graph
    st.subheader("Interactive Knowledge Graph")
    components.html(st.session_state.graph_html, height=graph_height + 50, scrolling=True)

    # Triplets table
    with st.expander("View Extracted Triplets", expanded=False):
        if st.session_state.triplets:
            display_data = []
            for t in st.session_state.triplets:
                display_data.append({
                    "Subject": t.get("subject", ""),
                    "Predicate": t.get("predicate", ""),
                    "Object": t.get("object", ""),
                    "Type": "Inferred" if t.get("inferred") else "Extracted",
                })
            st.dataframe(display_data, use_container_width=True, hide_index=True)

    # Download HTML
    st.download_button(
        label="Download Graph as HTML",
        data=st.session_state.graph_html,
        file_name="knowledge_graph.html",
        mime="text/html",
    )
