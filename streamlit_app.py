"""AI-Powered Knowledge Graph Generator - Desktop Application.

Inspired by ISC SANS Diary #32712 and Robert McDermott's ai-knowledge-graph project.
Takes unstructured text and generates interactive knowledge graphs using LLMs.
"""

from __future__ import annotations

import streamlit as st
import streamlit.components.v1 as components
import requests

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

# -- Demo data for offline/no-LLM usage --
DEMO_TRIPLETS = [
    {"subject": "APT28", "predicate": "is also known as", "object": "Fancy Bear"},
    {"subject": "APT28", "predicate": "attributed to", "object": "Russian GRU"},
    {"subject": "APT28", "predicate": "targets", "object": "Western logistics entities"},
    {"subject": "APT28", "predicate": "targets", "object": "Defense organizations"},
    {"subject": "APT28", "predicate": "targets", "object": "Energy research institutions"},
    {"subject": "APT28", "predicate": "uses", "object": "Spear phishing"},
    {"subject": "APT28", "predicate": "uses", "object": "Zero-day exploits"},
    {"subject": "APT28", "predicate": "exploits", "object": "CVE-2023-23397"},
    {"subject": "Russian GRU", "predicate": "is part of", "object": "Russian military intelligence"},
    {"subject": "Russian GRU", "predicate": "sponsors", "object": "APT29"},
    {"subject": "APT29", "predicate": "is also known as", "object": "Cozy Bear"},
    {"subject": "APT29", "predicate": "targets", "object": "Government agencies"},
    {"subject": "APT29", "predicate": "uses", "object": "Supply chain attacks"},
    {"subject": "APT29", "predicate": "linked to", "object": "SolarWinds breach"},
    {"subject": "SolarWinds breach", "predicate": "affected", "object": "US federal agencies"},
    {"subject": "SolarWinds breach", "predicate": "occurred in", "object": "2020"},
    {"subject": "CISA", "predicate": "published advisory on", "object": "Russian GRU"},
    {"subject": "CISA", "predicate": "warns about", "object": "APT28"},
    {"subject": "CISA", "predicate": "protects", "object": "US critical infrastructure"},
    {"subject": "Spear phishing", "predicate": "is a type of", "object": "Social engineering"},
    {"subject": "Zero-day exploits", "predicate": "target", "object": "Unpatched software"},
    {"subject": "NATO", "predicate": "targeted by", "object": "APT28"},
    {"subject": "NATO", "predicate": "collaborates with", "object": "CISA"},
    {"subject": "Defense organizations", "predicate": "part of", "object": "NATO"},
    {"subject": "Mandiant", "predicate": "tracks", "object": "APT28"},
    {"subject": "Mandiant", "predicate": "tracks", "object": "APT29"},
    {"subject": "Mandiant", "predicate": "is owned by", "object": "Google"},
    {"subject": "Microsoft", "predicate": "reported", "object": "CVE-2023-23397"},
    {"subject": "Microsoft", "predicate": "patches", "object": "Unpatched software"},
    {"subject": "Knowledge Graph", "predicate": "visualizes", "object": "Entity relationships"},
    {"subject": "Knowledge Graph", "predicate": "uses", "object": "SPO triplets"},
    {"subject": "Knowledge Graph", "predicate": "enables", "object": "Threat intelligence analysis"},
    {"subject": "Threat intelligence analysis", "predicate": "helps", "object": "Security analysts"},
    {"subject": "Security analysts", "predicate": "use", "object": "Visual link analysis"},
    {"subject": "Visual link analysis", "predicate": "implemented by", "object": "Maltego"},
    {"subject": "Visual link analysis", "predicate": "implemented by", "object": "PyVis"},
]


def test_llm_connection(base_url: str, api_key: str = "") -> tuple[bool, str]:
    """Test if the LLM endpoint is reachable."""
    url = base_url.rstrip("/")
    # Try the models endpoint for a lightweight check
    if "/v1" in url:
        test_url = url.split("/v1")[0] + "/v1/models"
    else:
        test_url = url.rstrip("/") + "/v1/models"

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    try:
        resp = requests.get(test_url, headers=headers, timeout=5)
        if resp.status_code == 200:
            return True, "Connected!"
        return False, f"HTTP {resp.status_code}"
    except requests.ConnectionError:
        return False, "Connection refused - is the LLM server running?"
    except requests.Timeout:
        return False, "Connection timed out"
    except Exception as e:
        return False, str(e)


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

    provider = st.selectbox(
        "Provider",
        ["OpenRouter", "Kimi (Moonshot)", "OpenAI", "Ollama (local)", "Custom endpoint"],
        help="Select your LLM provider",
    )

    # -- Model catalogs per provider --
    OPENROUTER_MODELS = [
        "meta-llama/llama-4-maverick:free",
        "meta-llama/llama-4-scout:free",
        "deepseek/deepseek-chat-v3-0324:free",
        "deepseek/deepseek-r1-zero:free",
        "google/gemini-2.5-pro-exp-03-25:free",
        "mistralai/mistral-small-3.1-24b-instruct:free",
        "nvidia/llama-3.1-nemotron-nano-8b-v1:free",
        "nousresearch/deephermes-3-llama-3-8b-preview:free",
        "qwen/qwen2.5-vl-3b-instruct:free",
    ]
    KIMI_MODELS = [
        "moonshot-v1-8k",
        "moonshot-v1-32k",
        "moonshot-v1-128k",
    ]
    OPENAI_MODELS = [
        "gpt-4o-mini",
        "gpt-4o",
        "gpt-4-turbo",
        "gpt-3.5-turbo",
    ]
    OLLAMA_MODELS = [
        "gemma3",
        "llama3",
        "mistral",
        "deepseek-r1",
        "qwen2.5",
        "phi3",
    ]

    if provider == "OpenRouter":
        base_url = "https://openrouter.ai/api/v1"
        model = st.selectbox("Model", OPENROUTER_MODELS)
        api_key = st.text_input("API Key", type="password", help="Get yours at openrouter.ai/keys")
    elif provider == "Kimi (Moonshot)":
        base_url = "https://api.moonshot.cn/v1"
        model = st.selectbox("Model", KIMI_MODELS)
        api_key = st.text_input("API Key", type="password", help="Get yours at platform.moonshot.cn")
    elif provider == "OpenAI":
        base_url = "https://api.openai.com/v1"
        model = st.selectbox("Model", OPENAI_MODELS)
        api_key = st.text_input("API Key", type="password", help="Required for OpenAI")
    elif provider == "Ollama (local)":
        base_url = st.text_input("API Base URL", value="http://localhost:11434/v1")
        model = st.selectbox("Model", OLLAMA_MODELS, help="ollama pull <model>")
        api_key = ""
    else:
        base_url = st.text_input("API Base URL")
        model = st.text_input("Model")
        api_key = st.text_input("API Key (optional)", type="password")

    # Connection test button
    if st.button("Test Connection"):
        ok, msg = test_llm_connection(base_url, api_key)
        if ok:
            st.success(f"✅ {msg}")
        else:
            st.error(f"❌ {msg}")

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
tab_paste, tab_upload, tab_demo = st.tabs(["Paste Text", "Upload File", "Demo (No LLM)"])

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

with tab_demo:
    st.info(
        "**Demo Mode** — LLM server not required!\n\n"
        "This uses pre-built APT threat intelligence data to demonstrate "
        "the knowledge graph visualization. Click the button below to see it in action."
    )
    demo_generate = st.button("Generate Demo Graph", type="secondary")

# Determine which text to use
text = input_text if "input_text" in dir() and input_text else ""

# Generate button
col1, col2 = st.columns([1, 4])
with col1:
    generate = st.button("Generate Graph", type="primary", disabled=not text.strip())
with col2:
    if not text.strip():
        st.info("Paste text, upload a file, or try the **Demo** tab to get started.")

# Session state for results
if "triplets" not in st.session_state:
    st.session_state.triplets = []
if "graph_html" not in st.session_state:
    st.session_state.graph_html = ""
if "stats" not in st.session_state:
    st.session_state.stats = {}

# -- Demo mode processing --
if demo_generate:
    with st.status("Generating demo graph...", expanded=True) as status:
        st.write("Loading APT threat intelligence demo data...")
        all_triplets = list(DEMO_TRIPLETS)
        st.write(f"Loaded **{len(all_triplets)}** triplets.")

        st.write("Building knowledge graph...")
        G = build_graph(all_triplets)
        communities = detect_communities(G)

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

        status.update(label="Demo graph ready!", state="complete")

# -- LLM-powered processing --
if generate and text.strip():
    # Pre-flight connection check
    ok, msg = test_llm_connection(base_url, api_key)
    if not ok:
        st.error(
            f"**LLM connection failed:** {msg}\n\n"
            "Please check your setup:\n"
            "- **Ollama**: Install from [ollama.com](https://ollama.com), then run `ollama serve` and `ollama pull gemma3`\n"
            "- **OpenAI**: Select 'OpenAI' in sidebar and enter your API key\n"
            "- **Demo**: Try the 'Demo' tab to see the app without an LLM"
        )
    else:
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

            if not all_triplets:
                st.error("No triplets could be extracted. Check your LLM model and try again.")
                status.update(label="Extraction failed", state="error")
            else:
                # Step 3: Entity standardization
                if do_standardize:
                    st.write("Standardizing entity names...")
                    try:
                        all_triplets = standardize_entities(all_triplets, base_url, model, api_key)
                        st.write("Entity standardization complete.")
                    except Exception as e:
                        st.warning(f"Standardization failed (continuing without): {e}")

                # Step 4: Build graph
                st.write("Building knowledge graph...")
                G = build_graph(all_triplets)
                communities = detect_communities(G)

                # Step 5: Relationship inference
                if do_inference and len(set(communities.values())) > 1:
                    st.write("Inferring relationships between communities...")
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
                        st.warning(f"Inference failed (continuing without): {e}")

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
