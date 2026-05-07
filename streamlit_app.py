"""Streamlit interface for the MS MARCO RAG course project."""

from __future__ import annotations

from typing import Any, Dict, List

import streamlit as st

from pipeline import RAGPipeline
from settings import AppSettings
from utils import annotate_results, filter_annotated_results


st.set_page_config(
    page_title="RAG-Based Passage Retrieval and Answer Generation using GPT",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource(show_spinner=True)
def load_pipeline() -> RAGPipeline:
    """
    Loads and caches the RAG pipeline.

    @return: Initialized RAGPipeline instance.
    """

    settings = AppSettings()
    rag_pipeline = RAGPipeline(settings)
    rag_pipeline.load_subset_data()
    rag_pipeline.load_dense_index()
    return rag_pipeline


def _initialize_session_state() -> None:
    """
    Initializes session state variables used by the UI.

    @return: None.
    """

    defaults = {
        "last_query": "",
        "last_response": None,
        "last_filtered_results": [],
        "last_retriever": "hybrid",
        "last_backend": "local",
        "last_generate_answer": False,
        "last_evaluation_summary": None,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _safe_top_k(pipeline: RAGPipeline) -> int:
    """
    Safely gets the configured retrieval top-k value.

    @param pipeline: Active RAG pipeline.
    @return: Retrieval top-k value.
    """

    retrieval_settings = getattr(pipeline.settings, "retrieval", None)
    return int(getattr(retrieval_settings, "top_k", 10))


def _safe_max_context(pipeline: RAGPipeline) -> int:
    """
    Safely gets the configured max context passage count.

    @param pipeline: Active RAG pipeline.
    @return: Max context passage count.
    """

    generation_settings = getattr(pipeline.settings, "generation", None)
    retrieval_settings = getattr(pipeline.settings, "retrieval", None)

    if retrieval_settings is not None and hasattr(retrieval_settings, "max_context_passages"):
        return int(retrieval_settings.max_context_passages)

    if generation_settings is not None and hasattr(generation_settings, "max_context_passages"):
        return int(generation_settings.max_context_passages)

    return 5


def _render_header() -> None:
    """
    Renders the page header.

    @return: None.
    """

    st.title("RAG-Based Passage Retrieval and Answer Generation using GPT")
    st.caption(
        "Dense retrieval with Sentence-BERT + FAISS, BM25 baseline, hybrid reciprocal-rank fusion, "
        "grounded answer generation, and lightweight faceted filtering."
    )


def _render_sidebar(pipeline: RAGPipeline) -> Dict[str, Any]:
    """
    Renders the sidebar controls.

    @param pipeline: Active RAG pipeline.
    @return: Dictionary containing sidebar option values.
    """

    retrieval_top_k = _safe_top_k(pipeline)
    max_context = _safe_max_context(pipeline)
    retriever_options = ["hybrid", "dense", "bm25", "hybrid-rerank"]
    selected_retriever = st.session_state.get("last_retriever", "hybrid")
    retriever_index = retriever_options.index(selected_retriever) if selected_retriever in retriever_options else 0

    with st.sidebar:
        st.header("Search Options")

        retriever_type = st.selectbox(
            "Retriever",
            options=retriever_options,
            index=retriever_index,
            help="Hybrid uses reciprocal-rank fusion. Hybrid-rerank adds a slower cross-encoder reranker.",
        )

        generator_backend = st.selectbox(
            "Generator Backend",
            options=["local", "openai"],
            index=0 if st.session_state["last_backend"] == "local" else 1,
            help="Choose the answer generation backend when grounded generation is enabled.",
        )

        generate_answer = st.checkbox(
            "Generate grounded answer",
            value=st.session_state["last_generate_answer"],
            help="Generate a grounded answer using the top retrieved passages.",
        )

        st.divider()
        st.header("Lightweight Facets")

        length_buckets = st.multiselect(
            "Passage length",
            options=["short", "medium", "long"],
            default=["short", "medium", "long"],
            help="Filter passages by rough length bucket.",
        )

        overlap_buckets = st.multiselect(
            "Lexical overlap with query",
            options=["low", "medium", "high"],
            default=["low", "medium", "high"],
            help="Filter passages by rough lexical overlap with the query.",
        )

        require_digits = st.checkbox(
            "Only passages containing digits",
            value=False,
            help="Keep only passages that contain at least one digit.",
        )

        require_year = st.checkbox(
            "Only passages containing a year",
            value=False,
            help="Keep only passages that contain a likely year such as 1945 or 2020.",
        )

        st.divider()
        st.header("Run Summary")
        subset_summary = pipeline.subset_summary()
        st.markdown(f"**Top-K retrieval:** {retrieval_top_k}")
        st.markdown(f"**Max prompt passages:** {max_context}")
        st.markdown(f"**Subset passages:** {subset_summary['passages']}")
        st.markdown(f"**Judged queries:** {subset_summary['judged_queries']}")
        st.caption(str(subset_summary["limitation"]))

    return {
        "retriever_type": retriever_type,
        "generator_backend": generator_backend,
        "generate_answer": generate_answer,
        "length_buckets": length_buckets,
        "overlap_buckets": overlap_buckets,
        "require_digits": require_digits,
        "require_year": require_year,
    }


def _run_search(
    pipeline: RAGPipeline,
    query: str,
    retriever_type: str,
    generate_answer: bool,
    generator_backend: str,
) -> Dict[str, Any]:
    """
    Runs either retrieval-only search or retrieval plus answer generation.

    @param pipeline: Active RAG pipeline.
    @param query: User query text.
    @param retriever_type: Selected retriever type.
    @param generate_answer: Whether grounded answer generation is enabled.
    @param generator_backend: Selected answer generation backend.
    @return: Search response dictionary.
    """

    if generate_answer:
        return pipeline.answer_query(
            query,
            backend=generator_backend,
            retriever_type=retriever_type,
        )

    return pipeline.search_query(
        query,
        retriever_type=retriever_type,
    )


def _apply_facets(
    query: str,
    results: List[Any],
    length_buckets: List[str],
    require_digits: bool,
    require_year: bool,
    overlap_buckets: List[str],
) -> List[Dict[str, Any]]:
    """
    Annotates results and applies UI facet filters.

    @param query: User query text.
    @param results: Retrieval result objects.
    @param length_buckets: Enabled length buckets.
    @param require_digits: Whether digits are required.
    @param require_year: Whether a year is required.
    @param overlap_buckets: Enabled overlap buckets.
    @return: Filtered annotated results.
    """

    annotated = annotate_results(query, results)
    return filter_annotated_results(
        annotated,
        length_buckets,
        require_digits,
        require_year,
        overlap_buckets,
    )


def _render_status_bar(
    query: str,
    retriever_type: str,
    generate_answer: bool,
    generator_backend: str,
    result_count: int,
) -> None:
    """
    Renders a compact status summary.

    @param query: User query text.
    @param retriever_type: Selected retriever type.
    @param generate_answer: Whether answer generation is enabled.
    @param generator_backend: Selected generation backend.
    @param result_count: Number of filtered results.
    @return: None.
    """

    answer_mode = "on" if generate_answer else "off"
    backend_display = generator_backend if generate_answer else "n/a"

    st.info(
        f"Query: {query} | Retriever: {retriever_type} | Answer generation: {answer_mode} | "
        f"Backend: {backend_display} | Visible results: {result_count}"
    )


def _render_results_panel(filtered_results: List[Dict[str, Any]], top_k: int, retriever_type: str) -> None:
    """
    Renders the left results panel.

    @param filtered_results: Filtered annotated results.
    @param top_k: Retrieval top-k size.
    @param retriever_type: Selected retriever type.
    @return: None.
    """

    st.subheader(f"{retriever_type.upper()} Top-{top_k} Results")

    if not filtered_results:
        st.warning("No retrieved passages match the selected facet filters.")
        return

    for item in filtered_results:
        expander_title = (
            f"Rank {item['rank']} | pid={item['pid']} | "
            f"source={item.get('source', retriever_type)} | score={item['score']:.4f}"
        )
        with st.expander(expander_title, expanded=item["rank"] <= 3):
            st.write(item["text"])
            score_details = item.get("score_details") or {}
            if score_details:
                st.caption(
                    "score details: "
                    + ", ".join(f"{key}={value:.4f}" for key, value in score_details.items())
                )
            st.caption(
                f"length={item['length_bucket']} | "
                f"digits={item['contains_digits']} | "
                f"year={item['contains_year']} | "
                f"overlap={item['overlap_bucket']}"
            )


def _render_answer_panel(response: Dict[str, Any], generate_answer: bool, generator_backend: str) -> None:
    """
    Renders the right answer and prompt panel.

    @param response: Search response dictionary.
    @param generate_answer: Whether answer generation is enabled.
    @param generator_backend: Selected generation backend.
    @return: None.
    """

    st.subheader("Grounded Answer")

    if generate_answer:
        answer_text = response.get("answer")
        if answer_text:
            st.success(f"Generated with backend: {generator_backend}")
            st.write(answer_text)
        else:
            st.warning("Answer generation was enabled, but no answer text was returned.")
    else:
        st.info("Enable answer generation in the sidebar to produce a grounded RAG answer.")

    prompt_text = response.get("prompt")
    if prompt_text:
        with st.expander("Prompt Preview", expanded=False):
            st.code(prompt_text, language="text")


def _render_evaluation_panel(pipeline: RAGPipeline) -> None:
    """
    Renders an optional compact evaluation summary panel.

    @param pipeline: Active RAG pipeline.
    @return: None.
    """

    with st.expander("Dense vs BM25 vs Hybrid Evaluation Summary", expanded=False):
        st.write(
            "Evaluation uses random judged MS MARCO dev queries plus manual project queries. "
            "Manual queries without official qrels are displayed but excluded from mean judged P@10."
        )
        if st.button("Run evaluation summary", use_container_width=False):
            with st.spinner("Running evaluation..."):
                tables = pipeline.run_evaluation()
            st.session_state["last_evaluation_summary"] = tables["comparison_summary"]

        summary = st.session_state.get("last_evaluation_summary")
        if summary is not None:
            st.dataframe(summary, use_container_width=True)


def main() -> None:
    """
    Main Streamlit entry point.

    @return: None.
    """

    _initialize_session_state()
    _render_header()

    try:
        pipeline = load_pipeline()
    except Exception as exc:
        st.error(f"Failed to load pipeline: {exc}")
        st.stop()

    sidebar_options = _render_sidebar(pipeline)

    query = st.text_input(
        "Query",
        value=st.session_state["last_query"],
        placeholder="Enter a question to search the passage subset...",
    )

    search_clicked = st.button("Search", type="primary", use_container_width=False)

    if search_clicked:
        normalized_query = query.strip()
        if not normalized_query:
            st.warning("Please enter a query before searching.")
        else:
            with st.spinner("Running retrieval..."):
                response = _run_search(
                    pipeline=pipeline,
                    query=normalized_query,
                    retriever_type=sidebar_options["retriever_type"],
                    generate_answer=sidebar_options["generate_answer"],
                    generator_backend=sidebar_options["generator_backend"],
                )

            filtered_results = _apply_facets(
                query=normalized_query,
                results=response["results"],
                length_buckets=sidebar_options["length_buckets"],
                require_digits=sidebar_options["require_digits"],
                require_year=sidebar_options["require_year"],
                overlap_buckets=sidebar_options["overlap_buckets"],
            )

            st.session_state["last_query"] = normalized_query
            st.session_state["last_response"] = response
            st.session_state["last_filtered_results"] = filtered_results
            st.session_state["last_retriever"] = sidebar_options["retriever_type"]
            st.session_state["last_backend"] = sidebar_options["generator_backend"]
            st.session_state["last_generate_answer"] = sidebar_options["generate_answer"]

    response = st.session_state.get("last_response")
    filtered_results = st.session_state.get("last_filtered_results", [])

    if response:
        _render_status_bar(
            query=st.session_state["last_query"],
            retriever_type=st.session_state["last_retriever"],
            generate_answer=st.session_state["last_generate_answer"],
            generator_backend=st.session_state["last_backend"],
            result_count=len(filtered_results),
        )

        left_col, right_col = st.columns([2.15, 1.15], gap="large")

        with left_col:
            _render_results_panel(
                filtered_results=filtered_results,
                top_k=_safe_top_k(pipeline),
                retriever_type=st.session_state["last_retriever"],
            )

        with right_col:
            _render_answer_panel(
                response=response,
                generate_answer=st.session_state["last_generate_answer"],
                generator_backend=st.session_state["last_backend"],
            )
    else:
        st.markdown("Enter a query and click **Search** to retrieve ranked passages.")

    _render_evaluation_panel(pipeline)


if __name__ == "__main__":
    main()
