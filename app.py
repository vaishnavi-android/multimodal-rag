
import requests
import streamlit as st


API_URL = "http://127.0.0.1:8000/query"


# ---------------------------------------------------------
# Page configuration
# ---------------------------------------------------------

st.set_page_config(
    page_title="Great-INDIAN-Query-RAG",
    page_icon="🧠",
    layout="centered",
)


# ---------------------------------------------------------
# Header
# ---------------------------------------------------------

st.title("🧠 Great-INDIA-Query-rag")

st.caption(
    "Ask questions about your knowledge base. "
    "Answers are generated only from relevant knowledge-base evidence."
)


# ---------------------------------------------------------
# Question input
# ---------------------------------------------------------

st.subheader("Ask from the Knowledge Base")

query = st.text_input(
    "Question",
    placeholder="Type your question here...",
)


# ---------------------------------------------------------
# Ask button
# ---------------------------------------------------------

if st.button("🔍 Ask ", use_container_width=True):

    if not query.strip():

        st.warning("Please enter a question.")

    else:

        with st.spinner("Searching the knowledge base..."):

            try:

                response = requests.post(
                    API_URL,
                    json={"query": query},
                    timeout=120,
                )

                if response.status_code == 200:

                    result = response.json()

                    answer = result.get(
                        "answer",
                        "No answer found.",
                    )

                    sources = result.get(
                        "sources",
                        [],
                    )

                    # -------------------------------------------------
                    # ANSWER
                    # -------------------------------------------------

                    st.subheader("Answer")

                    st.write(answer)


                    # -------------------------------------------------
                    # SOURCES
                    # -------------------------------------------------

                    if sources:

                        st.subheader("Sources")

                        st.caption(
                            "Relevant knowledge-base evidence used for this answer:"
                        )

                        for source in sources:

                            file_name = source.get(
                                "file_name",
                                "Unknown file",
                            )

                            bucket_id = source.get(
                                "bucket_id",
                                "Unknown bucket",
                            )

                            page_number = source.get(
                                "page_number"
                            )

                            # -----------------------------------------
                            # Determine file icon
                            # -----------------------------------------

                            extension = (
                                file_name
                                .lower()
                                .rsplit(".", 1)[-1]
                                if "." in file_name
                                else ""
                            )

                            if extension == "pdf":

                                icon = "📄"

                            elif extension in {
                                "png",
                                "jpg",
                                "jpeg",
                                "webp",
                                "gif",
                            }:

                                icon = "🖼️"

                            else:

                                icon = "📁"


                            # -----------------------------------------
                            # Source display
                            # -----------------------------------------

                            source_text = (
                                f"{icon} **{file_name}**"
                            )

                            if page_number is not None:

                                source_text += (
                                    f" — Page {page_number}"
                                )

                            source_text += (
                                f" — `{bucket_id}`"
                            )

                            st.markdown(
                                source_text
                            )

                    else:

                        st.info(
                            "No relevant knowledge-base sources were found."
                        )


                else:

                    st.error(
                        "The RAG API returned an error."
                    )


            except requests.exceptions.ConnectionError:

                st.error(
                    "Could not connect to the RAG API. "
                    "Make sure the FastAPI server is running."
                )


            except requests.exceptions.Timeout:

                st.error(
                    "The request took too long. "
                    "Please try again."
                )


            except Exception as e:

                st.error(
                    f"Unexpected error: {e}"
                )


# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------

st.divider()

st.caption(
    "Multimodal RAG • Answers are grounded exclusively "
    "in the indexed knowledge base."
)
