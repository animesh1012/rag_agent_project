from langchain.tools import tool
from dotenv import load_dotenv
load_dotenv(override=True)




def create_retriver_tool(retriever, sink=None):
    @tool(response_format="content_and_artifact")
    def retrieve_docs(query: str, k: int = 4):
        """
        Search the knowledge base for information relevant to the user's question.

        Use this tool to retrieve factual information from the knowledge base.
        If the results are insufficient, search again with a refined query.

        Args:
            query: A concise, high-signal search query describing the information needed.
            k: Top k search result

        Returns:
            Relevant document chunks with their content and metadata.
        """
        retrieved_data = retriever.invoke(query)[:k]
        context = []
        for i in retrieved_data:
            context.append(f"Content: {i.page_content}\nMetadata: {i.metadata}")

        # Side channel for the UI: raw documents, independent of how the
        # agent framework forwards tool artifacts
        if sink is not None:
            sink.extend(retrieved_data)

        if not context:
            return "No relevant results found.", retrieved_data

        return "\n\n".join(context), retrieved_data

    return retrieve_docs