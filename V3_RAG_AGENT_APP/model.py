from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from dotenv import load_dotenv
load_dotenv(override=True)


def load_llm():
    model = ChatOpenAI(model="gpt-6-luna", model_kwargs={"reasoning_effort": "none"})
    return model


def load_embedding_model():
    embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    dimensions=1024,
)
    return embeddings
    