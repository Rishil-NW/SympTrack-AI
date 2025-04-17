from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.output_parsers import StrOutputParser
from langchain_community.graphs import Neo4jGraph
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Neo4jVector
from langchain_community.document_loaders import WebBaseLoader
import os

# Set environment variables
os.environ["OPENAI_API_KEY"] = "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"
os.environ["NEO4J_URI"] = "neo4j+s://4d0c8e88.databases.neo4j.io"
os.environ["NEO4J_USERNAME"] = "neo4j"
os.environ["NEO4J_PASSWORD"] = "_w90byudOjL8prteZcqs5le1TSksL-YIHaoBf1JfOYo"

# Step 1: Load documents from URLs
urls = [
    "https://www.mayoclinic.org/diseases-conditions/headache/symptoms-causes/syc-20361701",
    "https://www.nhs.uk/conditions/sore-throat/",
    "https://www.webmd.com/cold-and-flu/flu-guide/what-are-flu-symptoms",
    "https://www.drugs.com/mtm/acetaminophen.html",
    "https://medlineplus.gov/symptoms.html"
]

loader = WebBaseLoader(urls)
docs = loader.load()

# Step 2: Create Neo4j vector index
# Neo4j vector store initialization (check docs for updates)
vector_index = Neo4jVector.from_documents(
    documents=docs,
    embedding=OpenAIEmbeddings(),
    search_type="hybrid",
    node_label="Document",
    embedding_node_property="embedding"
)



# LLM setup
llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-0125")

# Lambda to get question
_search_query = RunnableLambda(lambda x: x["question"])

# Function to retrieve structured data
def structured_retriever(question: str) -> str:
    response = graph.query(
        """
        MATCH (s:Symptom)
        WHERE toLower($query) CONTAINS toLower(s.name)
        MATCH (s)-[:INDICATES]->(d:Disease)
        OPTIONAL MATCH (d)-[:TREATED_WITH]->(m:Medication)
        OPTIONAL MATCH (d)-[:PRECAUTION]->(p:Precaution)
        RETURN d.name AS Disease, collect(DISTINCT m.name) AS Medications, collect(DISTINCT p.name) AS Precautions
        """,
        {"query": question}
    )

    result = ""
    for row in response:
        result += f"🦠 **Disease**: {row['Disease']}\n"

        medications = row['Medications']
        if medications:
            result += f"💊 **Medications**: {', '.join(medications)}\n"
        else:
            result += "💊 **Medications**: No medications found.\n"

        precautions = row['Precautions']
        if precautions:
            result += f"⚠️ **Precautions**: {', '.join(precautions)}\n"
        else:
            result += "⚠️ **Precautions**: No precautions found.\n"

        result += "\n"

    return result.strip() if result else "No structured data found for your symptoms."

# ✅ Fixed function to retrieve unstructured + structured data
def retriever(question: str):
    print(f"Search query: {question}")
    structured_data = structured_retriever(question)
    unstructured_data = [el.page_content for el in vector_index.similarity_search(question)]

    # 🔧 Fix f-string issue by separating the expression
    unstructured_text = "\n\n#Document\n".join(unstructured_data)

    final_data = f"""Structured data:\n{structured_data}\n\nUnstructured data:\n{unstructured_text}"""
    return final_data

# Final chain to generate answer
def answerquery(question: str):
    template = """Answer the question based only on the following context:
{context}

Question: {question}
Use natural language and be concise.
Answer:"""

    prompt = ChatPromptTemplate.from_template(template)

    chain = (
        RunnableParallel(
            {
                "context": _search_query | retriever,
                "question": RunnablePassthrough(),
            }
        )
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain.invoke({"question": question})
