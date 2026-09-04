import os
import json
import hashlib
from typing import List, Dict, Any, Literal
from typing_extensions import TypedDict

import pandas as pd
from docx import Document
from pypdf import PdfReader

# LangChain / OpenAI Components
from langchain_core.documents import Document as LC_Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

# LangGraph Components
from langgraph.graph import StateGraph, START, END

DB_DIR = "local_faiss_index"
REGISTRY_FILE = "indexed_files.json"


# =====================================================================
# 1. FILE UTILITIES & PARSER (Document Processing Module)
# =====================================================================
def get_file_hash(file_path: str) -> str:
    hasher = hashlib.md5()
    with open(file_path, 'rb') as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()


def get_indexed_hashes() -> dict:
    if os.path.exists(REGISTRY_FILE):
        with open(REGISTRY_FILE, 'r') as f:
            return json.load(f)
    return {}


def save_indexed_hashes(registry: dict):
    with open(REGISTRY_FILE, 'w') as f:
        json.dump(registry, f, indent=4)


def parse_source_documents(file_paths: List[str]) -> List[LC_Document]:
    raw_documents = []
    for path in file_paths:
        filename = os.path.basename(path)
        if path.endswith('.pdf'):
            reader = PdfReader(path)
            for page_num, page in enumerate(reader.pages):
                text = page.extract_text()
                if text and text.strip():
                    raw_documents.append(LC_Document(
                        page_content=text,
                        metadata={"source": filename, "page": page_num + 1}
                    ))
        elif path.endswith('.docx'):
            doc = Document(path)
            paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
            full_text = "\n\n".join(paragraphs)
            raw_documents.append(LC_Document(
                page_content=full_text,
                metadata={"source": filename}
            ))
        elif path.endswith(('.xlsx', '.xls')):
            excel_file = pd.ExcelFile(path)
            for sheet_name in excel_file.sheet_names:
                df = pd.read_excel(path, sheet_name=sheet_name)
                df = df.dropna(how='all').dropna(axis=1, how='all')
                markdown_table = df.to_markdown(index=False)
                raw_documents.append(LC_Document(
                    page_content=markdown_table,
                    metadata={"source": filename, "sheet": sheet_name}
                ))
        elif path.endswith(".csv"):
            df = pd.read_csv(path)
            df = df.dropna(how="all").dropna(axis=1, how="all")
            markdown_table = df.to_markdown(index=False)
            raw_documents.append(
                LC_Document(
                    page_content=markdown_table, metadata={"source": filename}
                )
            )

        # --- 5. NEW PLAIN TEXT PARSER ---
        elif path.endswith(".txt"):
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
            if text and text.strip():
                raw_documents.append(
                    LC_Document(
                        page_content=text.strip(), metadata={"source": filename}
                    )
                )

    return raw_documents



def get_or_create_vector_store(file_paths: List[str], embeddings_model) -> FAISS:
    vector_store = None
    if os.path.exists(DB_DIR) and os.path.isdir(DB_DIR):
        vector_store = FAISS.load_local(DB_DIR, embeddings_model, allow_dangerous_deserialization=True)

    if not file_paths:
        if vector_store: return vector_store
        raise ValueError("No database exists and no files provided.")

    registry = get_indexed_hashes()
    files_to_process = []

    for path in file_paths:
        filename = os.path.basename(path)
        file_hash = get_file_hash(path)
        if not (filename in registry and registry[filename] == file_hash):
            files_to_process.append(path)
            registry[filename] = file_hash

    if not files_to_process:
        return vector_store

    raw_docs = parse_source_documents(files_to_process)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunked_docs = text_splitter.split_documents(raw_docs)

    if vector_store:
        vector_store.add_documents(chunked_docs)
    else:
        vector_store = FAISS.from_documents(chunked_docs, embeddings_model)

    vector_store.save_local(DB_DIR)
    save_indexed_hashes(registry)
    return vector_store


# =====================================================================
# 2. AGENT DEFINITIONS & STATE
# =====================================================================

# Define structured routing outputs for our Orchestrator
class RouterOutput(BaseModel):
    next_action: Literal["rag_search", "summarize"] = Field(
        description="Choose 'summarize' if the query asks for a broad overview/summary of the document. Choose 'rag_search' for specific data extraction, specific metrics, or deep factual QA."
    )
    reasoning: str = Field(description="Explanation for selecting this execution path.")


class AgentState(TypedDict):
    """Tracks state variables shared globally across all graph agents."""
    file_paths: List[str]
    user_query: str
    next_node: str
    retrieved_context: str
    draft_answer: str
    final_answer: str
    review_status: Literal["passed", "failed"]
    retry_count: int


# Initialize backing LLMs
llm_fast = ChatOpenAI(model="gpt-4o-mini", temperature=0)
llm_structured = ChatOpenAI(model="gpt-4o-mini", temperature=0)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")


# --- AGENT 1: Orchestrator / Supervisor Agent ---
def orchestrator_agent(state: AgentState) -> Dict[str, Any]:
    print("🤖 [Orchestrator Agent] Analyzing routing logic...")

    router_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are an intelligent supervisor orchestrating a multi-agent system. Decide whether the user query requires a full file summary or a precision local search."),
        ("human", "User Query: {user_query}")
    ])

    # Enforce structured selection routing output
    router_chain = router_prompt | llm_structured.with_structured_output(RouterOutput)
    decision = router_chain.invoke({"user_query": state["user_query"]})

    print(f"   ↳ Decision: {decision.next_action.upper()} | Reasoning: {decision.reasoning}")
    return {"next_node": decision.next_action}


# --- AGENT 2: Precision RAG / Search Agent ---
def rag_search_agent(state: AgentState) -> Dict[str, Any]:
    print("🔍 [RAG Search Agent] Querying vector indexes...")
    vector_store = get_or_create_vector_store(state["file_paths"], embeddings)
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})

    docs = retriever.invoke(state["user_query"])
    context = "\n\n".join([
        f"[Source: {d.metadata.get('source')} Page/Sheet: {d.metadata.get('page', d.metadata.get('sheet', 'N/A'))}]: {d.page_content}"
        for d in docs])

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Answer the user question precisely based strictly on the provided chunks. Cite your sources. Context:\n\n{context}"),
        ("human", "{user_query}")
    ])

    response = (qa_prompt | llm_fast).invoke({"context": context, "user_query": state["user_query"]})
    return {"retrieved_context": context, "draft_answer": response.content}


# --- AGENT 3: Document Summarization / Synthesis Agent ---
def summarization_agent(state: AgentState) -> Dict[str, Any]:
    print("📝 [Summarization Agent] Assembling complete file context...")
    # Load all items from vector store store to form an overarching context blueprint
    vector_store = get_or_create_vector_store(state["file_paths"], embeddings)

    # Pull maximum available context limits for summary synthesis
    docs = vector_store.similarity_search("", k=20)
    context = "\n\n".join([d.page_content for d in docs])

    summary_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Generate a clear, professional executive summary of the document contents. Structure your answer with clear headers and bullet points."),
        ("human", "Summarize this material based on this query: {user_query}\n\nDocument Data:\n{context}")
    ])

    response = (summary_prompt | llm_fast).invoke({"context": context, "user_query": state["user_query"]})
    return {"retrieved_context": context, "draft_answer": response.content}


# --- AGENT 4: Guardrail / Reviewer Agent ---
class GuardrailOutput(BaseModel):
    verdict: Literal["passed", "failed"] = Field(
        description="'passed' if response is perfectly grounded in context. 'failed' if any hallucinations, unverified numbers, or omissions are detected.")
    critique: str = Field(description="Constructive revision instructions if failed, otherwise empty.")


def guardrail_reviewer_agent(state: AgentState) -> Dict[str, Any]:
    print("🛡️ [Guardrail Agent] Verifying response alignment...")

    review_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are an audit agent checking a draft answer against original background text chunks. Flag any hallucinated figures or unsourced assumptions."),
        ("human", "Background Context:\n{context}\n\nDraft Answer:\n{draft}"),
    ])

    reviewer_chain = review_prompt | llm_structured.with_structured_output(GuardrailOutput)
    audit = reviewer_chain.invoke({"context": state["retrieved_context"], "draft": state["draft_answer"]})

    current_retries = state.get("retry_count", 0)
    print(f"   ↳ Audit Review: {audit.verdict.upper()} | Notes: {audit.critique}")

    if audit.verdict == "passed" or current_retries >= 2:
        return {"final_answer": state["draft_answer"], "review_status": "passed"}
    else:
        # Tweak query statement with corrective notes to guide re-generation iteration loops
        new_query = f"{state['user_query']} (Correction required: {audit.critique})"
        return {"review_status": "failed", "retry_count": current_retries + 1, "user_query": new_query}


# =====================================================================
# 3. GRAPH COMPOSITION & ROUTING LOGIC
# =====================================================================
def route_from_orchestrator(state: AgentState):
    return state["next_node"]


def route_from_reviewer(state: AgentState):
    if state["review_status"] == "passed":
        return END
    return "rag_search"  # Route back to re-try extraction with guardrail notesConstruct State Graph Layoutworkflow = StateGraph(AgentState)Add Agent Process Blocksworkflow.add_node("orchestrator", orchestrator_agent)workflow.add_node("rag_search", rag_search_agent)workflow.add_node("summarize", summarization_agent)workflow.add_node("reviewer", guardrail_reviewer_agent)Set Graph Interconnections / Workflow Sequenceworkflow.add_edge(START, "orchestrator")workflow.add_conditional_edges("orchestrator",route_from_orchestrator,{"rag_search": "rag_search","summarize": "summarize"})workflow.add_edge("rag_search", "reviewer")workflow.add_edge("summarize", "reviewer")workflow.add_conditional_edges("reviewer",route_from_reviewer,{END: END,"rag_search": "rag_search"})Compile Graph Appagent_pipeline = workflow.compile()=====================================================================4. EXECUTION INTERFACE=====================================================================def ask_multi_agent_rag(file_paths: List[str], user_query: str) -> str:initial_state: AgentState = {"file_paths": file_paths,"user_query": user_query,"next_node": "","retrieved_context": "","draft_answer": "","final_answer": "","review_status": "passed","retry_count": 0}final_output = agent_pipeline.invoke(initial_state)return final_output["final_answer"]--- Local Integration Testing Mock Execution ---if name == "main":# Ensure you set your environment variable: os.environ["OPENAI_API_KEY"] = "sk-..."# Create a dummy run if mock files exist# print(ask_multi_agent_rag(["sample.pdf"], "Summarize the entire document into 3 key takeaways"))pass


# construct a graph layout
workflow = StateGraph(AgentState)

#Add Agent Process Blocks
workflow.add_node("orchestrator", orchestrator_agent)
workflow.add_node("rag_search", rag_search_agent)
workflow.add_node("summarize", summarization_agent)
workflow.add_node("reviewer", guardrail_reviewer_agent)

# Set Graph Interconnections / Workflow Sequence
workflow.add_edge(START, "orchestrator")
workflow.add_conditional_edges(
    "orchestrator",
    route_from_orchestrator,
    {"rag_search": "rag_search",
     "summarize": "summarize"}
)

workflow.add_edge("rag_search", "reviewer")
workflow.add_edge("summarize", "reviewer")

workflow.add_conditional_edges(
    "reviewer",route_from_reviewer,
    {
        END: END,
        "rag_search": "rag_search"}
)

# Compile Graph App
agent_pipeline = workflow.compile()

# 4. EXECUTION INTERFACE
def ask_multi_agent_rag(file_paths: List[str], user_query: str) -> str:
    initial_state: AgentState = {
        "file_paths": file_paths,
        "user_query": user_query,
        "next_node": "",
        "retrieved_context": "",
        "draft_answer": "",
        "final_answer": "",
        "review_status": "passed",
        "retry_count": 0
    }
    final_output = agent_pipeline.invoke(initial_state)
    return final_output["final_answer"]









# How to use it

# files = ["data/quarterly_report.pdf", "data/notes.docx", "data/simple_sheet.xlsx"]
#
# # --- RUN 1 (Processes, embeds, and saves to disk) ---
# res1 = ask_persistent_rag(files, "What is our main objective?")
# print(res1["answer"])
#
# # --- RUN 2 (Bypasses text processing completely, loads from disk instantly) ---
# res2 = ask_persistent_rag(files, "Summarize the column headers in the excel sheet.")
# print(res2["answer"])
