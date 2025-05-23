import os
import uuid
import json
from typing import List, Tuple, Dict, Optional, TypedDict
import dataclasses
import chromadb.api
from langgraph.graph import StateGraph, END
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
import chromadb
from chromadb.config import Settings


@dataclasses.dataclass
class ExtractionResult:
    """Data class to store question extraction results."""

    questions: List[str]
    confidence_score: float


@dataclasses.dataclass
class AnswerResult:
    """Data class to store RAG answer results."""

    answer: str
    chunk_ids: List[str]
    confidence_score: float


# LangGraph State Classes
class IndexState(TypedDict):
    content: str
    db_identifier: str
    chunks: List[Tuple[str, Dict]]
    embeddings: List[List[float]]
    chunk_ids: List[str]


class ExtractionState(TypedDict):
    content: str
    questions: List[str]
    confidence_score: float


class AnswerState(TypedDict):
    question: str
    db_id: str
    retrieved_chunks: List[Tuple[str, str, float]]
    answer: str
    chunk_ids: List[str]
    confidence_score: float


class DocumentProcessor:
    """
    A utility class for processing documents, extracting questions, and generating answers
    using Retrieval-Augmented Generation (RAG) techniques.

    Features:
    - Document indexing and chunking with database isolation
    - Question extraction from document content
    - Database-scoped answer generation
    """

    def __init__(self, vector_store_config: Dict = {}):
        """
        Initialize the DocumentProcessor.

        Args:
            vector_store_config: Configuration for the vector store
        """
        self.chunk_size = 1000
        self.chunk_overlap = 200

        # Initialize clients and vector store
        ## Initialize OpenAI-compatible clients from environment variables.
        base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        api_key = os.getenv("OPENAI_API_KEY")
        embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-ada-002")
        chat_model = os.getenv("CHAT_MODEL", "gpt-3.5-turbo")

        # print(f"Using OpenAI base URL: {base_url}")
        # print(f"Using OpenAI API key: {api_key}")
        # print(f"Using embedding model: {embedding_model}")
        # print(f"Using chat model: {chat_model}")

        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")

        self.embeddings_client = FastEmbedEmbeddings(
            model_name=embedding_model,
        )

        self.chat_client = ChatOpenAI(
            model=chat_model,
            base_url=base_url,
        )

        self.vector_store = self._initialize_vector_store(vector_store_config)

        # Build LangGraph workflows
        self._build_index_graph()
        self._build_extraction_graph()
        self._build_answer_graph()

    def _initialize_vector_store(self, config: Dict) -> chromadb.api.ClientAPI:
        """Initialize the vector store with given configuration."""
        try:
            # Use persistent storage
            persist_directory = config.get("persist_directory", "./chroma_db") if config else "./chroma_db"

            vector_store = chromadb.PersistentClient(path=persist_directory)
            print(f"Initialized Chroma vector store at {persist_directory}")
            return vector_store
        except Exception as e:
            print(f"Error initializing vector store: {e}")
            raise

    # LangGraph Workflow Builders

    def _build_index_graph(self):
        """Build the document indexing workflow graph."""
        workflow = StateGraph(IndexState)

        workflow.add_node("chunk", self._chunk_node)
        workflow.add_node("embed", self._embed_node)
        workflow.add_node("store", self._store_node)

        workflow.set_entry_point("chunk")
        workflow.add_edge("chunk", "embed")
        workflow.add_edge("embed", "store")
        workflow.add_edge("store", END)

        self.index_graph = workflow.compile()

    def _build_extraction_graph(self):
        """Build the question extraction workflow graph."""
        workflow = StateGraph(ExtractionState)

        workflow.add_node("extract", self._extract_node)

        workflow.set_entry_point("extract")
        workflow.add_edge("extract", END)

        self.extraction_graph = workflow.compile()

    def _build_answer_graph(self):
        """Build the answer generation workflow graph."""
        workflow = StateGraph(AnswerState)

        workflow.add_node("retrieve", self._retrieve_node)
        workflow.add_node("generate", self._generate_node)

        workflow.set_entry_point("retrieve")
        workflow.add_edge("retrieve", "generate")
        workflow.add_edge("generate", END)

        self.answer_graph = workflow.compile()

    # LangGraph Node Functions

    def _chunk_node(self, state: IndexState) -> IndexState:
        """Chunking node for the indexing workflow."""
        content = state["content"]
        db_identifier = state["db_identifier"]

        preprocessed_content = self._preprocess_content(content)
        chunks = self._chunk_content(preprocessed_content, db_identifier)

        state["chunks"] = chunks
        return state

    def _embed_node(self, state: IndexState) -> IndexState:
        """Embedding node for the indexing workflow."""
        chunks = state["chunks"]
        chunk_texts = [chunk[0] for chunk in chunks]

        embeddings = self._generate_embeddings(chunk_texts)
        state["embeddings"] = embeddings
        return state

    def _store_node(self, state: IndexState) -> IndexState:
        """Storage node for the indexing workflow."""
        chunks = state["chunks"]
        embeddings = state["embeddings"]
        db_identifier = state["db_identifier"]

        chunk_ids = self._store_chunks(chunks, embeddings, db_identifier)
        state["chunk_ids"] = chunk_ids
        return state

    def _extract_node(self, state: ExtractionState) -> ExtractionState:
        """Question extraction node."""
        content = state["content"]

        prompt = f"""Extract all questions from the following document content. 
        Return the questions as a JSON array.
        
        Document content:
        {content}
        
        Format your response as:
        {{"questions": ["question1", "question2", ...], "confidence": 0.95}}
        
        If no questions are found, return {{"questions": [], "confidence": 0.0}}
        """

        try:
            response = self.chat_client.invoke(prompt)
            result = json.loads(response.content)

            state["questions"] = result.get("questions", [])
            state["confidence_score"] = result.get("confidence", 0.0)
        except Exception as e:
            print(f"Error in question extraction: {e}")
            state["questions"] = []
            state["confidence_score"] = 0.0

        return state

    def _retrieve_node(self, state: AnswerState) -> AnswerState:
        """Retrieval node for the answer workflow."""
        question = state["question"]
        db_id = state["db_id"]

        retrieved_chunks = self._retrieve_relevant_chunks(question, db_id)
        state["retrieved_chunks"] = retrieved_chunks
        return state

    def _generate_node(self, state: AnswerState) -> AnswerState:
        """Answer generation node."""
        question = state["question"]
        retrieved_chunks = state["retrieved_chunks"]

        if not retrieved_chunks:
            state["answer"] = "No relevant information found."
            state["chunk_ids"] = []
            state["confidence_score"] = 0.0
            return state

        # Prepare context from retrieved chunks
        context = "\n\n".join([chunk[0] for chunk in retrieved_chunks])
        chunk_ids = [chunk[1] for chunk in retrieved_chunks]
        similarity_scores = [chunk[2] for chunk in retrieved_chunks]

        prompt = f"""Based on the following context, answer the question.
        
        Context:
        {context}
        
        Question: {question}
        
        Answer:"""

        try:
            response = self.chat_client.invoke(prompt)
            answer = response.content

            confidence = self._calculate_confidence(similarity_scores)

            state["answer"] = answer
            state["chunk_ids"] = chunk_ids
            state["confidence_score"] = confidence
        except Exception as e:
            print(f"Error in answer generation: {e}")
            state["answer"] = "Error generating answer."
            state["chunk_ids"] = []
            state["confidence_score"] = 0.0

        return state

    # Public Methods

    def index_document(self, file_content: str, db_identifier: str) -> List[str]:
        """
        Process a document by splitting, indexing, and storing embeddings with database isolation.

        Args:
            file_content: Raw content of the text document to be processed
            db_identifier: Database/company identifier for data isolation

        Returns:
            List of chunk IDs generated during indexing

        Raises:
            ValueError: If the document content is empty or format is unsupported
        """
        if not file_content.strip():
            raise ValueError("Document content cannot be empty")

        try:
            initial_state = IndexState(content=file_content, db_identifier=db_identifier, chunks=[], embeddings=[], chunk_ids=[])

            result = self.index_graph.invoke(initial_state)
            return result["chunk_ids"]
        except Exception as e:
            print(f"Error indexing document: {e}")
            raise ValueError(f"Failed to index document: {str(e)}")

    def extract_questions(self, file_content: str) -> ExtractionResult:
        """
        Extract questions from given document content along with confidence score.

        Args:
            file_content: Raw content of the text document to process

        Returns:
            ExtractionResult containing:
                - List of extracted questions
                - Overall confidence score for the extraction

        Raises:
            ValueError: If content is empty or question extraction fails
        """
        if not file_content.strip():
            raise ValueError("Document content cannot be empty")

        try:
            initial_state = ExtractionState(content=file_content, questions=[], confidence_score=0.0)

            result = self.extraction_graph.invoke(initial_state)
            return ExtractionResult(questions=result["questions"], confidence_score=result["confidence_score"])
        except Exception as e:
            print(f"Error extracting questions: {e}")
            raise ValueError(f"Failed to extract questions: {str(e)}")

    def answer_question(self, question: str, db_id: str) -> AnswerResult:
        """
        Generate an answer to a question using RAG from specified database-scoped documents.

        Args:
            question: The question to answer
            db_id: Database/company identifier for document isolation

        Returns:
            AnswerResult containing:
                - Generated answer
                - List of chunk IDs used
                - Confidence score for the answer

        Raises:
            ValueError: If no relevant documents/chunks are found
        """
        if not question.strip():
            raise ValueError("Question cannot be empty")

        try:
            initial_state = AnswerState(
                question=question, db_id=db_id, retrieved_chunks=[], answer="", chunk_ids=[], confidence_score=0.0
            )

            result = self.answer_graph.invoke(initial_state)

            if not result["chunk_ids"]:
                raise ValueError("No relevant documents found for the given question")

            return AnswerResult(
                answer=result["answer"], chunk_ids=result["chunk_ids"], confidence_score=result["confidence_score"]
            )
        except Exception as e:
            print(f"Error answering question: {e}")
            if "No relevant documents found" in str(e):
                raise
            raise ValueError(f"Failed to answer question: {str(e)}")

    # Private Helper Methods

    def _preprocess_content(self, content: str) -> str:
        """Clean and normalize document content."""
        # Basic preprocessing - remove extra whitespace
        return " ".join(content.split())

    def _chunk_content(self, text: str, db_identifier: str) -> List[Tuple[str, Dict]]:
        """
        Split document into chunks with database-aware metadata.

        Returns:
            List of tuples (chunk_text, chunk_metadata)
        """
        chunks = []

        # Simple fixed-size chunking with overlap
        for i in range(0, len(text), self.chunk_size - self.chunk_overlap):
            chunk_text = text[i : i + self.chunk_size]

            if len(chunk_text.strip()) == 0:
                continue

            chunk_id = str(uuid.uuid4())
            metadata = {
                "chunk_id": chunk_id,
                "db_identifier": db_identifier,
                "start_index": i,
                "end_index": min(i + self.chunk_size, len(text)),
            }

            chunks.append((chunk_text, metadata))

        return chunks

    def _generate_embeddings(self, chunks: List[str]) -> List[List[float]]:
        """Generate embeddings for document chunks."""
        try:
            embeddings = self.embeddings_client.embed_documents(chunks)
            return embeddings
        except Exception as e:
            print(f"Error generating embeddings: {e}")
            raise

    def _store_chunks(self, chunks: List[Tuple[str, Dict]], embeddings: List[List[float]], db_identifier: str) -> List[str]:
        """
        Store chunks and embeddings in vector store with database isolation.

        Returns:
            List of generated chunk IDs
        """
        try:
            # Get or create collection for this db_identifier
            collection_name = f"docs_{db_identifier}"
            collection = self.vector_store.get_or_create_collection(
                name=collection_name, metadata={"db_identifier": db_identifier}
            )

            # Prepare data for storage
            chunk_ids = []
            documents = []
            metadatas = []

            for (chunk_text, metadata), embedding in zip(chunks, embeddings):
                chunk_id = metadata["chunk_id"]
                chunk_ids.append(chunk_id)
                documents.append(chunk_text)
                metadatas.append(metadata)

            # Store in collection
            collection.add(ids=chunk_ids, documents=documents, embeddings=embeddings, metadatas=metadatas)

            print(f"Stored {len(chunk_ids)} chunks for db_identifier: {db_identifier}")
            return chunk_ids

        except Exception as e:
            print(f"Error storing chunks: {e}")
            raise

    def _retrieve_relevant_chunks(self, question: str, db_id: str, top_k: int = 3) -> List[Tuple[str, str, float]]:
        """
        Retrieve relevant chunks from specified database scope.

        Returns:
            List of tuples (chunk_text, chunk_id, similarity_score)
        """
        try:
            collection_name = f"docs_{db_id}"
            collection = self.vector_store.get_collection(name=collection_name)

            # Generate embedding for the question
            question_embedding = self.embeddings_client.embed_query(question)

            # Query the collection
            results = collection.query(
                query_embeddings=[question_embedding], n_results=top_k, include=["documents", "distances", "metadatas"]
            )

            # Convert distances to similarity scores (1 - distance)
            retrieved_chunks = []
            if results["documents"] and results["documents"][0]:
                for i, doc in enumerate(results["documents"][0]):
                    chunk_id = results["ids"][0][i]
                    distance = results["distances"][0][i]
                    similarity_score = max(0, 1 - distance)  # Convert distance to similarity

                    retrieved_chunks.append((doc, chunk_id, similarity_score))

            return retrieved_chunks

        except Exception as e:
            print(f"Error retrieving chunks: {e}")
            return []

    def _calculate_confidence(self, similarity_scores: List[float]) -> float:
        """Calculate overall confidence score from similarity scores."""
        if not similarity_scores:
            return 0.0

        # Use average of similarity scores as confidence
        return sum(similarity_scores) / len(similarity_scores)
