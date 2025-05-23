## **Key Features:**

✅ **LangGraph Workflows**: 
- **Indexing**: chunk → embed → store
- **Extraction**: Single-node LLM-based extraction  
- **Answering**: retrieve → generate

✅ **Environment Variables**: Reads `OPENAI_BASE_URL`, `EMBEDDING_MODEL`, `CHAT_MODEL`, `OPENAI_API_KEY`

✅ **Chroma Integration**: Separate collections per company (`docs_{db_identifier}`)

✅ **Simple Chunking**: 1000 chars with 200 char overlap

✅ **Confidence Scoring**: Based on similarity scores for answers, LLM parsing success for extraction

## **Required Dependencies:**
```bash
uv add langgraph langchain-openai chromadb
uv run <your_file.py>
```

## **Environment Variables Needed:**
```bash
OPENAI_API_KEY=your_api_key
OPENAI_BASE_URL=https://your-provider.com/v1  # Optional, defaults to OpenAI
EMBEDDING_MODEL=text-embedding-ada-002        # Optional
CHAT_MODEL=gpt-3.5-turbo                     # Optional
```

## **Usage Example:**
```python
# Initialize
processor = DocumentProcessor()

# Index document
chunk_ids = processor.index_document("Document content here", "company_123")

# Extract questions  
result = processor.extract_questions("Document content here")

# Answer questions
answer = processor.answer_question("What is the main topic?", "company_123")
```
