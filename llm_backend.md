# DocumentProcessor Utility Class

## Overview

Utility class for document processing, question extraction, and RAG-based answering.

## Public Methods

### `index_document(file_content: str, db_identifier: str) -> List[str]`

- **Description**: Processes a document by splitting, indexing, and storing embeddings
- **Input**:
- `file_content`: the data blob of the text document  
- `db_identifier`: the ddatabase specific identifier (prpably the company id)
- **Output**:
  - List of chunk IDs generated during indexing
- **Errors**:
  - `FileNotFoundError` if document doesn't exist
  - `ValueError` for unsupported formats

### `extract_questions(file_content: str) -> ExtractionResult`

- **Description**: Extracts questions from document with confidence score
- **Input**:
- `file_content`: the data blob of the text document  
- **Output**:
  - `ExtractionResult` object with:
    - `questions`: List of extracted questions
    - `confidence_score`: Overall extraction confidence (0-1)
- **Errors**:
  - `FileNotFoundError` if document doesn't exist
  - `ValueError` if extraction fails

### `answer_question(question: str, db_id: str) -> AnswerResult`

- **Description**: Generates answer using RAG from specified documents
- **Input**:
  - `question`: Question to answer
  - `db_id`: Identifier for relevant document(s)
- **Output**:
  - `AnswerResult` object with:
    - `answer`: Generated answer text
    - `chunk_ids`: IDs of chunks used
    - `confidence_score`: Answer confidence (0-1)
- **Errors**:
  - `ValueError` if no relevant documents found

## Data Structures

### `ExtractionResult`

```python
@dataclasses.dataclass
class ExtractionResult:
    questions: List[str]
    confidence_score: float
```

### `AnswerResult`

```python
@dataclasses.dataclass
class AnswerResult:
    answer: str
    chunk_ids: List[str]
    confidence_score: float
```

## Usage Example

```python
processor = DocumentProcessor()

# Index a document
chunk_ids = processor.index_document("data.txt", "company_id_123")

# Extract questions
extraction = processor.extract_questions("data.txt")

# Get answers
answer = processor.answer_question(
    "What is the main topic?", 
    "company_id_123"
)
```
