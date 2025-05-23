from rag_document_processor import DocumentProcessor


def main():
    print("Hello from doc-processor!")
    processor = DocumentProcessor()

    # Index document
    # print("1. Indexed document")
    # chunk_ids = processor.index_document("My Entitled Name is ANDREW", "company_123")
    # print(chunk_ids)

    # Extract questions
    print("2. Extracted questions")
    result = processor.extract_questions("i want to know about the entitled name")
    print(result)

    # Answer questions
    print("3. Answered question")
    answer = processor.answer_question("What is the entitled name?", "company_123")
    print(answer)


if __name__ == "__main__":
    main()
