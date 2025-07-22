# Custom Q&A RAG Bot Setup Guide

## Overview
This is a complete implementation of a Retrieval-Augmented Generation (RAG) chatbot that can answer questions from your custom documents using OpenAI GPT-4 and FAISS vector database.

## Features
- 📄 **Multi-format support**: PDF and TXT files
- 🔍 **Semantic search**: Uses OpenAI embeddings for document retrieval
- 🤖 **GPT-4 powered**: Advanced reasoning and natural language responses
- 💾 **Vector database**: Fast similarity search with FAISS
- 🌐 **Web interface**: User-friendly Streamlit UI
- 🖥️ **CLI mode**: Command-line interface for automation
- 📚 **Source citations**: Shows which documents were used for answers

## Installation

### 1. Clone and Setup
```bash
# Create project directory
mkdir rag-qa-bot
cd rag-qa-bot

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Get OpenAI API Key
1. Go to [OpenAI API Keys](https://platform.openai.com/api-keys)
2. Create a new API key
3. Copy the key (you'll enter it in the app)

### 3. Prepare Your Documents
Create a `documents/` folder and add your files:
```
documents/
├── company_handbook.pdf
├── product_specs.pdf
├── meeting_notes.txt
└── research_paper.pdf
```

## Usage

### Web Interface (Recommended)
```bash
streamlit run rag_bot.py
```

Then:
1. **Enter API Key**: Paste your OpenAI API key in the sidebar
2. **Upload Documents**: Use the file uploader to select your PDFs/TXT files
3. **Process Documents**: Click "Process Documents" to create the vector database
4. **Ask Questions**: Type questions about your documents and get AI-powered answers

### Command Line Interface
```bash
python rag_bot.py cli
```

Follow the prompts to:
1. Enter your OpenAI API key
2. Specify the documents directory path
3. Ask questions interactively

## Configuration Options

### Document Processing
- **Chunk Size** (500-2000): Size of text chunks for processing
- **Chunk Overlap** (50-500): Overlap between chunks for context preservation
- **Retrieved Documents** (2-10): Number of relevant chunks to retrieve per query

### Optimal Settings
- **For technical documents**: Chunk Size: 1000, Overlap: 200
- **For narrative content**: Chunk Size: 1500, Overlap: 300
- **For short Q&A**: Chunk Size: 500, Overlap: 100

## Example Questions

Once your documents are processed, try these types of questions:

### Specific Information
- "What is the company's vacation policy?"
- "How do I configure the database connection?"
- "What were the key findings in the research?"

### Analytical Questions
- "Summarize the main points from the meeting notes"
- "Compare the different product features mentioned"
- "What are the security requirements?"

### Complex Queries
- "Based on the documents, what should I do if the system fails?"
- "Explain the step-by-step process for onboarding new employees"
- "What are the pros and cons of each approach discussed?"

## Advanced Features

### Save/Load Vector Database
```python
# Save for future use
rag_bot.save_vectorstore("./vectorstore")

# Load previously created database
rag_bot.load_vectorstore("./vectorstore")
```

### Custom Prompt Templates
Modify the prompt template in the `create_qa_chain()` method to customize response style:

```python
prompt_template = """
You are a helpful assistant that answers questions based on the provided context.
Be concise and accurate. If the answer isn't in the context, say so clearly.

Context: {context}
Question: {question}
Answer:
"""
```

## Troubleshooting

### Common Issues

**1. API Key Error**
```
Error: Incorrect API key provided
```
- Verify your OpenAI API key is correct
- Check you have sufficient credits in your OpenAI account

**2. PDF Loading Issues**
```
Error loading document.pdf: PdfReadError
```
- Try using a different PDF (some PDFs are image-based)
- Consider using OCR tools to extract text first

**3. Memory Issues**
```
Error: Out of memory
```
- Reduce chunk size (e.g., 500 instead of 1000)
- Process fewer documents at once
- Use `faiss-cpu` instead of `faiss-gpu` if on limited hardware

**4. Slow Performance**
- Reduce the number of retrieved documents (k parameter)
- Use smaller chunk sizes
- Consider using `text-embedding-ada-002` instead of `text-embedding-3-small`

### Performance Optimization

**For Large Document Collections:**
```python
# Use hierarchical retrieval
retriever = vectorstore.as_retriever(
    search_type="mmr",  # Maximum Marginal Relevance
    search_kwargs={
        "k": 4,
        "lambda_mult": 0.5
    }
)
```

**For Faster Response Times:**
```python
# Use GPT-3.5 instead of GPT-4
llm = ChatOpenAI(
    temperature=0.1,
    model="gpt-3.5-turbo",
    openai_api_key=openai_api_key
)
```

## Cost Estimation

### OpenAI API Costs (approximate)
- **Embeddings**: ~$0.0001 per 1K tokens
- **GPT-4 queries**: ~$0.03 per 1K tokens (input) + $0.06 per 1K tokens (output)
- **GPT-3.5 queries**: ~$0.001 per 1K tokens (input) + $0.002 per 1K tokens (output)

**Example**: Processing 100 pages of documents + 50 questions ≈ $2-5 with GPT-4

## Security Best Practices

1. **API Key Protection**: Never commit API keys to version control
2. **Environment Variables**: Use `.env` files for sensitive data
3. **Data Privacy**: Be mindful of uploading sensitive documents
4. **Access Control**: Implement user authentication for production use

## Extensions and Improvements

### Add More Document Types
```python
# Word documents
from langchain.document_loaders import Docx2txtLoader

# Web pages
from langchain.document_loaders import WebBaseLoader

# Notion pages
from langchain.document_loaders import NotionDBLoader
```

### Add Conversation Memory
```python
from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)
```

### Add Metadata Filtering
```python
# Filter by document type, date, etc.
retriever = vectorstore.as_retriever(
    search_kwargs={
        "filter": {"source": "important_doc.pdf"},
        "k": 4
    }
)
```

## Support and Contributing

- **Issues**: Report bugs and feature requests
- **Documentation**: Improve setup instructions
- **Features**: Add support for more document types
- **Performance**: Optimize for larger document collections

---

**Happy questioning! 🚀**