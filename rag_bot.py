#!/usr/bin/env python3
"""
Error-Free Custom Q&A Bot using RAG (Retrieval-Augmented Generation)
Bulletproof implementation with comprehensive error handling
"""

import os
import sys
import streamlit as st
from typing import List, Optional, Dict, Any
import tempfile
import traceback
from pathlib import Path
import logging

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")
logging.getLogger().setLevel(logging.ERROR)

def install_missing_packages():
    """Auto-install missing packages."""
    required_packages = [
        'openai==1.3.7',
        'langchain==0.1.0',
        'pypdf2==3.0.1',
        'faiss-cpu==1.7.4',
        'tiktoken==0.5.2'
    ]
    
    for package in required_packages:
        try:
            package_name = package.split('==')[0]
            __import__(package_name.replace('-', '_'))
        except ImportError:
            try:
                import subprocess
                subprocess.check_call([sys.executable, '-m', 'pip', 'install', package, '--quiet'])
            except:
                pass

# Auto-install packages
install_missing_packages()

# Import with fallbacks
try:
    import openai
    from openai import OpenAI
except ImportError:
    st.error("OpenAI package not available. Please install: pip install openai")
    st.stop()

try:
    from langchain.document_loaders import PyPDFLoader, TextLoader
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain.embeddings.openai import OpenAIEmbeddings
    from langchain.vectorstores import FAISS
    from langchain.chains import RetrievalQA
    from langchain.chat_models import ChatOpenAI
    from langchain.prompts import PromptTemplate
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

class SafeRAGBot:
    def __init__(self, openai_api_key: str):
        """Initialize RAG bot with maximum error protection."""
        self.openai_api_key = openai_api_key
        self.client = None
        self.embeddings = None
        self.llm = None
        self.vectorstore = None
        self.qa_chain = None
        self.documents = []
        self.chunks = []
        self.error_log = []
        
        self._initialize_safely()
    
    def _initialize_safely(self):
        """Safe initialization with error handling."""
        try:
            # Set environment
            os.environ["OPENAI_API_KEY"] = self.openai_api_key
            
            # Initialize OpenAI client
            self.client = OpenAI(api_key=self.openai_api_key)
            
            # Test API key
            try:
                self.client.models.list()
                st.success("OpenAI API key validated")
            except Exception as e:
                st.error("Invalid OpenAI API key")
                return False
            
            if LANGCHAIN_AVAILABLE:
                try:
                    self.embeddings = OpenAIEmbeddings(
                        openai_api_key=self.openai_api_key,
                        max_retries=3
                    )
                    
                    self.llm = ChatOpenAI(
                        temperature=0.1,
                        model="gpt-3.5-turbo",  # More reliable than GPT-4
                        openai_api_key=self.openai_api_key,
                        max_retries=3,
                        request_timeout=60
                    )
                    st.success("LangChain components initialized")
                except Exception as e:
                    st.warning("LangChain initialization issue, using fallback")
                    self._setup_fallback_mode()
            else:
                self._setup_fallback_mode()
                
        except Exception as e:
            st.error(f"Initialization error: {str(e)}")
            self._setup_fallback_mode()
    
    def _setup_fallback_mode(self):
        """Setup fallback mode without LangChain."""
        st.info("Running in fallback mode (Direct OpenAI API)")
        self.fallback_mode = True
        self.document_texts = []
        
    def safe_load_pdf(self, file_path: str) -> List[str]:
        """Safely load PDF with multiple fallback methods."""
        texts = []
        
        # Method 1: PyPDF2
        try:
            import PyPDF2
            with open(file_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                for page in reader.pages:
                    text = page.extract_text()
                    if text.strip():
                        texts.append(text)
            if texts:
                return texts
        except Exception:
            pass
        
    
        try:
            import pdfplumber
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    text = page.extract_text()
                    if text and text.strip():
                        texts.append(text)
            if texts:
                return texts
        except Exception:
            pass
        
    
        try:
            with open(file_path, 'rb') as file:
                content = file.read()
                # Simple text extraction (very basic)
                text = content.decode('utf-8', errors='ignore')
                if text.strip():
                    texts.append(text)
        except Exception:
            pass
        
        return texts
    
    def safe_load_text(self, file_path: str) -> List[str]:
        """Safely load text files with encoding detection."""
        texts = []
        encodings = ['utf-8', 'utf-16', 'latin-1', 'cp1252', 'ascii']
        
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as file:
                    content = file.read()
                    if content.strip():
                        texts.append(content)
                        break
            except Exception:
                continue
                
        return texts
    
    def load_documents(self, file_paths: List[str]) -> bool:
        """Load documents with comprehensive error handling."""
        all_texts = []
        successful_files = []
        
        for file_path in file_paths:
            try:
                file_extension = Path(file_path).suffix.lower()
                filename = Path(file_path).name
                
                texts = []
                if file_extension == '.pdf':
                    texts = self.safe_load_pdf(file_path)
                elif file_extension == '.txt':
                    texts = self.safe_load_text(file_path)
                
                if texts:
                    all_texts.extend(texts)
                    successful_files.append(filename)
                    st.success(f"Loaded: {filename}")
                else:
                    st.warning(f"Could not extract text from: {filename}")
                    
            except Exception as e:
                st.warning(f"Skipped {Path(file_path).name}: {str(e)}")
        
        if all_texts:
            if hasattr(self, 'fallback_mode') and self.fallback_mode:
                self.document_texts = all_texts
            else:
            
                from langchain.schema import Document
                self.documents = [Document(page_content=text) for text in all_texts]
            
            st.success(f"Successfully loaded {len(successful_files)} files")
            return True
        else:
            st.error("No documents could be loaded")
            return False
    
    def safe_split_text(self, text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[str]:
        """Safely split text into chunks."""
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            
            # Try to break at sentence boundary
            if end < len(text):
                # Look for sentence endings
                for punct in ['. ', '! ', '? ', '\n\n', '\n']:
                    last_punct = text.rfind(punct, start, end)
                    if last_punct > start:
                        end = last_punct + len(punct)
                        break
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            start = end - chunk_overlap
            if start >= len(text):
                break
                
        return chunks
    
    def create_embeddings_safe(self, texts: List[str]) -> List[List[float]]:
        """Create embeddings with error handling."""
        embeddings = []
        
        for i, text in enumerate(texts):
            try:
                # Truncate text if too long
                if len(text) > 8000:
                    text = text[:8000]
                
                response = self.client.embeddings.create(
                    model="text-embedding-ada-002",
                    input=text
                )
                embedding = response.data[0].embedding
                embeddings.append(embedding)
                
                # Progress indicator
                if i % 10 == 0:
                    st.progress((i + 1) / len(texts))
                    
            except Exception as e:
                st.warning(f"Skipped embedding for chunk {i+1}")
                # Use zero vector as fallback
                embeddings.append([0.0] * 1536)
        
        return embeddings
    
    def simple_similarity_search(self, query: str, texts: List[str], top_k: int = 4) -> List[str]:
        """Simple text similarity search fallback."""
        query_lower = query.lower()
        scored_texts = []
        
        for text in texts:
            text_lower = text.lower()
            
            # Simple keyword matching score
            query_words = set(query_lower.split())
            text_words = set(text_lower.split())
            common_words = query_words.intersection(text_words)
            
            # Calculate score
            score = len(common_words) / max(len(query_words), 1)
            
            # exact phrase matching
            if query_lower in text_lower:
                score += 0.5
            
            scored_texts.append((score, text))
        
        # Sort by score and return top_k
        scored_texts.sort(reverse=True, key=lambda x: x[0])
        return [text for score, text in scored_texts[:top_k] if score > 0]
    
    def process_documents(self, chunk_size: int = 1000, chunk_overlap: int = 200) -> bool:
        """Process documents with comprehensive error handling."""
        try:
            if hasattr(self, 'fallback_mode') and self.fallback_mode:
                # Fallback mode processing
                all_chunks = []
                for text in self.document_texts:
                    chunks = self.safe_split_text(text, chunk_size, chunk_overlap)
                    all_chunks.extend(chunks)
                
                self.chunks = all_chunks
                st.success(f"Created {len(all_chunks)} text chunks")
                return True
            
            elif LANGCHAIN_AVAILABLE and self.documents:
                # LangChain mode processing
                try:
                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                        separators=["\n\n", "\n", ". ", " ", ""]
                    )
                    
                    chunks = text_splitter.split_documents(self.documents)
                    self.chunks = chunks
                    
                    if FAISS_AVAILABLE and self.embeddings:
                        with st.spinner("Creating vector database..."):
                            self.vectorstore = FAISS.from_documents(chunks, self.embeddings)
                        st.success("Vector database created!")
                    
                    st.success(f"Created {len(chunks)} document chunks")
                    return True
                    
                except Exception as e:
                    st.warning("LangChain processing failed, using fallback")
                    return self._fallback_processing(chunk_size, chunk_overlap)
            
            return False
            
        except Exception as e:
            st.error(f"Processing error: {str(e)}")
            return self._fallback_processing(chunk_size, chunk_overlap)
    
    def _fallback_processing(self, chunk_size: int, chunk_overlap: int) -> bool:
        """Fallback document processing."""
        try:
            all_chunks = []
            for doc in self.documents:
                content = doc.page_content if hasattr(doc, 'page_content') else str(doc)
                chunks = self.safe_split_text(content, chunk_size, chunk_overlap)
                all_chunks.extend(chunks)
            
            self.chunks = all_chunks
            st.success(f"Fallback processing: {len(all_chunks)} chunks created")
            return True
        except Exception:
            return False
    
    def ask_question_safe(self, question: str) -> Dict[str, Any]:
        """Ask question with multiple fallback methods."""
        if not question.strip():
            return {"error": "Please enter a valid question"}
        
        try:
            if hasattr(self, 'vectorstore') and self.vectorstore and not hasattr(self, 'fallback_mode'):
                return self._langchain_query(question)
            else:
                return self._fallback_query(question)
                
        except Exception as e:
            st.error(f"Query error: {str(e)}")
            return self._emergency_fallback_query(question)
    
    def _langchain_query(self, question: str) -> Dict[str, Any]:
        """Query using LangChain."""
        try:
            if not self.qa_chain:
                self._create_qa_chain()
            
            if self.qa_chain:
                result = self.qa_chain({"query": question})
                return {
                    "answer": result.get("result", "No answer generated"),
                    "sources": result.get("source_documents", [])
                }
        except Exception:
            pass
        
        return self._fallback_query(question)
    
    def _create_qa_chain(self):
        """Create QA chain safely."""
        try:
            prompt_template = """
            Based on the following context, answer the question clearly and accurately.
            If the answer is not in the context, say "I don't have enough information to answer that question."

            Context: {context}
            Question: {question}
            Answer:
            """
            
            PROMPT = PromptTemplate(
                template=prompt_template,
                input_variables=["context", "question"]
            )
            
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=self.vectorstore.as_retriever(search_kwargs={"k": 4}),
                chain_type_kwargs={"prompt": PROMPT},
                return_source_documents=True
            )
        except Exception as e:
            st.warning("QA chain creation failed")
            self.qa_chain = None
    
    def _fallback_query(self, question: str) -> Dict[str, Any]:
        """Fallback query using direct OpenAI API."""
        try:
            # Get relevant chunks
            if hasattr(self, 'chunks') and self.chunks:
                if isinstance(self.chunks[0], str):
                    relevant_chunks = self.simple_similarity_search(question, self.chunks)
                else:
                    chunk_texts = [chunk.page_content for chunk in self.chunks]
                    relevant_chunks = self.simple_similarity_search(question, chunk_texts)
            else:
                relevant_chunks = []
            
            # Create context
            context = "\n\n".join(relevant_chunks[:4]) if relevant_chunks else "No relevant context found"
            
            # Query OpenAI
            messages = [
                {
                    "role": "system", 
                    "content": "You are a helpful assistant that answers questions based on provided context. If the context doesn't contain the answer, say so clearly."
                },
                {
                    "role": "user", 
                    "content": f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
                }
            ]
            
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=messages,
                temperature=0.1,
                max_tokens=500
            )
            
            answer = response.choices[0].message.content
            
            return {
                "answer": answer,
                "sources": relevant_chunks[:2],  # Return top 2 sources
                "method": "Direct OpenAI API"
            }
            
        except Exception as e:
            return {"error": f"Query failed: {str(e)}"}
    
    def _emergency_fallback_query(self, question: str) -> Dict[str, Any]:
        """Last resort query method."""
        try:
            # Simple GPT query without context
            messages = [
                {
                    "role": "system",
                    "content": "You are a helpful assistant. Answer the question based on your general knowledge, but mention that you don't have access to the specific documents."
                },
                {
                    "role": "user",
                    "content": question
                }
            ]
            
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=messages,
                temperature=0.3,
                max_tokens=300
            )
            
            answer = response.choices[0].message.content
            disclaimer = "\n\nNote: This answer is based on general knowledge as I couldn't access your documents properly."
            
            return {
                "answer": answer + disclaimer,
                "sources": [],
                "method": "Emergency fallback"
            }
            
        except Exception as e:
            return {
                "answer": f"I apologize, but I'm unable to process your question due to technical issues. Please try again or contact support. Error: {str(e)[:100]}",
                "sources": [],
                "method": "Error response"
            }

def main():
    """Main application with comprehensive error handling."""
    try:
        st.set_page_config(
            page_title="Q&A RAG Bot",
            page_icon="🤖",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        st.title("RAG Q&A Bot")
        st.markdown("**Simple RAG implementation**")
        
        # Sidebar
        with st.sidebar:
            st.header("⚙️ Configuration")
            
            api_key = st.text_input(
                "OpenAI API Key",
                type="password",
                placeholder="sk-...",
                help="Your OpenAI API key (required)"
            )
            
            if not api_key:
                st.warning("Please enter your OpenAI API key")
                st.info("Get your API key from: https://platform.openai.com/api-keys")
                st.stop()
            
            st.markdown("---")
            st.header("⚙️ Settings")
            chunk_size = st.slider("Chunk Size", 500, 2000, 1000, help="Size of text chunks")
            chunk_overlap = st.slider("Chunk Overlap", 50, 400, 200, help="Overlap between chunks")
        
        # Initialize bot
        if 'rag_bot' not in st.session_state:
            with st.spinner("Initializing RAG Bot..."):
                st.session_state.rag_bot = SafeRAGBot(api_key)
        
        bot = st.session_state.rag_bot
        
        # Main interface
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.header("Upload Documents")
            
            uploaded_files = st.file_uploader(
                "Choose your files",
                type=['pdf', 'txt'],
                accept_multiple_files=True,
                help="Upload PDF or TXT files"
            )
            
            if uploaded_files:
                st.info(f"{len(uploaded_files)} files selected")
                
                if st.button("Process Documents", type="primary"):
                    with st.spinner("Processing documents..."):
                        # Save files temporarily
                        temp_files = []
                        for uploaded_file in uploaded_files:
                            try:
                                suffix = f".{uploaded_file.name.split('.')[-1]}"
                                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                                    tmp.write(uploaded_file.getvalue())
                                    temp_files.append(tmp.name)
                            except Exception as e:
                                st.warning(f"Issue with {uploaded_file.name}: {str(e)}")
                        
                        # Process documents
                        if temp_files:
                            success = bot.load_documents(temp_files)
                            if success:
                                processing_success = bot.process_documents(chunk_size, chunk_overlap)
                                if processing_success:
                                    st.session_state.docs_ready = True
                                    st.balloons()
                                    st.success("Documents ready! Ask questions in the next column.")
                        
                        # Cleanup
                        for temp_file in temp_files:
                            try:
                                os.unlink(temp_file)
                            except:
                                pass
        
        with col2:
            st.header("💬 Ask Questions")
            
            if st.session_state.get('docs_ready', False):
                
                # Question input
                question = st.text_input(
                    "Your question:",
                    placeholder="What is this document about?",
                    help="Ask anything about your uploaded documents"
                )
                
                if st.button("🔍 Get Answer") or (question and st.session_state.get('auto_answer', False)):
                    if question.strip():
                        with st.spinner("🤔 Thinking..."):
                            result = bot.ask_question_safe(question)
                        
                        if "error" in result:
                            st.error(f"{result['error']}")
                        else:
                            st.success("Answer:")
                            st.write(result["answer"])
                            
                            # Show method used
                            if "method" in result:
                                st.caption(f"Method: {result['method']}")
                            
                            # Show sources if available
                            if result.get("sources"):
                                with st.expander("Source Context"):
                                    for i, source in enumerate(result["sources"][:3]):
                                        st.markdown(f"**Source {i+1}:**")
                                        source_text = source if isinstance(source, str) else str(source)
                                        preview = source_text[:300] + "..." if len(source_text) > 300 else source_text
                                        st.text(preview)
                                        st.markdown("---")
                            
                            # Add to history
                            if 'chat_history' not in st.session_state:
                                st.session_state.chat_history = []
                            
                            st.session_state.chat_history.append({
                                "question": question,
                                "answer": result["answer"]
                            })
                    else:
                        st.warning("Please enter a question")
                
                # Chat history
                if st.session_state.get('chat_history'):
                    st.markdown("---")
                    st.header("💭 Recent Questions")
                    for i, chat in enumerate(reversed(st.session_state.chat_history[-3:])):
                        with st.expander(f"Q: {chat['question'][:50]}..."):
                            st.markdown(f"**Question:** {chat['question']}")
                            st.markdown(f"**Answer:** {chat['answer']}")
                
            else:
                st.info("😒 Please upload and process documents first")
                
                # Quick test without documents
                st.markdown("---")
                st.subheader("Test without documents")
                test_question = st.text_input("Ask me any general question:", placeholder="What is the difference between Machine Learning and Atificial Intelligence?")
                
                if st.button("🚀 Test Query") and test_question:
                    with st.spinner("Testing..."):
                        result = bot._emergency_fallback_query(test_question)
                        st.write(result["answer"])
        
        # Footer
        st.markdown("---")
        st.markdown(
            "🔧 **Simple RAG Q&A bot Implementation** | Built with Streamlit + OpenAI | "
            "Using multiple seamless fallback methods"
        )
        
        # System status
        with st.expander("🔍 System Status"):
            st.write(f"**LangChain Available:** {'✅' if LANGCHAIN_AVAILABLE else '❌'}")
            st.write(f"**FAISS Available:** {'✅' if FAISS_AVAILABLE else '❌'}")
            st.write(f"**Fallback Mode:** {'✅' if hasattr(bot, 'fallback_mode') else '❌'}")
            st.write(f"**Documents Loaded:** {len(getattr(bot, 'documents', []))}")
            st.write(f"**Chunks Created:** {len(getattr(bot, 'chunks', []))}")
    
    except Exception as e:
        st.error(f"Critical Error: {str(e)}")
        st.error("Please refresh the page and try again.")
        st.code(traceback.format_exc())

if __name__ == "__main__":
    main()
