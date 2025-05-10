"""
RAG (Retrieval Augmented Generation) service for Athena.

This module handles document embedding, storage, retrieval, and generation
using vector databases and language models.
"""

import os
from typing import List, Dict, Any, Optional

# Update imports to use new LangChain paths
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.schema import Document
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

from athena.core.config import settings


class RAGService:
    """
    Service for RAG operations including vector embeddings, 
    document retrieval, and answer generation.
    """
    
    def __init__(self):
        """Initialize the RAG service with necessary components."""
        # Initialize OpenAI embedding model
        api_key = settings.OPENAI_API_KEY
        if not api_key:
            raise ValueError("OpenAI API key not found. Please set OPENAI_API_KEY in your .env file.")
            
        self.embeddings = OpenAIEmbeddings(
            model=settings.EMBEDDING_MODEL,
            openai_api_key=api_key
        )
        
        # Initialize vector database
        self._init_vector_db()
        
        # Initialize LLM
        self.llm = ChatOpenAI(
            model=settings.LLM_MODEL,
            openai_api_key=api_key,
            temperature=0.2
        )

    def _init_vector_db(self) -> None:
        """Initialize or load the vector database."""
        # Ensure the vector DB directory exists
        os.makedirs(settings.VECTOR_DB_PATH, exist_ok=True)
        
        # Load or create Chroma vector store
        try:
            self.vector_store = Chroma(
                persist_directory=settings.VECTOR_DB_PATH,
                embedding_function=self.embeddings
            )
            print(f"Vector database loaded from {settings.VECTOR_DB_PATH}")
        except Exception as e:
            print(f"Error loading vector database: {e}")
            print("Creating new vector database")
            self.vector_store = Chroma(
                persist_directory=settings.VECTOR_DB_PATH,
                embedding_function=self.embeddings
            )
    
    async def add_documents(self, documents: List[Dict[str, str]], metadata_list: List[Dict] = None) -> List[str]:
        """
        Add documents to the vector database.
        
        Args:
            documents: List of document dictionaries with 'content' and optional other fields
            metadata_list: Optional list of metadata dictionaries for each document
        
        Returns:
            List of document IDs added to the database
        """
        if metadata_list and len(documents) != len(metadata_list):
            raise ValueError("Number of documents and metadata entries must match")
        
        # Convert to Document objects
        docs = []
        for i, doc in enumerate(documents):
            content = doc.get("content", "")
            meta = metadata_list[i] if metadata_list else {}
            
            # Add any other document fields to metadata
            for k, v in doc.items():
                if k != "content":
                    meta[k] = v
            
            docs.append(Document(page_content=content, metadata=meta))
        
        # Add to vector store
        ids = self.vector_store.add_documents(docs)
        self.vector_store.persist()  # Ensure changes are saved to disk
        return ids
    
    async def retrieve_similar_documents(self, query: str, k: int = 5) -> List[Document]:
        """
        Retrieve documents similar to the query.
        
        Args:
            query: The query string
            k: Number of documents to retrieve
            
        Returns:
            List of similar documents
        """
        return self.vector_store.similarity_search(query, k=k)
    
    async def generate_answer(self, query: str, context_docs: List[Document] = None, k: int = 3) -> Dict[str, Any]:
        """
        Generate an answer for the query using RAG.
        
        Args:
            query: User query
            context_docs: Optional pre-retrieved context documents
            k: Number of documents to retrieve if context_docs not provided
            
        Returns:
            Dictionary with answer and metadata
        """
        # Retrieve relevant documents if not provided
        if not context_docs:
            context_docs = await self.retrieve_similar_documents(query, k=k)
        
        # Format context for the prompt
        contexts = []
        for i, doc in enumerate(context_docs):
            contexts.append(f"Document {i+1}:\n{doc.page_content}\n\n")
        
        context_str = "\n".join(contexts)
        
        # Create a RAG prompt
        template = """你是一个专注于物流行业的智能客服专家。
        请基于以下参考文档回答用户的问题。
        如果参考文档中没有足够的信息，请说明无法确定答案。
        回答应该专业、准确、全面，但也要简洁易懂。
        
        参考文档:
        {context}
        
        用户问题: {question}
        
        请提供你的回答:
        """
        
        # Create the RAG chain
        prompt = ChatPromptTemplate.from_template(template)
        
        rag_chain = (
            {"context": lambda x: context_str, "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )
        
        # Run the chain
        answer = rag_chain.invoke(query)
        
        # Calculate confidence based on document similarities
        # This is a simplified approach
        confidence = 0.7  # Default medium-high confidence
        
        # Check if we have any contexts
        if not context_docs:
            confidence = 0.3  # Low confidence if no context
        
        # Prepare result
        result = {
            "answer": answer,
            "confidence": confidence,
            "needs_human_review": confidence < 0.5,  # Flag for human review if low confidence
            "sources": [
                {
                    "content": doc.page_content[:100] + "..." if len(doc.page_content) > 100 else doc.page_content,
                    "metadata": doc.metadata
                }
                for doc in context_docs
            ]
        }
        
        return result


# Create a singleton instance
rag_service = RAGService()