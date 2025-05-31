# Add this at the very top of your main application file (app.py)
import sys
import sqlite3

import os
import sys

# Debugging output
print(f"Python version: {sys.version}")
print(f"Current directory: {os.getcwd()}")
print(f"Files in data dir: {os.listdir('data')}")

# Workaround for pipeline import
try:
    from transformers import pipeline
except ImportError:
    from transformers.pipelines import pipeline
    print("Used fallback pipeline import")

# Check SQLite version and use pysqlite3 if needed
sqlite_version = sqlite3.sqlite_version_info
if sqlite_version < (3, 35, 0):
    try:
        # Replace sqlite3 with pysqlite3
        import pysqlite3 as sqlite3
        sys.modules['sqlite3'] = sqlite3
        print(f"Using pysqlite3-binary (SQLite {sqlite3.sqlite_version})")
    except ImportError:
        print("pysqlite3-binary not installed. Install with: pip install pysqlite3-binary")
        raise

# Now import ChromaDB and other dependencies
import chromadb


import os
import re
import unicodedata
import shutil
import torch
from transformers import pipeline
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import google.generativeai as genai
from tavily import TavilyClient
import streamlit as st

import sys

# Force use of modern SQLite via pysqlite3
try:
    import pysqlite3
    sys.modules["sqlite3"] = pysqlite3
except ImportError:
    pass


# Utility Functions
def initialize_gemini(model_name):
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        raise ValueError("GEMINI_API_KEY is not set.")
    genai.configure(api_key=gemini_api_key)
    generation_config = {
        "temperature": 0.7, "top_p": 0.9, "top_k": 40,
        "max_output_tokens": 4096, "response_mime_type": "text/plain",
    }
    return genai.GenerativeModel(model_name=model_name, generation_config=generation_config)

def remove_non_sinhala(text):
    """Remove characters not in the Sinhala Unicode range or basic Latin punctuation/digits."""
    allowed_ranges = [
        (0x0D80, 0x0DFF),  # Sinhala Unicode range
        (0x0020, 0x007E),  # Basic Latin (punctuation and digits)
    ]
    return ''.join(c for c in text if any(start <= ord(c) <= end for start, end in allowed_ranges))

def clean_response(response_text):
    """Clean the response text by removing unwanted patterns and non-Sinhala characters."""
    processed_response = response_text.replace("මූලාශ්‍ර: /content/context_facts.csv", "")
    processed_response = re.sub(r"\(මූලාශ්‍ර \d+(?:, \d+)*\)", "", processed_response)
    processed_response = unicodedata.normalize('NFC', processed_response).replace('\u200d', '')
    processed_response = remove_non_sinhala(processed_response)
    return processed_response

def setup_vector_store(csv_path, persist_directory, force_rebuild=False):
    os.makedirs(os.path.dirname(persist_directory), exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        model_kwargs={"device": device}
    )
    if os.path.exists(persist_directory) and os.path.isdir(persist_directory) and not force_rebuild:
        try:
            vector_store = Chroma(persist_directory=persist_directory, embedding_function=embedding_model)
            return vector_store, vector_store._collection.count()
        except Exception as e:
            print(f"Error loading vector store: {e}")
            print("Will create a new vector store")
    if not csv_path:
        raise ValueError("CSV file path is required")
    if force_rebuild and os.path.exists(persist_directory):
        shutil.rmtree(persist_directory)
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found at {csv_path}")
    loader = CSVLoader(file_path=csv_path)
    documents = loader.load()
    documents = [doc for doc in documents if doc.page_content.strip()]
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)
    vector_store = Chroma.from_documents(documents=texts, embedding=embedding_model, persist_directory=persist_directory)
    return vector_store, len(texts)

def retrieve_relevant_documents(question, vector_store, k=5):
    try:
        docs = vector_store.similarity_search(question, k=k)
        results = []
        for i, doc in enumerate(docs):
            content = doc.page_content if hasattr(doc, 'page_content') else ""
            if not content.strip():
                continue
            metadata = doc.metadata if hasattr(doc, 'metadata') else {}
            source = metadata.get('source', 'Unknown source')
            results.append({'content': content, 'source': source})
        return results
    except Exception as e:
        print(f"Error retrieving documents: {e}")
        return []

def initialize_search_clients():
    clients = {"available": False}
    tavily_api_key = os.getenv("TAVILY_API_KEY")
    if tavily_api_key:
        try:
            clients["tavily"] = TavilyClient(tavily_api_key)
            clients["available"] = True
        except Exception as e:
            print(f"Error initializing Tavily client: {e}")
    return clients

RECOMMENDED_SITES = [
    "https://hashtaggeneration.org/fact-check/",
    "https://srilanka.factcrescendo.com/",
    "https://www.bbc.com/sinhala",
    "https://sinhala.newsfirst.lk/",
    "https://www.adaderana.lk/",
    "https://www.hirunews.lk/english/",
    "https://www.cbsl.gov.lk/si"
]

# Agent Definitions
class DomainClassificationAgent:
    def __init__(self, gemini_model):
        self.gemini_model = gemini_model

    def classify(self, statement):
        chat_session = self.gemini_model.start_chat()
        prompt = f"""
        පහත සිංහල ප්‍රකාශය කුමන විෂය ක්ෂේත්‍රයට අයත්ද යන්න තීරණය කරන්න: '{statement}'
        විෂය ක්ෂේත්‍ර: politics, economics, health
        ඔබේ තීරණය එක් වචනයකින් පමණක් ලබා දෙන්න් (උදා: politics).
        """
        response = chat_session.send_message(prompt)
        domain = response.text.strip().lower()
        if domain.startswith("තීරණය:"):
            domain = domain.replace("තීරණය:", "").strip()
        return domain

class DataRetrievalAgent:
    def __init__(self, domain, vector_store):
        self.domain = domain
        self.vector_store = vector_store

    def retrieve(self, statement):
        return retrieve_relevant_documents(statement, self.vector_store)

class DecisionAgent:
    def __init__(self, gemini_model):
        self.model = gemini_model

    def decide(self, statement, retrieved_docs):
        chat_session = self.model.start_chat()
        prompt = f"""
        පහත සිංහල ප්‍රකාශය සඳහා ලබා දී ඇති තොරතුරු ප්‍රමාණවත්ද යන්න තීරණය කරන්න: '{statement}'

        ලබා ඇති තොරතුරු: {retrieved_docs}

        තොරතුරු ප්‍රමාණවත් යැයි සැලකෙන්නේ, එම තොරතුරු මගින් ප්‍රකාශයේ සඳහන් කරුණු පිළිබඳ නිශ්චිත තොරතුරු තිබේ නම් පමණි.
        මෙම තොරතුරු  ලබාදී ඇති ප්‍රකාශයේ සත්‍ය අසත්‍ය බව නිගමනය සදහා යොදාගන්නා බැවින් වගකීම් සහගත ලෙස පිළිතුරු ලබාදෙන්න.

        කරුණාකර, ඔබේ පිළිතුරේ පළමු පේළිය ස්පස්ට "නිගමනය: [සත්‍ය/අසත්‍ය/තීරණය කළ නොහැක]" ලෙස ලබා දෙන්න.
        """
        response = chat_session.send_message(prompt)
        response_text = response.text.strip().lower()
        # Search for the first line starting with "නිගමනය:" and use that for verdict extraction.
        for line in response_text.split('\n'):
            line = line.strip()
            if line.startswith("නිගමනය:"):
                verdict_line = line
                break
        else:
            verdict_line = ""
        if re.search(r"නිගමනය:\s*සත්‍?ය", verdict_line):
            return "sufficient"
        else:
            return "insufficient"

class FactAnalysisAgent:
    def __init__(self, gemini_model):
        self.model = gemini_model

    def verify_with_rag(self, claim, vector_store):
        retrieved_docs = retrieve_relevant_documents(claim, vector_store)
        if not retrieved_docs:
            return ("මට කනගාටුයි, ඔබේ ප්‍රකාශය පරීක්ෂා කිරීමට අවශ්‍ය තොරතුරු සොයාගත නොහැකි විය. "
                    "කරුණාකර වෙනත් ප්‍රකාශයක් උත්සාහ කරන්න.")
        chat_session = self.model.start_chat()
        final_prompt = f"""
        ඔබ සිංහල ප්‍රකශ වල සත්‍ය අසත්‍ය බව සාක්ෂි සමග සහතික කරන AI සභායකයෙකි.
        කරුණාකර ඔබේ නිගමනය පළමු පේළිය "නිගමනය: [සත්‍ය/අසත්‍ය/තීරණය කළ නොහැක]" ලෙස ලබා දෙන්න.
        පහත ප්‍රකාශය විමර්ශනය කරන්න: '{claim}'
        ලබා ඇති තොරතුරු: {retrieved_docs}
        කරුණාකර:
        1. පළමු පේළියට පමණක් ඔබේ නිගමනය (උදා: නිගමනය: සත්‍ය)
        2. ඉතිරි පේළි වල ඔබේ විස්තරාත්මක හේතු සඳහන් කරන්න.
        """
        response = chat_session.send_message(final_prompt)
        return clean_response(response.text)

    def verify_with_search(self, claim, search_clients):
        if not search_clients.get("available"):
            return ("මට කනගාටුයි, සෙවුම් යාන්ත්‍රණය භාවිතා කළ නොහැකි බැවින් "
                    "තහවුරු කිරීම් සිදු කළ නොහැක. API යතුරු සැකසීමෙන් පසු නැවත උත්සාහ කරන්න.")
        tavily_client = search_clients.get("tavily")
        if not tavily_client:
            return ("මට කනගාටුයි, Tavily සෙවුම් යාන්ත්‍රණය භාවිතා කළ නොහැකි බැවින් "
                    "තහවුරු කිරීම් සිදු කළ නොහැක.")
        try:
            tavily_results = tavily_client.search(
                query=claim + " Sri Lanka", search_depth="advanced", include_domains=RECOMMENDED_SITES
            )
            fast_check_results = [
                {'title': result['title'], 'content': result['content']}
                for result in tavily_results.get('results', []) if 'title' in result and 'content' in result
            ]
            if not fast_check_results:
                return ("මට කනගාටුයි, ඔබේ ප්‍රකාශය සම්බන්ධ තොරතුරු අන්තර්ජාලයෙන් සොයාගත නොහැකි විය.")
            chat_session = self.model.start_chat()
            final_prompt = f"""
            ඔබ සිංහල ප්‍රකශ වල සත්‍ය අසත්‍ය බව සාක්ෂි සමග සහතික කරන AI සභායකයෙකි.
            කරුණාකර ඔබේ නිගමනය පළමු පේළිය "නිගමනය: [සත්‍ය/අසත්‍ය/තීරණය කළ නොහැක]" ලෙස ලබා දෙන්න.
            පහත ප්‍රකාශය විමර්ශනය කරන්න්: '{claim}'
            මෙම ප්‍රකාශය සත්‍යාපනය කිරීම සඳහා පහත සෙවුම් ප්‍රතිඵල සලකා බලන්න:
            {fast_check_results}
            """
            response = chat_session.send_message(final_prompt)
            return clean_response(response.text)
        except Exception as e:
            return f"මට කනගාටුයි, සෙවීමේදී දෝෂයක් ඇතිවිය: {str(e)}"

    def verify_combined(self, claim, vector_store, search_clients):
        rag_result = self.verify_with_rag(claim, vector_store)
        search_result = self.verify_with_search(claim, search_clients)
        chat_session = self.model.start_chat()
        final_prompt = f"""
        ඔබ සිංහල ප්‍රකශ වල සත්‍ය අසත්‍ය බව සාක්ෂි සමග සහතික කරන AI සභායකයෙකි.
        කරුණාකර ඔබේ නිගමනය පළමු පේළිය "නිගමනය: [සත්‍ය/අසත්‍ය/තීරණය කළ නොහැක]" ලෙස ලබා දෙන්න.
        පහත ප්‍රකාශය විමර්ශනය කරන්න: '{claim}'
        මෙම ප්‍රකාශය සත්‍යාපනය කිරීම සඳහා පහත ප්‍රතිඵල ලැබී ඇත:
        1. වෙක්ටර් දත්ත ගබඩාවෙන් (RAG) ලැබුණු ප්‍රතිඵලය: {rag_result}
        2. සජීවී අන්තර්ජාල සෙවීමෙන් ලැබුණු ප්‍රතිඵලය: {search_result}
        ඉහත දෙකම සලකා බැලීමෙන්, කරුණාකර පළමු පේළියේ පමණක් ඔබේ නිගමනය (උදා: නිගමනය: සත්‍ය) ලබා දෙන්න,
        හා ඉතිරි පේළි වල හේතු විස්තර කරන්න.
        """
        response = chat_session.send_message(final_prompt)
        return clean_response(response.text)

class VerdictAgent:
    def extract_verdict(self, analysis):
        """
        Search through the analysis for the first line starting with "නිගමනය:" and extract the verdict.
        """
        analysis_lines = analysis.split('\n')
        verdict_line = ""
        for line in analysis_lines:
            line = line.strip()
            if line.startswith("නිගමනය:"):
                verdict_line = line
                break

        # Normalize the text to handle Unicode variations
        normalized_line = unicodedata.normalize('NFC', verdict_line)
        if re.search(r"නිගමනය:\s*සත්‍?ය", normalized_line):
            return "true"
        elif re.search(r"නිගමනය:\s*අසත්‍?ය", normalized_line):
            return "false"
        else:
            return "insufficient information"

class OrchestrationAgent:
    def __init__(self, domain_classifier, retrieval_agents, fact_analyzer, verdict_agent, search_clients, decision_agent):
        self.domain_classifier = domain_classifier
        self.retrieval_agents = retrieval_agents
        self.fact_analyzer = fact_analyzer
        self.verdict_agent = verdict_agent
        self.search_clients = search_clients
        self.decision_agent = decision_agent

    def verify_statement(self, statement, method="rag"):
        if method not in ["rag", "search", "both", "automatic"]:
            raise ValueError("Invalid method. Choose from 'rag', 'search', 'both', 'automatic'.")

        domain = self.domain_classifier.classify(statement)
        if domain not in self.retrieval_agents:
            raise ValueError(f"Domain '{domain}' not supported. Supported domains: {list(self.retrieval_agents.keys())}")

        if method == "automatic":
            retrieval_agent = self.retrieval_agents[domain]
            vector_store = retrieval_agent.vector_store
            retrieved_docs = retrieve_relevant_documents(statement, vector_store)
            decision = self.decision_agent.decide(statement, retrieved_docs)
            if decision == "sufficient":
                analysis = self.fact_analyzer.verify_with_rag(statement, vector_store)
                method_used = "ස්ථානීය දත්ත (RAG)"
            else:
                analysis = self.fact_analyzer.verify_with_search(statement, self.search_clients)
                method_used = "අන්තර්ජාල සෙවීම"
        else:
            retrieval_agent = self.retrieval_agents[domain]
            vector_store = retrieval_agent.vector_store
            if method == "rag":
                analysis = self.fact_analyzer.verify_with_rag(statement, vector_store)
                method_used = "ස්ථානීය දත්ත"
            elif method == "search":
                analysis = self.fact_analyzer.verify_with_search(statement, self.search_clients)
                method_used = "අන්තර්ජාල සෙවීම"
            elif method == "both":
                analysis = self.fact_analyzer.verify_combined(statement, vector_store, self.search_clients)
                method_used = "ස්ථානීය දත්ත සහ අන්තර්ජාල සෙවීම"

        verdict = self.verdict_agent.extract_verdict(analysis)
        return verdict, analysis, method_used

@st.cache_resource
def initialize_app():
    try:
        # Get current script directory for absolute paths
        current_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(current_dir, 'data')
        
        st.write("🔍 Debug Info:")
        st.write(f"Current script directory: {current_dir}")
        st.write(f"Data directory: {data_dir}")
        
        # Verify data directory exists
        if not os.path.exists(data_dir):
            st.error(f"❌ 'data' directory not found at: {data_dir}")
            return None
            
        # List files in data directory
        files_in_data = os.listdir(data_dir)
        st.write(f"Files in data directory: {files_in_data}")
        
        # Find CSV files with absolute paths
        csv_files_found = [f for f in files_in_data if f.endswith('.csv')]
        st.write(f"CSV files found: {csv_files_found}")
        
        # Build csv_paths with absolute paths
        csv_paths = {}
        file_domain_mapping = {
            'economics_data.csv': 'economics',
            'politics_data.csv': 'politics', 
            'health_data.csv': 'health'
        }
        
        for filename, domain in file_domain_mapping.items():
            full_path = os.path.join(data_dir, filename)
            if os.path.exists(full_path):
                csv_paths[domain] = full_path
                st.success(f"✅ Found {domain} data: {full_path}")
            else:
                st.warning(f"⚠️ Missing {domain} data: {filename}")
        
        # Verify we found at least one CSV
        if not csv_paths:
            st.error("🚫 No CSV files found! Please check your data directory")
            return None
            
        st.info(f"📁 Using {len(csv_paths)} CSV file(s): {list(csv_paths.keys())}")
        
        # Load vector stores
        vector_stores = {}
        for domain, csv_path in csv_paths.items():
            try:
                persist_directory = f"chroma_db_{domain}"
                vector_store, doc_count = setup_vector_store(csv_path, persist_directory)
                vector_stores[domain] = vector_store
                st.success(f"✅ Loaded {doc_count} documents for {domain}")
            except Exception as e:
                st.error(f"❌ Error loading {domain} data: {str(e)}")
                st.exception(e)  # Show full traceback
        
        if not vector_stores:
            st.error("🚫 Failed to load any vector stores. Check your CSV files and try again.")
            return None
            
        return vector_stores
        
    except Exception as e:
        st.error(f"🔥 Critical error in app initialization: {str(e)}")
        st.exception(e)
        return None
        
        # Initialize models
        general_model = initialize_gemini("models/gemma-3-27b-it")
        decision_model = initialize_gemini("models/gemma-3-27b-it")
        
        # Initialize agents
        retrieval_agents = {domain: DataRetrievalAgent(domain, vector_stores[domain]) 
                          for domain in vector_stores.keys()}
        domain_classifier = DomainClassificationAgent(general_model)
        fact_analyzer = FactAnalysisAgent(general_model)
        verdict_agent = VerdictAgent()
        search_clients = initialize_search_clients()
        decision_agent = DecisionAgent(decision_model)
        
        orchestrator = OrchestrationAgent(
            domain_classifier, retrieval_agents, fact_analyzer, 
            verdict_agent, search_clients, decision_agent
        )
        
        st.success(f"🎉 Application initialized successfully with {len(vector_stores)} domain(s)!")
        return orchestrator
        
    except Exception as e:
        st.error(f"💥 Error initializing application: {str(e)}")
        st.write("Full error details:", str(e))
        return None

# Main Streamlit UI
def main():
    st.set_page_config(page_title="සිංහල සත්‍ය සෙවුම්කරු", layout="centered")
    st.title("සිංහල සත්‍ය සෙවුම්කරු")
    st.markdown("**Sinhala Fact Verification System**")
    st.write("මෙම පද්ධතිය ඔබ ලබා දෙන සිංහල ප්‍රකාශයන්හි සත්‍යතාවය පරීක්ෂා කරයි.")
    
    # Initialize the app
    orchestrator = initialize_app()
    if orchestrator is None:
        st.stop()
    
    with st.form("fact_check_form"):
        statement = st.text_area(
            "සත්‍යාපනය කිරීමට අවශ්‍ය ප්‍රකාශය ඇතුළත් කරන්න",
            height=150,
            placeholder="උදාහරණය: ශ්‍රී ලංකාවේ ආර්ථිකය 2023 දී 5% කින් වර්ධනය විය."
        )
        
        options = ["Automatic", "RAG only"]
        search_clients = initialize_search_clients()
        if search_clients.get("available", False):
            options.extend(["Search only", "Both"])
        
        method = st.selectbox("තීරණ ගැනීමේ ක්‍රමය", options)
        submit_button = st.form_submit_button("සත්‍යාපනය කරන්න")

    if submit_button and statement:
        method_map = {
            "Automatic": "automatic",
            "RAG only": "rag",
            "Search only": "search",
            "Both": "both"
        }
        selected_method = method_map[method]
        
        with st.spinner("ප්‍රකාශය සත්‍යාපනය කරමින් පවතී..."):
            try:
                verdict, analysis, method_used = orchestrator.verify_statement(statement, method=selected_method)
                
                st.subheader("නිගමනය")
                st.markdown(f"**භාවිත කළ ක්‍රමය**: {method_used}")
                
                if verdict == "true":
                    st.success("**සත්‍යයි**")
                elif verdict == "false":
                    st.error("**අසත්‍යයි**")
                else:
                    st.warning("**තොරතුරු ප්‍රමාණවත් නොවේ**")

                # Remove verdict phrases from analysis
                cleaned_analysis = re.sub(r"නිගමනය:\s*(සත්ය|අසත්ය|තොරතුරු ප්‍රමාණවත් නොවේ)", "", analysis)
                st.subheader("විශ්ලේෂණය")
                st.write(cleaned_analysis)
            except Exception as e:
                st.error(f"Error during fact checking: {str(e)}")


if __name__ == "__main__":
    main()
# Run the Streamlit app
# To run the app, use the command: streamlit run app.py