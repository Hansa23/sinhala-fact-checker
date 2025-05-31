# # app.py

# # ---------------------
# # Top‐Level Imports & SQLite Patch
# # ---------------------

import os
import sys
import re
import unicodedata
import shutil
import sqlite3 as _sqlite  # used to check system SQLite version

# Debugging output (appears in logs)
print(f"Python version: {sys.version}")
print(f"Current directory: {os.getcwd()}")
if os.path.isdir("data"):
    print(f"Files in data dir: {os.listdir('data')}")

# Workaround for pipeline import (some versions of transformers structure differs)
try:
    from transformers import pipeline
except ImportError:
    from transformers.pipelines import pipeline
    print("Used fallback pipeline import")

# If system SQLite is older than 3.35.0, swap in pysqlite3 from pysqlite3-binary
version_tuple = tuple(int(x) for x in _sqlite.sqlite_version.split("."))
if version_tuple < (3, 35, 0):
    try:
        import pysqlite3 as sqlite3
        sys.modules["sqlite3"] = sqlite3
        print(f"Using pysqlite3-binary (SQLite {sqlite3.sqlite_version})")
    except ImportError:
        raise RuntimeError(
            "pysqlite3-binary not installed. Install with: pip install pysqlite3-binary"
        )

# # ---------------------
# # Now import core dependencies
# # ---------------------

import torch
from transformers import pipeline  # after the fallback above, this should succeed
import chromadb
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import google.generativeai as genai
from tavily import TavilyClient
import streamlit as st

# # ---------------------
# # Utility Functions
# # ---------------------

# def initialize_gemini(model_name: str):
#     """
#     Configure and return a Gemini (Google Generative AI) model instance.
#     """
#     gemini_api_key = os.getenv("GEMINI_API_KEY")
#     if not gemini_api_key:
#         raise ValueError("GEMINI_API_KEY is not set in environment variables.")
#     genai.configure(api_key=gemini_api_key)
#     generation_config = {
#         "temperature": 0.7,
#         "top_p": 0.9,
#         "top_k": 40,
#         "max_output_tokens": 4096,
#         "response_mime_type": "text/plain",
#     }
#     return genai.GenerativeModel(model_name=model_name, generation_config=generation_config)

# def remove_non_sinhala(text: str) -> str:
#     """
#     Remove characters not in the Sinhala Unicode range or basic Latin punctuation/digits.
#     """
#     allowed_ranges = [
#         (0x0D80, 0x0DFF),  # Sinhala Unicode
#         (0x0020, 0x007E),  # Basic Latin (punctuation, digits, etc.)
#     ]
#     return "".join(
#         c for c in text
#         if any(start <= ord(c) <= end for start, end in allowed_ranges)
#     )

# def clean_response(response_text: str) -> str:
#     """
#     Clean the response text by removing unwanted patterns and non-Sinhala characters.
#     """
#     processed = response_text.replace("මූලාශ්‍ර: /content/context_facts.csv", "")
#     processed = re.sub(r"\(මූලාශ්‍ර \d+(?:, \d+)*\)", "", processed)
#     processed = unicodedata.normalize("NFC", processed).replace("\u200d", "")
#     return remove_non_sinhala(processed)

# def setup_vector_store(csv_path: str, persist_directory: str, force_rebuild: bool = False):
#     """
#     Build or load a Chroma vector store from a CSV file.
#     - csv_path: absolute path to the CSV file.
#     - persist_directory: folder to store/load Chroma artifacts (must be non-empty).
#     - force_rebuild: if True, remove any existing persist_directory and rebuild from scratch.
#     Returns: (vector_store_instance, document_count)
#     """
#     if not persist_directory:
#         raise ValueError("persist_directory cannot be empty.")

#     # Ensure parent directory exists (e.g., "chroma_db_economics" has no parent to create, so dirname is "")
#     parent_dir = os.path.dirname(persist_directory)
#     if parent_dir:
#         os.makedirs(parent_dir, exist_ok=True)
#     else:
#         # If persist_directory has no parent (e.g., "chroma_db_economics"), ensure current dir is okay.
#         os.makedirs(persist_directory, exist_ok=True)

#     # Determine device for embeddings
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     embedding_model = HuggingFaceEmbeddings(
#         model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
#         model_kwargs={"device": device}
#     )

#     # If an existing vector store folder is present and not force_rebuild, try loading it
#     if os.path.isdir(persist_directory) and not force_rebuild:
#         try:
#             vector_store = Chroma(persist_directory=persist_directory, embedding_function=embedding_model)
#             return vector_store, vector_store._collection.count()
#         except Exception as e:
#             print(f"Error loading existing vector store at '{persist_directory}': {e}")
#             print("Rebuilding vector store from CSV.")
#             shutil.rmtree(persist_directory)

#     # If force_rebuild is True, remove the directory first
#     if force_rebuild and os.path.isdir(persist_directory):
#         shutil.rmtree(persist_directory)

#     # Ensure CSV exists
#     if not os.path.isfile(csv_path):
#         raise FileNotFoundError(f"CSV file not found at {csv_path}")

#     # Load CSV documents
#     loader = CSVLoader(file_path=csv_path, encoding="utf-8")
#     documents = loader.load()
#     documents = [doc for doc in documents if doc.page_content.strip()]

#     # Split into chunks
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
#     texts = text_splitter.split_documents(documents)

#     # Create Chroma vector store
#     vector_store = Chroma.from_documents(
#         documents=texts,
#         embedding=embedding_model,
#         persist_directory=persist_directory
#     )
#     return vector_store, len(texts)

# def retrieve_relevant_documents(question: str, vector_store, k: int = 5):
#     """
#     Perform a similarity search on the vector_store.
#     Returns a list of dicts: [{'content': str, 'source': str}, ...]
#     """
#     try:
#         docs = vector_store.similarity_search(question, k=k)
#         results = []
#         for doc in docs:
#             content = getattr(doc, "page_content", "").strip()
#             if not content:
#                 continue
#             metadata = getattr(doc, "metadata", {})
#             source = metadata.get("source", "Unknown source")
#             results.append({"content": content, "source": source})
#         return results
#     except Exception as e:
#         print(f"Error retrieving documents for question '{question}': {e}")
#         return []

# def initialize_search_clients():
#     """
#     Initialize any external search clients (e.g., Tavily).
#     Returns a dict: {'available': bool, 'tavily': TavilyClient or None}
#     """
#     clients = {"available": False, "tavily": None}
#     tavily_api_key = os.getenv("TAVILY_API_KEY")
#     if tavily_api_key:
#         try:
#             tavily_client = TavilyClient(tavily_api_key)
#             clients["tavily"] = tavily_client
#             clients["available"] = True
#         except Exception as e:
#             print(f"Error initializing Tavily client: {e}")
#     return clients

# # ---------------------
# # Agent Classes
# # ---------------------

# class DomainClassificationAgent:
#     def __init__(self, gemini_model):
#         self.gemini_model = gemini_model

#     def classify(self, statement: str) -> str:
#         """
#         Use Gemini to classify the given Sinhala statement into one of: 'politics', 'economics', 'health'.
#         Returns the lower‐cased domain.
#         """
#         chat_session = self.gemini_model.start_chat()
#         prompt = f"""
#         පහත සිංහල ප්‍රකාශය කුමන විෂය ක්ෂේත්‍රයට අයත්ද යන්න තීරණය කරන්න: '{statement}'
#         විෂය ක්ෂේත්‍ර: politics, economics, health
#         ඔබේ තීරණය එක් වචනයකින් පමණක් ලබා දෙන්න (උදා: politics).
#         """
#         response = chat_session.send_message(prompt)
#         domain = response.text.strip().lower()
#         if domain.startswith("තීරණය:"):
#             domain = domain.replace("තීරණය:", "").strip()
#         return domain

# class DataRetrievalAgent:
#     def __init__(self, domain: str, vector_store):
#         self.domain = domain
#         self.vector_store = vector_store

#     def retrieve(self, statement: str):
#         return retrieve_relevant_documents(statement, self.vector_store)

# class DecisionAgent:
#     def __init__(self, gemini_model):
#         self.model = gemini_model

#     def decide(self, statement: str, retrieved_docs) -> str:
#         """
#         Decide whether local RAG info is sufficient. Returns 'sufficient' or 'insufficient'.
#         """
#         chat_session = self.model.start_chat()
#         prompt = f"""
#         පහත සිංහල ප්‍රකාශය සඳහා ලබා දී ඇති තොරතුරු ප්‍රමාණවත්ද යන්න තීරණය කරන්න: '{statement}'

#         ලබා ඇති තොරතුරු: {retrieved_docs}

#         තොරතුරු ප්‍රමාණවත් යැයි සැලකෙන්නේ, එම තොරතුරු මගින් ප්‍රකාශයේ සඳහන් කරුණු පිළිබඳ නිශ්චිත තොරතුරු තිබේ නම් පමණි.
#         මෙම තොරතුරු ලබාදී ඇති ප්‍රකාශයේ සත්‍ය අසත්‍ය බව නිගමනය සදහා යොදාගන්නා බැවින් වගකීම් සහගත ලෙස පිළිතුරු ලබාදෙන්න.

#         කරුණාකර, ඔබේ පිළිතුරේ පළමු පේළිය ස්පස්ට "නිගමනය: [සත්‍ය/අසත්‍ය/තීරණය කළ නොහැක]" ලෙස ලබා දෙන්න.
#         """
#         response = chat_session.send_message(prompt)
#         response_text = response.text.strip().lower()
#         verdict_line = ""
#         for line in response_text.split("\n"):
#             if line.strip().startswith("නිගමනය:"):
#                 verdict_line = line.strip()
#                 break
#         if re.search(r"නිගමනය:\s*සත්‍?ය", verdict_line):
#             return "sufficient"
#         else:
#             return "insufficient"

# class FactAnalysisAgent:
#     def __init__(self, gemini_model):
#         self.model = gemini_model

#     def verify_with_rag(self, claim: str, vector_store):
#         """
#         Perform RAG verification on the claim using the given vector_store.
#         Returns a cleaned Sinhala response (analysis).
#         """
#         retrieved_docs = retrieve_relevant_documents(claim, vector_store)
#         if not retrieved_docs:
#             return (
#                 "මට කනගාටුයි, ඔබේ ප්‍රකාශය පරීක්ෂා කිරීමට අවශ්‍ය තොරතුරු සොයාගත නොහැකි විය. "
#                 "කරුණාකර වෙනත් ප්‍රකාශයක් උත්සාහ කරන්න."
#             )
#         chat_session = self.model.start_chat()
#         final_prompt = f"""
#         ඔබ සිංහල ප්‍රකශ වල සත්‍ය අසත්‍ය බව සාක්ෂි සමග සහතික කරන AI සභායකයෙකි.
#         කරුණාකර ඔබේ නිගමනය පළමු පේළිය "නිගමනය: [සත්‍ය/අසත්‍ය/තීරණය කළ නොහැක]" ලෙස ලබා දෙන්න.
#         පහත ප්‍රකාශය විමර්ශනය කරන්න: '{claim}'
#         ලබා ඇති තොරතුරු: {retrieved_docs}
#         කරුණාකර:
#         1. පළමු පේළියට පමණක් ඔබේ නිගමනය (උදා: නිගමනය: සත්‍ය)
#         2. ඉතිරි පේළි වල ඔබේ විස්තරාත්මක හේතු සඳහන් කරන්න.
#         """
#         response = chat_session.send_message(final_prompt)
#         return clean_response(response.text)

#     def verify_with_search(self, claim: str, search_clients):
#         """
#         Perform a live search (e.g., via Tavily) and return a cleaned Sinhala response.
#         """
#         if not search_clients.get("available"):
#             return (
#                 "මට කනගාටුයි, සෙවුම් යාන්ත්‍රණය භාවිතා කළ නොහැකි බැවින් "
#                 "තහවුරු කිරීම් සිදු කළ නොහැක. API යතුරු සැකසීමෙන් පසු නැවත උත්සාහ කරන්න."
#             )
#         tavily_client = search_clients.get("tavily")
#         if not tavily_client:
#             return (
#                 "මට කනගාටුයි, Tavily සෙවුම් යාන්ත්‍රණය භාවිතා කළ නොහැකි බැවින් "
#                 "තහවුරු කිරීම් සිදු කළ නොහැක."
#             )
#         try:
#             tavily_results = tavily_client.search(
#                 query=claim + " Sri Lanka",
#                 search_depth="advanced",
#                 include_domains=RECOMMENDED_SITES
#             )
#             fast_check_results = [
#                 {"title": r["title"], "content": r["content"]}
#                 for r in tavily_results.get("results", [])
#                 if "title" in r and "content" in r
#             ]
#             if not fast_check_results:
#                 return (
#                     "මට කනගාටුයි, ඔබේ ප්‍රකාශය සම්බන්ධ තොරතුරු අන්තර්ජාලයෙන් සොයාගත නොහැකි විය."
#                 )
#             chat_session = self.model.start_chat()
#             final_prompt = f"""
#             ඔබ සිංහල ප්‍රකශ වල සත්‍ය අසත්‍ය බව සාක්ෂි සමග සහතික කරන AI සභායකයෙකි.
#             කරුණාකර ඔබේ නිගමනය පළමු පේළිය "නිගමනය: [සත්‍ය/අසත්‍ය/තීරණය කළ නොහැක]" ලෙස ලබා දෙන්න.
#             පහත ප්‍රකාශය විමර්ශනය කරන්න: '{claim}'
#             මෙම ප්‍රකාශය සත්‍යාපනය කිරීම සඳහා පහත සෙවුම් ප්‍රතිඵල සලකා බලන්න:
#             {fast_check_results}
#             """
#             response = chat_session.send_message(final_prompt)
#             return clean_response(response.text)
#         except Exception as e:
#             return f"මට කනගාටුයි, සෙවීමේදී දෝෂයක් ඇතිවිය: {str(e)}"

#     def verify_combined(self, claim: str, vector_store, search_clients):
#         """
#         Combine RAG + live search verifications. Return a cleaned Sinhala response.
#         """
#         rag_result = self.verify_with_rag(claim, vector_store)
#         search_result = self.verify_with_search(claim, search_clients)
#         chat_session = self.model.start_chat()
#         final_prompt = f"""
#         ඔබ සිංහල ප්‍රකාශ වල සත්‍ය අසත්‍ය බව සාක්ෂි සමග සහතික කරන AI සභායකයෙකි.
#         කරුණාකර ඔබේ නිගමනය පළමු පේළිය "නිගමනය: [සත්‍ය/අසත්‍ය/තීරණය කළ නොහැක]" ලෙස ලබා දෙන්න.
#         පහත ප්‍රකාශය විමර්ශනය කරන්න: '{claim}'
#         මෙම ප්‍රකාශය සත්‍යාපනය කිරීම සඳහා පහත ප්‍රතිඵල ලැබී ඇත:
#         1. වෙක්ටර් දත්තගබඩාවෙන් (RAG) ලැබුණු ප්‍රතිඵලය: {rag_result}
#         2. සජීවී අන්තර්ජාල සෙවීමෙන් (Search) ලැබුණු ප්‍රතිඵලය: {search_result}
#         ඉහත දෙකම සලකා බැලීමෙන්, පළමු පේළියේ පමණක් ඔබේ නිගමනය (උදා: නිගමනය: සත්‍ය) ලබා දෙන්න,
#         හා ඉතිරි පේළි වල හේතු විස්තර කරන්න.
#         """
#         response = chat_session.send_message(final_prompt)
#         return clean_response(response.text)

# class VerdictAgent:
#     def extract_verdict(self, analysis: str) -> str:
#         """
#         Extract the verdict ('true', 'false', or 'insufficient information') from a cleaned analysis.
#         """
#         lines = analysis.split("\n")
#         verdict_line = ""
#         for line in lines:
#             if line.strip().startswith("නිගමනය:"):
#                 verdict_line = line.strip()
#                 break

#         normalized = unicodedata.normalize("NFC", verdict_line)
#         if re.search(r"නිගමනය:\s*සත්‍?ය", normalized):
#             return "true"
#         elif re.search(r"නිගමනය:\s*අසත්‍?ය", normalized):
#             return "false"
#         else:
#             return "insufficient information"

# class OrchestrationAgent:
#     def __init__(
#         self,
#         domain_classifier: DomainClassificationAgent,
#         retrieval_agents: dict,
#         fact_analyzer: FactAnalysisAgent,
#         verdict_agent: VerdictAgent,
#         search_clients: dict,
#         decision_agent: DecisionAgent
#     ):
#         self.domain_classifier = domain_classifier
#         self.retrieval_agents = retrieval_agents
#         self.fact_analyzer = fact_analyzer
#         self.verdict_agent = verdict_agent
#         self.search_clients = search_clients
#         self.decision_agent = decision_agent

#     def verify_statement(self, statement: str, method: str = "rag"):
#         """
#         method can be: "rag", "search", "both", or "automatic"
#         Returns: (verdict: str, analysis: str, method_used: str)
#         """
#         if method not in ["rag", "search", "both", "automatic"]:
#             raise ValueError("Invalid method. Choose from 'rag', 'search', 'both', 'automatic'.")

#         domain = self.domain_classifier.classify(statement)
#         if domain not in self.retrieval_agents:
#             raise ValueError(
#                 f"Domain '{domain}' not supported. Supported domains: {list(self.retrieval_agents.keys())}"
#             )

#         retrieval_agent = self.retrieval_agents[domain]
#         vector_store = retrieval_agent.vector_store

#         if method == "automatic":
#             retrieved_docs = retrieve_relevant_documents(statement, vector_store)
#             decision = self.decision_agent.decide(statement, retrieved_docs)
#             if decision == "sufficient":
#                 analysis = self.fact_analyzer.verify_with_rag(statement, vector_store)
#                 method_used = "ස්ථානීය දත්ත (RAG)"
#             else:
#                 analysis = self.fact_analyzer.verify_with_search(statement, self.search_clients)
#                 method_used = "අන්තර්ජාල සෙවීම"
#         else:
#             if method == "rag":
#                 analysis = self.fact_analyzer.verify_with_rag(statement, vector_store)
#                 method_used = "ස්ථානීය දත්ත (RAG)"
#             elif method == "search":
#                 analysis = self.fact_analyzer.verify_with_search(statement, self.search_clients)
#                 method_used = "අන්තර්ජාල සෙවීම"
#             else:  # both
#                 analysis = self.fact_analyzer.verify_combined(statement, vector_store, self.search_clients)
#                 method_used = "ස්ථානීය දත්ත (RAG) සහ අන්තර්ජාල සෙවීම"

#         verdict = self.verdict_agent.extract_verdict(analysis)
#         return verdict, analysis, method_used

# # ---------------------
# # Core Initialization Logic
# # ---------------------

# RECOMMENDED_SITES = [
#     "https://hashtaggeneration.org/fact-check/",
#     "https://srilanka.factcrescendo.com/",
#     "https://www.bbc.com/sinhala",
#     "https://sinhala.newsfirst.lk/",
#     "https://www.adaderana.lk/",
#     "https://www.hirunews.lk/english/",
#     "https://www.cbsl.gov.lk/si"
# ]

# @st.cache_resource(show_spinner=False)
# def initialize_app():
#     """
#     Build vector stores for each domain and return an OrchestrationAgent instance.
#     If errors occur, show them via Streamlit and return None.
#     """
#     try:
#         # Determine folders
#         current_dir = os.path.dirname(os.path.abspath(__file__))
#         data_dir = os.path.join(current_dir, "data")

#         st.write("🔍 Debug Info:")
#         st.write(f"Current script directory: {current_dir}")
#         st.write(f"Data directory: {data_dir}")

#         # Verify data directory exists
#         if not os.path.isdir(data_dir):
#             st.error(f"❌ 'data' directory not found at: {data_dir}")
#             return None

#         files_in_data = os.listdir(data_dir)
#         st.write(f"Files in data directory: {files_in_data}")

#         # Map CSV filenames → domain keys
#         file_domain_mapping = {
#             "economics_data.csv": "economics",
#             "politics_data.csv": "politics",
#             "health_data.csv": "health"
#         }

#         # Build absolute CSV paths for domains
#         csv_paths = {}
#         for filename, domain in file_domain_mapping.items():
#             full_path = os.path.join(data_dir, filename)
#             if os.path.isfile(full_path):
#                 csv_paths[domain] = full_path
#                 st.success(f"✅ Found {domain} data: {full_path}")
#             else:
#                 st.warning(f"⚠️ Missing {domain} data: {filename}")

#         if not csv_paths:
#             st.error("🚫 No CSV files found! Please check your data directory.")
#             return None

#         st.info(f"📁 Using {len(csv_paths)} CSV file(s): {list(csv_paths.keys())}")

#         # Build/load vector stores
#         vector_stores = {}
#         for domain, csv_path in csv_paths.items():
#             try:
#                 persist_directory = os.path.join(current_dir, f"chroma_db_{domain}")
#                 vector_store, doc_count = setup_vector_store(csv_path, persist_directory)
#                 vector_stores[domain] = vector_store
#                 st.success(f"✅ Loaded {doc_count} documents for '{domain}'")
#             except Exception as e:
#                 st.error(f"❌ Error loading '{domain}' data: {str(e)}")
#                 st.exception(e)

#         if not vector_stores:
#             st.error("🚫 Failed to load any vector stores. Check your CSV files and try again.")
#             return None

#         # Initialize Gemini models
#         general_model = initialize_gemini("models/gemma-3-27b-it")
#         decision_model = initialize_gemini("models/gemma-3-27b-it")

#         # Instantiate agents
#         retrieval_agents = {
#             domain: DataRetrievalAgent(domain, vs)
#             for domain, vs in vector_stores.items()
#         }
#         domain_classifier = DomainClassificationAgent(general_model)
#         fact_analyzer = FactAnalysisAgent(general_model)
#         verdict_agent = VerdictAgent()
#         search_clients = initialize_search_clients()
#         decision_agent = DecisionAgent(decision_model)

#         orchestrator = OrchestrationAgent(
#             domain_classifier=domain_classifier,
#             retrieval_agents=retrieval_agents,
#             fact_analyzer=fact_analyzer,
#             verdict_agent=verdict_agent,
#             search_clients=search_clients,
#             decision_agent=decision_agent
#         )

#         st.success("🎉 Application initialized successfully!")
#         return orchestrator

#     except Exception as e:
#         st.error(f"🔥 Critical error in app initialization: {str(e)}")
#         st.exception(e)
#         return None

# # ---------------------
# # Main Streamlit UI
# # ---------------------

# def main():
#     st.set_page_config(page_title="සිංහල සත්‍ය සෙවුම්කරු", layout="centered")
#     st.title("සිංහල සත්‍ය සෙවුම්කරු")
#     st.markdown("**Sinhala Fact Verification System**")
#     st.write("මෙම පද්ධතිය ඔබ ලබා දෙන සිංහල ප්‍රකාශයන්හි සත්‍යතාවය පරීක්ෂා කරයි.")

#     # Initialize the orchestrator (vector stores + agents)
#     orchestrator = initialize_app()
#     if orchestrator is None:
#         # Initialization failed; stop rendering further UI
#         st.stop()

#     with st.form("fact_check_form"):
#         statement = st.text_area(
#             "සත්‍යාපනය කිරීමට අවශ්‍ය ප්‍රකාශය ඇතුළත් කරන්න",
#             height=150,
#             placeholder="උදාහරණය: ශ්‍රී ලංකාවේ ආර්ථිකය 2023 දී 5% කින් වර්ධනය විය."
#         )

#         # Build method options
#         options = ["Automatic", "RAG only"]
#         search_clients = initialize_search_clients()
#         if search_clients.get("available", False):
#             options.extend(["Search only", "Both"])

#         method = st.selectbox("තීරණ ගැනීමේ ක්‍රමය", options)
#         submit_button = st.form_submit_button("සත්‍යාපනය කරන්න")

#     if submit_button and statement:
#         # Map human‐readable option to internal method name
#         method_map = {
#             "Automatic": "automatic",
#             "RAG only": "rag",
#             "Search only": "search",
#             "Both": "both"
#         }
#         selected_method = method_map.get(method, "rag")

#         with st.spinner("ප්‍රකාශය සත්‍යාපනය කරමින් පවතී..."):
#             try:
#                 verdict, analysis, method_used = orchestrator.verify_statement(
#                     statement, method=selected_method
#                 )

#                 st.subheader("නිගමනය")
#                 st.markdown(f"**භාවිත කළ ක්‍රමය**: {method_used}")

#                 if verdict == "true":
#                     st.success("**සත්‍යයි**")
#                 elif verdict == "false":
#                     st.error("**අසත්‍යයි**")
#                 else:
#                     st.warning("**තොරතුරු ප්‍රමාණවත් නොවේ**")

#                 # Remove verdict prefix from analysis before showing
#                 cleaned_analysis = re.sub(
#                     r"නිගමනය:\s*(සත්ය|අසත්ය|තොරතුරු ප්‍රමාණවත් නොවේ)", "",
#                     analysis
#                 )
#                 st.subheader("විශ්ලේෂණය")
#                 st.write(cleaned_analysis)

#             except Exception as e:
#                 st.error(f"Error during fact checking: {str(e)}")
#                 st.exception(e)

# if __name__ == "__main__":
#     main()

# app.py

# ---------------------
# Top-Level Imports & SQLite Patch
# ---------------------

import os
import sys
import re
import unicodedata
import sqlite3 as _sqlite  # used to check system SQLite version
import torch
from transformers import pipeline
import chromadb
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import google.generativeai as genai
from tavily import TavilyClient
import streamlit as st
import requests
import zipfile

# Debugging output (appears in logs)
print(f"Python version: {sys.version}")
print(f"Current directory: {os.getcwd()}")

# Workaround for pipeline import (some versions of transformers structure differs)
try:
    from transformers import pipeline
except ImportError:
    from transformers.pipelines import pipeline
    print("Used fallback pipeline import")

# If system SQLite is older than 3.35.0, swap in pysqlite3 from pysqlite3-binary
version_tuple = tuple(int(x) for x in _sqlite.sqlite_version.split("."))
if version_tuple < (3, 35, 0):
    try:
        import pysqlite3 as sqlite3
        sys.modules["sqlite3"] = sqlite3
        print(f"Using pysqlite3-binary (SQLite {sqlite3.sqlite_version})")
    except ImportError:
        raise RuntimeError(
            "pysqlite3-binary not installed. Install with: pip install pysqlite3-binary"
        )

# ---------------------
# Utility Functions
# ---------------------

def download_and_extract_zip(url, extract_to):
    """
    Download a zip file from a URL and extract it to a specified directory.
    """
    if not os.path.exists(extract_to):
        os.makedirs(extract_to, exist_ok=True)
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        zip_path = os.path.join(extract_to, "temp.zip")
        with open(zip_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(extract_to)
        os.remove(zip_path)
        # st.write(f"Downloaded and extracted {url} to {extract_to}")
        print(f"Downloaded and extracted {url} to {extract_to}")
    except Exception as e:
        st.error(f"Failed to download/extract {url}: {str(e)}")
        raise

def initialize_gemini(model_name: str):
    """
    Configure and return a Gemini (Google Generative AI) model instance.
    """
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        raise ValueError("GEMINI_API_KEY is not set in environment variables.")
    genai.configure(api_key=gemini_api_key)
    generation_config = {
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 40,
        "max_output_tokens": 4096,
        "response_mime_type": "text/plain",
    }
    return genai.GenerativeModel(model_name=model_name, generation_config=generation_config)

def remove_non_sinhala(text: str) -> str:
    """
    Remove characters not in the Sinhala Unicode range or basic Latin punctuation/digits.
    """
    allowed_ranges = [
        (0x0D80, 0x0DFF),  # Sinhala Unicode
        (0x0020, 0x007E),  # Basic Latin (punctuation, digits, etc.)
    ]
    return "".join(
        c for c in text
        if any(start <= ord(c) <= end for start, end in allowed_ranges)
    )

def clean_response(response_text: str) -> str:
    """
    Clean the response text by removing unwanted patterns and non-Sinhala characters.
    """
    processed = response_text.replace("මූලාශ්‍ර: /content/context_facts.csv", "")
    processed = re.sub(r"\(මූලාශ්‍ර \d+(?:, \d+)*\)", "", processed)
    processed = unicodedata.normalize("NFC", processed).replace("\u200d", "")
    return remove_non_sinhala(processed)

def retrieve_relevant_documents(question: str, vector_store, k: int = 5):
    """
    Perform a similarity search on the vector_store.
    Returns a list of dicts: [{'content': str, 'source': str}, ...]
    """
    try:
        docs = vector_store.similarity_search(question, k=k)
        results = []
        for doc in docs:
            content = getattr(doc, "page_content", "").strip()
            if not content:
                continue
            metadata = getattr(doc, "metadata", {})
            source = metadata.get("source", "Unknown source")
            results.append({"content": content, "source": source})
        return results
    except Exception as e:
        print(f"Error retrieving documents for question '{question}': {e}")
        return []

def initialize_search_clients():
    """
    Initialize any external search clients (e.g., Tavily).
    Returns a dict: {'available': bool, 'tavily': TavilyClient or None}
    """
    clients = {"available": False, "tavily": None}
    tavily_api_key = os.getenv("TAVILY_API_KEY")
    if tavily_api_key:
        try:
            tavily_client = TavilyClient(tavily_api_key)
            clients["tavily"] = tavily_client
            clients["available"] = True
        except Exception as e:
            print(f"Error initializing Tavily client: {e}")
    return clients

# ---------------------
# Agent Classes
# ---------------------

class DomainClassificationAgent:
    def __init__(self, gemini_model):
        self.gemini_model = gemini_model

    def classify(self, statement: str) -> str:
        """
        Use Gemini to classify the given Sinhala statement into one of: 'politics', 'economics', 'health'.
        Returns the lower-cased domain.
        """
        chat_session = self.gemini_model.start_chat()
        prompt = f"""
        පහත සිංහල ප්‍රකාශය කුමන විෂය ක්ෂේත්‍රයට අයත්ද යන්න තීරණය කරන්න: '{statement}'
        විෂය ක්ෂේත්‍ර: politics, economics, health
        ඔබේ තීරණය එක් වචනයකින් පමණක් ලබා දෙන්න (උදා: politics).
        """
        response = chat_session.send_message(prompt)
        domain = response.text.strip().lower()
        if domain.startswith("තීරණය:"):
            domain = domain.replace("තීරණය:", "").strip()
        return domain

class DataRetrievalAgent:
    def __init__(self, domain: str, vector_store):
        self.domain = domain
        self.vector_store = vector_store

    def retrieve(self, statement: str):
        return retrieve_relevant_documents(statement, self.vector_store)

class DecisionAgent:
    def __init__(self, gemini_model):
        self.model = gemini_model

    def decide(self, statement: str, retrieved_docs) -> str:
        """
        Decide whether local RAG info is sufficient. Returns 'sufficient' or 'insufficient'.
        """
        chat_session = self.model.start_chat()
        prompt = f"""
        පහත සිංහල ප්‍රකාශය සඳහා ලබා දී ඇති තොරතුරු ප්‍රමාණවත්ද යන්න තීරණය කරන්න: '{statement}'

        ලබා ඇති තොරතුරු: {retrieved_docs}

        තොරතුරු ප්‍රමාණවත් යැයි සැලකෙන්නේ, එම තොරතුරු මගින් ප්‍රකාශයේ සඳහන් කරුණු පිළිබඳ නිශ්චිත තොරතුරු තිබේ නම් පමණි.
        මෙම තොරතුරු ලබාදී ඇති ප්‍රකාශයේ සත්‍ය අසත්‍ය බව නිගමනය සදහා යොදාගන්නා බැවින් වගකීම් සහගත ලෙස පිළිතුරු ලබාදෙන්න.

        කරුණාකර, ඔබේ පිළිතුරේ පළමු පේළිය ස්පස්ට "නිගමනය: [සත්‍ය/අසත්‍ය/තීරණය කළ නොහැක]" ලෙස ලබා දෙන්න.
        """
        response = chat_session.send_message(prompt)
        response_text = response.text.strip().lower()
        verdict_line = ""
        for line in response_text.split("\n"):
            if line.strip().startswith("නිගමනය:"):
                verdict_line = line.strip()
                break
        if re.search(r"නිගමනය:\s*සත්‍?ය", verdict_line):
            return "sufficient"
        else:
            return "insufficient"

class FactAnalysisAgent:
    def __init__(self, gemini_model):
        self.model = gemini_model

    def verify_with_rag(self, claim: str, vector_store):
        """
        Perform RAG verification on the claim using the given vector_store.
        Returns a cleaned Sinhala response (analysis).
        """
        retrieved_docs = retrieve_relevant_documents(claim, vector_store)
        if not retrieved_docs:
            return (
                "මට කනගාටුයි, ඔබේ ප්‍රකාශය පරීක්ෂා කිරීමට අවශ්‍ය තොරතුරු සොයාගත නොහැකි විය. "
                "කරුණාකර වෙනත් ප්‍රකාශයක් උත්සාහ කරන්න."
            )
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

    def verify_with_search(self, claim: str, search_clients):
        """
        Perform a live search (e.g., via Tavily) and return a cleaned Sinhala response.
        """
        if not search_clients.get("available"):
            return (
                "මට කනගාටුයි, සෙවුම් යාන්ත්‍රණය භාවිතා කළ නොහැකි බැවින් "
                "තහවුරු කිරීම් සිදු කළ නොහැක. API යතුරු සැකසීමෙන් පසු නැවත උත්සාහ කරන්න."
            )
        tavily_client = search_clients.get("tavily")
        if not tavily_client:
            return (
                "මට කනගාටුයි, Tavily සෙවුම් යාන්ත්‍රණය භාවිතා කළ නොහැකි බැවින් "
                "තහවුරු කිරීම් සිදු කළ නොහැක."
            )
        try:
            tavily_results = tavily_client.search(
                query=claim + " Sri Lanka",
                search_depth="advanced",
                include_domains=RECOMMENDED_SITES
            )
            fast_check_results = [
                {"title": r["title"], "content": r["content"]}
                for r in tavily_results.get("results", [])
                if "title" in r and "content" in r
            ]
            if not fast_check_results:
                return (
                    "මට කනගාටුයි, ඔබේ ප්‍රකාශය සම්බන්ධ තොරතුරු අන්තර්ජාලයෙන් සොයාගත නොහැකි විය."
                )
            chat_session = self.model.start_chat()
            final_prompt = f"""
            ඔබ සිංහල ප්‍රකශ වල සත්‍ය අසත්‍ය බව සාක්ෂි සමග සහතික කරන AI සභායකයෙකි.
            කරුණාකර ඔබේ නිගමනය පළමු පේළිය "නිගමනය: [සත්‍ය/අසත්‍ය/තීරණය කළ නොහැක]" ලෙස ලබා දෙන්න.
            පහත ප්‍රකාශය විමර්ශනය කරන්න: '{claim}'
            මෙම ප්‍රකාශය සත්‍යාපනය කිරීම සඳහා පහත සෙවුම් ප්‍රතිඵල සලකා බලන්න:
            {fast_check_results}
            """
            response = chat_session.send_message(final_prompt)
            return clean_response(response.text)
        except Exception as e:
            return f"මට කනගාටුයි, සෙවීමේදී දෝෂයක් ඇතිවිය: {str(e)}"

    def verify_combined(self, claim: str, vector_store, search_clients):
        """
        Combine RAG + live search verifications. Return a cleaned Sinhala response.
        """
        rag_result = self.verify_with_rag(claim, vector_store)
        search_result = self.verify_with_search(claim, search_clients)
        chat_session = self.model.start_chat()
        final_prompt = f"""
        ඔබ සිංහල ප්‍රකාශ වල සත්‍ය අසත්‍ය බව සාක්ෂි සමග සහතික කරන AI සභායකයෙකි.
        කරුණාකර ඔබේ නිගමනය පළමු පේළිය "නිගමනය: [සත්‍ය/අසත්‍ය/තීරණය කළ නොහැක]" ලෙස ලබා දෙන්න.
        පහත ප්‍රකාශය විමර්ශනය කරන්න: '{claim}'
        මෙම ප්‍රකාශය සත්‍යාපනය කිරීම සඳහා පහත ප්‍රතිඵල ලැබී ඇත:
        1. වෙක්ටර් දත්තගබඩාවෙන් (RAG) ලැබුණු ප්‍රතිඵලය: {rag_result}
        2. සජීවී අන්තර්ජාල සෙවීමෙන් (Search) ලැබුණු ප්‍රතිඵලය: {search_result}
        ඉහත දෙකම සලකා බැලීමෙන්, පළමු පේළියේ පමණක් ඔබේ නිගමනය (උදා: නිගමනය: සත්‍ය) ලබා දෙන්න,
        හා ඉතිරි පේළි වල හේතු විස්තර කරන්න.
        """
        response = chat_session.send_message(final_prompt)
        return clean_response(response.text)

class VerdictAgent:
    def extract_verdict(self, analysis: str) -> str:
        """
        Extract the verdict ('true', 'false', or 'insufficient information') from a cleaned analysis.
        """
        lines = analysis.split("\n")
        verdict_line = ""
        for line in lines:
            if line.strip().startswith("නිගමනය:"):
                verdict_line = line.strip()
                break

        normalized = unicodedata.normalize("NFC", verdict_line)
        if re.search(r"නිගමනය:\s*සත්‍?ය", normalized):
            return "true"
        elif re.search(r"නිගමනය:\s*අසත්‍?ය", normalized):
            return "false"
        else:
            return "insufficient information"

class OrchestrationAgent:
    def __init__(
        self,
        domain_classifier: DomainClassificationAgent,
        retrieval_agents: dict,
        fact_analyzer: FactAnalysisAgent,
        verdict_agent: VerdictAgent,
        search_clients: dict,
        decision_agent: DecisionAgent
    ):
        self.domain_classifier = domain_classifier
        self.retrieval_agents = retrieval_agents
        self.fact_analyzer = fact_analyzer
        self.verdict_agent = verdict_agent
        self.search_clients = search_clients
        self.decision_agent = decision_agent

    def verify_statement(self, statement: str, method: str = "rag"):
        """
        method can be: "rag", "search", "both", or "automatic"
        Returns: (verdict: str, analysis: str, method_used: str)
        """
        if method not in ["rag", "search", "both", "automatic"]:
            raise ValueError("Invalid method. Choose from 'rag', 'search', 'both', 'automatic'.")

        domain = self.domain_classifier.classify(statement)
        if domain not in self.retrieval_agents:
            raise ValueError(
                f"Domain '{domain}' not supported. Supported domains: {list(self.retrieval_agents.keys())}"
            )

        retrieval_agent = self.retrieval_agents[domain]
        vector_store = retrieval_agent.vector_store

        if method == "automatic":
            retrieved_docs = retrieve_relevant_documents(statement, vector_store)
            decision = self.decision_agent.decide(statement, retrieved_docs)
            if decision == "sufficient":
                analysis = self.fact_analyzer.verify_with_rag(statement, vector_store)
                method_used = "ස්ථානීය දත්ත (RAG)"
            else:
                analysis = self.fact_analyzer.verify_with_search(statement, self.search_clients)
                method_used = "අන්තර්ජාල සෙවීම"
        else:
            if method == "rag":
                analysis = self.fact_analyzer.verify_with_rag(statement, vector_store)
                method_used = "ස්ථානීය දත්ත (RAG)"
            elif method == "search":
                analysis = self.fact_analyzer.verify_with_search(statement, self.search_clients)
                method_used = "අන්තර්ජාල සෙවීම"
            else:  # both
                analysis = self.fact_analyzer.verify_combined(statement, vector_store, self.search_clients)
                method_used = "ස්ථානීය දත්ත (RAG) සහ අන්තර්ජාල සෙවීම"

        verdict = self.verdict_agent.extract_verdict(analysis)
        return verdict, analysis, method_used

# ---------------------
# Core Initialization Logic
# ---------------------

RECOMMENDED_SITES = [
    "https://hashtaggeneration.org/fact-check/",
    "https://srilanka.factcrescendo.com/",
    "https://www.bbc.com/sinhala",
    "https://sinhala.newsfirst.lk/",
    "https://www.adaderana.lk/",
    "https://www.hirunews.lk/english/",
    "https://www.cbsl.gov.lk/si"
]

@st.cache_resource(show_spinner=False)
def initialize_app():
    """
    Initialize the application by loading prebuilt Chroma databases and setting up agents.
    """
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        domains = ["politics", "economics", "health"]
        # Replace with your actual GitHub Release URLs
        db_urls = {
            "politics": "https://github.com/Hansa23/sinhala-fact-checker/releases/download/v1.0/chroma_db_politics.zip",
            "economics": "https://github.com/Hansa23/sinhala-fact-checker/releases/download/v1.0/chroma_db_economics.zip",
            "health": "https://github.com/Hansa23/sinhala-fact-checker/releases/download/v1.0/chroma_db_health.zip"
        }
        vector_stores = {}
        device = "cuda" if torch.cuda.is_available() else "cpu"
        embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
            model_kwargs={"device": device}
        )
        for domain in domains:
            persist_directory = os.path.join(current_dir, f"chroma_db_{domain}")
            if not os.path.isdir(persist_directory):
                # st.info(f"Downloading Chroma DB for {domain}...")
                print(f"Downloading Chroma DB for {domain}...")
                url = db_urls[domain]
                download_and_extract_zip(url, current_dir)
                if not os.path.isdir(persist_directory):
                    raise FileNotFoundError(f"Persist directory {persist_directory} not found after extraction from {url}.")
            try:
                vector_store = Chroma(persist_directory=persist_directory, embedding_function=embedding_model)
                doc_count = vector_store._collection.count()
                vector_stores[domain] = vector_store
                # st.success(f"Loaded {doc_count} documents for '{domain}'")
                print(f"Loaded {doc_count} documents for '{domain}'")
            except Exception as e:
                # st.error(f"Error loading vector store for '{domain}': {e}")
                print(f"Error loading vector store for '{domain}': {e}")
                raise
        if not vector_stores:
            st.error("No vector stores loaded.")
            return None
        general_model = initialize_gemini("models/gemma-3-27b-it")
        decision_model = initialize_gemini("models/gemma-3-27b-it")
        retrieval_agents = {
            domain: DataRetrievalAgent(domain, vs)
            for domain, vs in vector_stores.items()
        }
        domain_classifier = DomainClassificationAgent(general_model)
        fact_analyzer = FactAnalysisAgent(general_model)
        verdict_agent = VerdictAgent()
        search_clients = initialize_search_clients()
        decision_agent = DecisionAgent(decision_model)
        orchestrator = OrchestrationAgent(
            domain_classifier=domain_classifier,
            retrieval_agents=retrieval_agents,
            fact_analyzer=fact_analyzer,
            verdict_agent=verdict_agent,
            search_clients=search_clients,
            decision_agent=decision_agent
        )
        st.success("Application initialized successfully!")
        return orchestrator
    except Exception as e:
        st.error(f"Critical error in app initialization: {str(e)}")
        st.exception(e)
        return None

# ---------------------
# Main Streamlit UI
# ---------------------

def main():
    st.set_page_config(page_title="සිංහල සත්‍ය සෙවුම්කරු", layout="centered")
    st.title("සිංහල සත්‍ය සෙවුම්කරු")
    st.markdown("**Sinhala Fact Verification System**")
    st.write("මෙම පද්ධතිය ඔබ ලබා දෙන සිංහල ප්‍රකාශයන්හි සත්‍යතාවය පරීක්ෂා කරයි.")

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
        selected_method = method_map.get(method, "rag")

        with st.spinner("ප්‍රකාශය සත්‍යාපනය කරමින් පවතී..."):
            try:
                verdict, analysis, method_used = orchestrator.verify_statement(
                    statement, method=selected_method
                )

                st.subheader("නිගමනය")
                st.markdown(f"**භාවිත කළ ක්‍රමය**: {method_used}")

                if verdict == "true":
                    st.success("**සත්‍යයි**")
                elif verdict == "false":
                    st.error("**අසත්‍යයි**")
                else:
                    st.warning("**තොරතුරු ප්‍රමාණවත් නොවේ**")

                cleaned_analysis = re.sub(
                    r"නිගමනය:\s*(සත්ය|අසත්ය|තොරතුරු ප්‍රමාණවත් නොවේ)", "",
                    analysis
                )
                st.subheader("විශ්ලේෂණය")
                st.write(cleaned_analysis)

            except Exception as e:
                st.error(f"Error during fact checking: {str(e)}")
                st.exception(e)
                
# Add this at the end of your script or main() function
        st.markdown(
            """
            <div style="text-align: center; font-size: 12px; color: #888; margin-top: 20px;">
                Created by Devin de Silva | Email: as2020323@sci.sjp.ac.lk | As a part of final year research project
            </div>
            """,
            unsafe_allow_html=True
        )


if __name__ == "__main__":
    main()
