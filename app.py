import streamlit as st
import os
import tempfile
import json
from datetime import datetime
from src.biolit_miner.paper_ingestion import PaperIngestionEngine, PaperMetadata
from src.biolit_miner.method_extractor import MethodExtractor, ExtractedMethods
from src.biolit_miner.code_generator import CodeGenerator, GeneratedCode
from src.biolit_miner.qa_interface import InteractiveQA

# Page configuration
st.set_page_config(
    page_title="BioLit Miner",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
.main-header {
    font-size: 3rem;
    font-weight: bold;
    color: #1f77b4;
    text-align: center;
    margin-bottom: 2rem;
}
.section-header {
    font-size: 1.5rem;
    font-weight: bold;
    color: #2e8b57;
    margin-top: 2rem;
    margin-bottom: 1rem;
    border-bottom: 2px solid #2e8b57;
    padding-bottom: 0.5rem;
}
.info-box {
    background-color: #f0f8ff;
    padding: 1rem;
    border-radius: 0.5rem;
    border-left: 4px solid #1f77b4;
    margin: 1rem 0;
}
.success-box {
    background-color: #f0fff0;
    padding: 1rem;
    border-radius: 0.5rem;
    border-left: 4px solid #32cd32;
    margin: 1rem 0;
}
.warning-box {
    background-color: #fffacd;
    padding: 1rem;
    border-radius: 0.5rem;
    border-left: 4px solid #ffa500;
    margin: 1rem 0;
}
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    if 'paper_metadata' not in st.session_state:
        st.session_state.paper_metadata = None
    if 'extracted_methods' not in st.session_state:
        st.session_state.extracted_methods = None
    if 'generated_code' not in st.session_state:
        st.session_state.generated_code = []
    if 'qa_interface' not in st.session_state:
        st.session_state.qa_interface = InteractiveQA()
    if 'current_step' not in st.session_state:
        st.session_state.current_step = 1

def display_header():
    st.markdown('<h1 class="main-header">🔬 BioLit Miner</h1>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
    <strong>Research Paper Analysis Tool</strong><br>
    Upload PDFs or input PubMed IDs/DOIs to extract computational methods, generate reproducible code, and interact with paper content through AI-powered Q&A.
    </div>
    """, unsafe_allow_html=True)

def paper_ingestion_section():
    st.markdown('<div class="section-header">📄 Paper Ingestion</div>', unsafe_allow_html=True)
    
    ingestion_engine = PaperIngestionEngine()
    
    # Input method selection
    input_method = st.radio(
        "Select input method:",
        ["Upload PDF", "PubMed ID", "DOI"],
        horizontal=True
    )
    
    paper_metadata = None
    
    if input_method == "Upload PDF":
        uploaded_file = st.file_uploader(
            "Upload a research paper PDF",
            type=['pdf'],
            help="Upload a PDF of the research paper you want to analyze"
        )
        
        if uploaded_file is not None:
            with st.spinner("Processing PDF..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_file_path = tmp_file.name
                
                try:
                    paper_metadata = ingestion_engine.process_pdf_upload(tmp_file_path)
                    st.success("PDF processed successfully!")
                except Exception as e:
                    st.error(f"Error processing PDF: {e}")
                finally:
                    os.unlink(tmp_file_path)
    
    elif input_method == "PubMed ID":
        pubmed_id = st.text_input(
            "Enter PubMed ID",
            placeholder="e.g., 12345678",
            help="Enter the PubMed ID of the paper"
        )
        
        if st.button("Fetch Paper", key="pubmed_fetch"):
            if pubmed_id:
                with st.spinner("Fetching from PubMed..."):
                    try:
                        paper_metadata = ingestion_engine.process_pubmed_id(pubmed_id)
                        st.success("Paper fetched successfully!")
                    except Exception as e:
                        st.error(f"Error fetching paper: {e}")
    
    elif input_method == "DOI":
        doi = st.text_input(
            "Enter DOI",
            placeholder="e.g., 10.1038/nature12345",
            help="Enter the DOI of the paper"
        )
        
        if st.button("Fetch Paper", key="doi_fetch"):
            if doi:
                with st.spinner("Fetching by DOI..."):
                    try:
                        paper_metadata = ingestion_engine.process_doi(doi)
                        st.success("Paper fetched successfully!")
                    except Exception as e:
                        st.error(f"Error fetching paper: {e}")
    
    # Display paper metadata
    if paper_metadata:
        st.session_state.paper_metadata = paper_metadata
        st.session_state.current_step = 2
        
        with st.expander("📖 Paper Details", expanded=True):
            st.write(f"**Title:** {paper_metadata.title}")
            if paper_metadata.authors:
                st.write(f"**Authors:** {', '.join(paper_metadata.authors)}")
            if paper_metadata.journal:
                st.write(f"**Journal:** {paper_metadata.journal} ({paper_metadata.year})")
            if paper_metadata.doi:
                st.write(f"**DOI:** {paper_metadata.doi}")
            if paper_metadata.abstract:
                st.write(f"**Abstract:** {paper_metadata.abstract}")

def method_extraction_section():
    if not st.session_state.paper_metadata:
        st.warning("Please load a paper first in the Paper Ingestion section.")
        return
    
    st.markdown('<div class="section-header">🔍 Method Extraction</div>', unsafe_allow_html=True)
    
    if st.button("Extract Computational Methods", key="extract_methods"):
        extractor = MethodExtractor()
        
        with st.spinner("Analyzing paper with Claude AI..."):
            try:
                extracted_methods = extractor.extract_methods(
                    st.session_state.paper_metadata.full_text,
                    st.session_state.paper_metadata.title
                )
                st.session_state.extracted_methods = extracted_methods
                st.session_state.current_step = 3
                st.success("Methods extracted successfully!")
            except Exception as e:
                st.error(f"Error extracting methods: {e}")
                return
    
    # Display extracted methods
    if st.session_state.extracted_methods:
        methods = st.session_state.extracted_methods
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🛠️ Computational Methods")
            for method in methods.computational_methods:
                with st.expander(f"{method.name} ({method.category})"):
                    st.write(f"**Description:** {method.description}")
                    if method.software_tools:
                        st.write(f"**Tools:** {', '.join(method.software_tools)}")
                    if method.programming_languages:
                        st.write(f"**Languages:** {', '.join(method.programming_languages)}")
                    if method.parameters:
                        st.write(f"**Parameters:** {method.parameters}")
            
            st.subheader("📊 Datasets")
            for dataset in methods.datasets:
                with st.expander(dataset.name):
                    st.write(f"**Description:** {dataset.description}")
                    st.write(f"**Source:** {dataset.source}")
                    st.write(f"**Format:** {dataset.format}")
                    if dataset.size:
                        st.write(f"**Size:** {dataset.size}")
        
        with col2:
            st.subheader("🔄 Workflows")
            for workflow in methods.workflows:
                with st.expander(workflow.name):
                    st.write("**Steps:**")
                    for i, step in enumerate(workflow.steps, 1):
                        st.write(f"{i}. {step}")
                    if workflow.input_data:
                        st.write(f"**Input:** {', '.join(workflow.input_data)}")
                    if workflow.output_data:
                        st.write(f"**Output:** {', '.join(workflow.output_data)}")
                    if workflow.dependencies:
                        st.write(f"**Dependencies:** {', '.join(workflow.dependencies)}")
            
            st.subheader("🔑 Key Findings")
            for finding in methods.key_findings:
                st.write(f"• {finding}")
            
            if methods.reproducibility_notes:
                st.subheader("♻️ Reproducibility Notes")
                st.write(methods.reproducibility_notes)

def code_generation_section():
    if not st.session_state.extracted_methods:
        st.warning("Please extract methods first.")
        return
    
    st.markdown('<div class="section-header">💻 Code Generation</div>', unsafe_allow_html=True)
    
    language = st.selectbox(
        "Select programming language:",
        ["python", "r"],
        index=0
    )
    
    if st.button("Generate Code", key="generate_code"):
        generator = CodeGenerator()
        
        with st.spinner("Generating reproducible code..."):
            try:
                generated_scripts = generator.generate_code_from_methods(
                    st.session_state.extracted_methods,
                    st.session_state.paper_metadata.title,
                    language
                )
                st.session_state.generated_code = generated_scripts
                st.session_state.current_step = 4
                st.success(f"Generated {len(generated_scripts)} code scripts!")
            except Exception as e:
                st.error(f"Error generating code: {e}")
                return
    
    # Display generated code
    if st.session_state.generated_code:
        for i, script in enumerate(st.session_state.generated_code):
            st.subheader(f"Script {i+1}: {script.description}")
            
            # Code display with syntax highlighting
            st.code(script.script_content, language=script.language)
            
            # Download button
            st.download_button(
                label=f"Download Script {i+1}",
                data=script.script_content,
                file_name=f"script_{i+1}.{script.language}",
                mime="text/plain"
            )
            
            # Dependencies and usage
            with st.expander("Dependencies and Usage"):
                st.write("**Dependencies:**")
                for dep in script.dependencies:
                    st.write(f"• {dep}")
                
                st.write("**Usage Instructions:**")
                st.write(script.usage_instructions)
            
            st.divider()

def qa_interface_section():
    if not st.session_state.paper_metadata or not st.session_state.extracted_methods:
        st.warning("Please load a paper and extract methods first.")
        return
    
    st.markdown('<div class="section-header">❓ Interactive Q&A</div>', unsafe_allow_html=True)
    
    # Load paper context into QA interface
    if st.session_state.paper_metadata and st.session_state.extracted_methods:
        st.session_state.qa_interface.load_paper_context(
            st.session_state.paper_metadata,
            st.session_state.extracted_methods
        )
    
    # Suggested questions
    with st.expander("💡 Suggested Questions"):
        suggestions = st.session_state.qa_interface.suggest_questions()
        for suggestion in suggestions:
            if st.button(suggestion, key=f"suggestion_{hash(suggestion)}"):
                st.session_state.current_question = suggestion
    
    # Question input
    question = st.text_area(
        "Ask a question about the paper's methodology:",
        value=getattr(st.session_state, 'current_question', ''),
        placeholder="e.g., What parameters were used for the machine learning model?",
        height=100
    )
    
    if st.button("Ask Question", key="ask_question"):
        if question:
            with st.spinner("Analyzing question..."):
                answer = st.session_state.qa_interface.ask_question(question)
                st.success("Question answered!")
            
            # Clear the current question
            if hasattr(st.session_state, 'current_question'):
                delattr(st.session_state, 'current_question')
    
    # Display conversation history
    history = st.session_state.qa_interface.get_conversation_history()
    if history:
        st.subheader("💬 Conversation History")
        for exchange in reversed(history):
            with st.expander(f"Q: {exchange.question[:100]}..."):
                st.write(f"**Question:** {exchange.question}")
                st.write(f"**Answer:** {exchange.answer}")
                st.write(f"**Time:** {exchange.timestamp}")
        
        # Export conversation
        if st.button("Export Conversation"):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"biolit_conversation_{timestamp}.json"
            
            # Create temporary file for download
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as tmp_file:
                st.session_state.qa_interface.export_conversation(tmp_file.name)
                with open(tmp_file.name, 'r') as f:
                    conversation_data = f.read()
            
            st.download_button(
                label="Download Conversation",
                data=conversation_data,
                file_name=filename,
                mime="application/json"
            )

def sidebar():
    with st.sidebar:
        st.header("🔬 BioLit Miner")
        
        # Progress indicator
        steps = [
            "📄 Paper Ingestion",
            "🔍 Method Extraction", 
            "💻 Code Generation",
            "❓ Interactive Q&A"
        ]
        
        current_step = st.session_state.current_step
        
        for i, step in enumerate(steps, 1):
            if i < current_step:
                st.success(f"✅ {step}")
            elif i == current_step:
                st.info(f"🔄 {step}")
            else:
                st.write(f"⏳ {step}")
        
        st.divider()
        
        # API Key status
        api_key = os.getenv('ANTHROPIC_API_KEY')
        if api_key:
            st.success("✅ API Key configured")
        else:
            st.error("❌ API Key not found")
            st.info("Set ANTHROPIC_API_KEY in your .env file")
        
        st.divider()
        
        # Clear session
        if st.button("🔄 Reset Session"):
            for key in ['paper_metadata', 'extracted_methods', 'generated_code']:
                if key in st.session_state:
                    del st.session_state[key]
            st.session_state.qa_interface = InteractiveQA()
            st.session_state.current_step = 1
            st.experimental_rerun()

def main():
    initialize_session_state()
    
    display_header()
    sidebar()
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📄 Paper Ingestion",
        "🔍 Method Extraction", 
        "💻 Code Generation",
        "❓ Q&A Interface"
    ])
    
    with tab1:
        paper_ingestion_section()
    
    with tab2:
        method_extraction_section()
    
    with tab3:
        code_generation_section()
    
    with tab4:
        qa_interface_section()

if __name__ == "__main__":
    main()