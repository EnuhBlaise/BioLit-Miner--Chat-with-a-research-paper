# BioLit Miner 🔬

A comprehensive research paper analysis tool that extracts computational methods, generates reproducible code, and provides interactive Q&A capabilities for scientific literature.

## Features

### 📄 Paper Ingestion
- **PDF Upload**: Extract text and metadata from research paper PDFs
- **PubMed Integration**: Fetch papers using PubMed IDs
- **DOI Lookup**: Retrieve papers using DOI identifiers
- **Automatic Metadata Extraction**: Extract title, authors, abstract, and bibliographic information

### 🔍 Method Extraction
- **AI-Powered Analysis**: Use Claude AI to identify computational methods
- **Structured Extraction**: Extract methods, datasets, workflows, and parameters
- **Categorization**: Organize methods by type (statistical analysis, machine learning, bioinformatics)
- **Reproducibility Assessment**: Identify code availability and data sharing practices

### 💻 Code Generation
- **Template-Based Generation**: Create Python/R scripts from extracted methods
- **Multiple Categories**: Support for statistical analysis, machine learning, and bioinformatics
- **Customizable Templates**: Modular code templates for different analysis types
- **Dependencies Management**: Automatic identification of required packages

### ❓ Interactive Q&A
- **Context-Aware Responses**: Ask questions about paper methodology
- **Conversation History**: Track and export Q&A sessions
- **Suggested Questions**: AI-generated relevant questions
- **Export Functionality**: Save conversations for future reference

## Installation

1. **Clone the repository**:
```bash
git clone <repository-url>
cd biolit_miner
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Set up environment variables**:
```bash
cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY
```

## Usage

### Web Interface

Launch the Streamlit web application:

```bash
streamlit run app.py
```

Navigate to `http://localhost:8501` in your browser to access the interface.

### Programmatic Usage

```python
from src.biolit_miner.paper_ingestion import PaperIngestionEngine
from src.biolit_miner.method_extractor import MethodExtractor
from src.biolit_miner.code_generator import CodeGenerator
from src.biolit_miner.qa_interface import InteractiveQA

# Initialize components
ingestion = PaperIngestionEngine()
extractor = MethodExtractor()
generator = CodeGenerator()
qa = InteractiveQA()

# Process a paper
paper_metadata = ingestion.process_pdf_upload("paper.pdf")
extracted_methods = extractor.extract_methods(paper_metadata.full_text, paper_metadata.title)

# Generate code
generated_scripts = generator.generate_code_from_methods(extracted_methods, paper_metadata.title)

# Q&A interface
qa.load_paper_context(paper_metadata, extracted_methods)
answer = qa.ask_question("What machine learning methods were used?")
```

## Project Structure

```
biolit_miner/
├── src/
│   └── biolit_miner/
│       ├── __init__.py
│       ├── paper_ingestion.py      # PDF/PubMed paper processing
│       ├── method_extractor.py     # AI-powered method extraction
│       ├── code_generator.py       # Code generation from methods
│       ├── qa_interface.py         # Interactive Q&A system
│       └── templates/              # Code generation templates
│           ├── statistical_analysis.py
│           ├── machine_learning.py
│           └── bioinformatics.py
├── app.py                          # Streamlit web interface
├── requirements.txt                # Python dependencies
├── .env.example                    # Environment variables template
└── README.md                       # Project documentation
```

## Dependencies

### Core Dependencies
- `streamlit`: Web interface framework
- `anthropic`: Claude AI API client
- `PyPDF2` & `pdfplumber`: PDF text extraction
- `requests`: HTTP requests for PubMed API
- `python-dotenv`: Environment variable management

### Data Analysis
- `pandas`: Data manipulation and analysis
- `numpy`: Numerical computing
- `matplotlib` & `seaborn`: Data visualization
- `scikit-learn`: Machine learning library

### Bioinformatics
- `biopython`: Biological sequence analysis

## Configuration

### Environment Variables

You can use any model provider of your choice, for this project I used Anthropic.
Create a `.env` file with:

```env
ANTHROPIC_API_KEY=your_anthropic_api_key_here
```

### API Keys

- **Anthropic API Key**: Required for Claude AI integration
  - You can check it out @ [https://www.anthropic.com/](https://www.anthropic.com/)
  - Generate an API key from the console
  - Add to your `.env` file

## Use Cases

### Research Paper Analysis
- Extract computational methods from papers
- Understand analysis workflows
- Identify software tools and parameters used

### Code Reproduction
- Generate Python/R scripts based on paper methods
- Reproduce key analyses from publications
- Create templates for similar analyses

### Literature Review
- Ask questions about methodology across papers
- Compare computational approaches
- Extract reproducibility information

### Educational
- Learn about computational methods in research
- Understand analysis workflows
- Practice implementing published methods

## Limitations

- **Full Text Access**: Some papers may only provide abstracts
- **Method Complexity**: Very complex methods may require manual refinement
- **Code Accuracy**: Generated code should be reviewed and tested
- **API Dependencies**: Requires internet connection for Claude AI and PubMed

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- **Anthropic Claude**: AI-powered text analysis and code generation
- **PubMed**: Research paper metadata and abstracts
- **Streamlit**: Web interface framework
- **BioPython**: Bioinformatics sequence analysis tools

## Support

For questions, issues, or feature requests, please:
1. Check the existing issues on GitHub
2. Create a new issue with detailed information
3. Contact the maintainers

---

**Note**: This tool is designed for research and educational purposes. Always verify generated code and extracted information against the original papers.
