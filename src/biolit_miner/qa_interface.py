import os
from typing import List, Dict, Optional
from dataclasses import dataclass
from anthropic import Anthropic
from dotenv import load_dotenv
from .paper_ingestion import PaperMetadata
from .method_extractor import ExtractedMethods

load_dotenv()

@dataclass
class QAExchange:
    question: str
    answer: str
    timestamp: str
    context_used: str

class InteractiveQA:
    def __init__(self):
        api_key = os.getenv('ANTHROPIC_API_KEY')
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not found in environment variables")
        self.client = Anthropic(api_key=api_key)
        
        self.paper_metadata: Optional[PaperMetadata] = None
        self.extracted_methods: Optional[ExtractedMethods] = None
        self.conversation_history: List[QAExchange] = []
    
    def load_paper_context(self, paper_metadata: PaperMetadata, 
                          extracted_methods: ExtractedMethods):
        self.paper_metadata = paper_metadata
        self.extracted_methods = extracted_methods
    
    def ask_question(self, question: str) -> str:
        if not self.paper_metadata or not self.extracted_methods:
            return "Please load a paper first before asking questions."
        
        # Create context-aware prompt
        prompt = self._create_qa_prompt(question)
        
        try:
            response = self.client.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=2000,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            answer = response.content[0].text
            
            # Store in conversation history
            self._add_to_history(question, answer)
            
            return answer
            
        except Exception as e:
            return f"Error processing question: {e}"
    
    def _create_qa_prompt(self, question: str) -> str:
        # Prepare paper context
        paper_context = f"""
Paper Title: {self.paper_metadata.title}
Authors: {', '.join(self.paper_metadata.authors)}
Journal: {self.paper_metadata.journal} ({self.paper_metadata.year})
DOI: {self.paper_metadata.doi}

Abstract:
{self.paper_metadata.abstract}

Full Text (excerpt):
{self.paper_metadata.full_text[:8000]}  # Limit to avoid token limits
"""

        # Prepare methods context
        methods_context = "Extracted Computational Methods:\n"
        for method in self.extracted_methods.computational_methods:
            methods_context += f"""
- {method.name}:
  Description: {method.description}
  Software Tools: {', '.join(method.software_tools)}
  Programming Languages: {', '.join(method.programming_languages)}
  Parameters: {method.parameters}
  Category: {method.category}
"""

        datasets_context = "Datasets Used:\n"
        for dataset in self.extracted_methods.datasets:
            datasets_context += f"""
- {dataset.name}:
  Description: {dataset.description}
  Source: {dataset.source}
  Format: {dataset.format}
"""

        workflows_context = "Analysis Workflows:\n"
        for workflow in self.extracted_methods.workflows:
            workflows_context += f"""
- {workflow.name}:
  Steps: {' → '.join(workflow.steps)}
  Input Data: {', '.join(workflow.input_data)}
  Output Data: {', '.join(workflow.output_data)}
  Dependencies: {', '.join(workflow.dependencies)}
"""

        # Include recent conversation history for context
        history_context = ""
        if self.conversation_history:
            history_context = "Recent Conversation:\n"
            for exchange in self.conversation_history[-3:]:  # Last 3 exchanges
                history_context += f"Q: {exchange.question}\nA: {exchange.answer}\n\n"

        return f"""
You are an expert research assistant helping to answer questions about a scientific paper's methodology and computational approaches.

{paper_context}

{methods_context}

{datasets_context}

{workflows_context}

{history_context}

Key Findings from the Paper:
{chr(10).join(self.extracted_methods.key_findings)}

Reproducibility Notes:
{self.extracted_methods.reproducibility_notes}

User Question: {question}

Please provide a detailed, accurate answer based on the paper's content and methodology. Focus on:
1. Specific methodological details
2. Computational approaches and parameters
3. Data processing steps
4. Software tools and implementations
5. Reproducibility considerations

If the question cannot be answered from the provided paper content, clearly state that and suggest what additional information might be needed.
"""

    def _add_to_history(self, question: str, answer: str):
        from datetime import datetime
        
        exchange = QAExchange(
            question=question,
            answer=answer,
            timestamp=datetime.now().isoformat(),
            context_used="paper_content_and_methods"
        )
        self.conversation_history.append(exchange)
        
        # Keep only last 10 exchanges to manage memory
        if len(self.conversation_history) > 10:
            self.conversation_history = self.conversation_history[-10:]
    
    def get_conversation_history(self) -> List[QAExchange]:
        return self.conversation_history.copy()
    
    def clear_conversation(self):
        self.conversation_history.clear()
    
    def suggest_questions(self) -> List[str]:
        if not self.paper_metadata or not self.extracted_methods:
            return ["Please load a paper first."]
        
        # Generate context-aware question suggestions
        prompt = f"""
Based on this research paper's methodology and computational approaches, suggest 5-7 relevant questions that would help someone understand or reproduce the research.

Paper Title: {self.paper_metadata.title}

Computational Methods:
{chr(10).join([f"- {method.name}: {method.description}" for method in self.extracted_methods.computational_methods])}

Datasets:
{chr(10).join([f"- {dataset.name}: {dataset.description}" for dataset in self.extracted_methods.datasets])}

Workflows:
{chr(10).join([f"- {workflow.name}" for workflow in self.extracted_methods.workflows])}

Generate questions that cover:
1. Methodology clarification
2. Parameter settings
3. Software implementation details
4. Data preprocessing steps
5. Reproducibility aspects
6. Alternative approaches
7. Results interpretation

Return only the questions, one per line, without numbering or additional text.
"""
        
        try:
            response = self.client.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=1000,
                messages=[{"role": "user", "content": prompt}]
            )
            
            questions = response.content[0].text.strip().split('\n')
            return [q.strip() for q in questions if q.strip()]
            
        except Exception as e:
            return [
                "What computational methods were used in this study?",
                "What software tools and programming languages were employed?",
                "How was the data preprocessed?",
                "What were the key parameters used in the analysis?",
                "Is the code available for reproduction?",
                "What datasets were used and where can I access them?",
                "What are the main computational findings?"
            ]
    
    def export_conversation(self, output_path: str):
        import json
        
        export_data = {
            'paper_metadata': {
                'title': self.paper_metadata.title if self.paper_metadata else None,
                'authors': self.paper_metadata.authors if self.paper_metadata else [],
                'doi': self.paper_metadata.doi if self.paper_metadata else None
            },
            'conversation_history': [
                {
                    'question': exchange.question,
                    'answer': exchange.answer,
                    'timestamp': exchange.timestamp
                }
                for exchange in self.conversation_history
            ]
        }
        
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2)
    
    def load_conversation(self, input_path: str):
        import json
        from datetime import datetime
        
        with open(input_path, 'r') as f:
            data = json.load(f)
        
        self.conversation_history = [
            QAExchange(
                question=item['question'],
                answer=item['answer'],
                timestamp=item['timestamp'],
                context_used="paper_content_and_methods"
            )
            for item in data['conversation_history']
        ]