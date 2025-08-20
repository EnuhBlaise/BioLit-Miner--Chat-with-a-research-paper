import os
import json
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()

@dataclass
class ComputationalMethod:
    name: str
    description: str
    software_tools: List[str]
    programming_languages: List[str]
    parameters: Dict[str, str]
    category: str  # e.g., "statistical_analysis", "machine_learning", "bioinformatics"

@dataclass
class Dataset:
    name: str
    description: str
    source: str
    format: str
    size: Optional[str] = None

@dataclass
class Workflow:
    name: str
    steps: List[str]
    input_data: List[str]
    output_data: List[str]
    dependencies: List[str]

@dataclass
class ExtractedMethods:
    computational_methods: List[ComputationalMethod]
    datasets: List[Dataset]
    workflows: List[Workflow]
    key_findings: List[str]
    reproducibility_notes: str

class MethodExtractor:
    def __init__(self):
        api_key = os.getenv('ANTHROPIC_API_KEY')
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not found in environment variables")
        self.client = Anthropic(api_key=api_key)
    
    def extract_methods(self, paper_text: str, paper_title: str = "") -> ExtractedMethods:
        prompt = self._create_extraction_prompt(paper_text, paper_title)
        
        try:
            response = self.client.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=4000,
                messages=[{"role": "user", "content": prompt}]
            )
            
            result_text = response.content[0].text
            return self._parse_extraction_result(result_text)
            
        except Exception as e:
            raise Exception(f"Failed to extract methods using Claude: {e}")
    
    def _create_extraction_prompt(self, paper_text: str, paper_title: str = "") -> str:
        return f"""
Analyze the following research paper and extract computational methods, datasets, and workflows in a structured JSON format.

Paper Title: {paper_title}

Paper Text:
{paper_text[:15000]}  # Limit text to avoid token limits

Please extract and structure the following information as a JSON object:

1. **Computational Methods**: Identify all computational approaches, algorithms, statistical methods, and analysis techniques used. For each method include:
   - name: The name/type of method
   - description: Brief description of what it does
   - software_tools: List of software/tools mentioned (e.g., R, Python, MATLAB, specific packages)
   - programming_languages: Languages used
   - parameters: Key parameters or settings mentioned
   - category: Type of analysis (statistical_analysis, machine_learning, bioinformatics, image_analysis, etc.)

2. **Datasets**: Identify all datasets used. For each dataset include:
   - name: Dataset name or identifier
   - description: What the dataset contains
   - source: Where it came from (database, experiment, etc.)
   - format: File format or data type
   - size: Size if mentioned

3. **Workflows**: Identify analysis workflows or pipelines. For each workflow include:
   - name: Workflow name or purpose
   - steps: Ordered list of analysis steps
   - input_data: What data goes in
   - output_data: What results are produced
   - dependencies: Required software/tools

4. **Key Findings**: List the main computational/methodological findings or results

5. **Reproducibility Notes**: Any notes about code availability, data sharing, or reproducibility

Return your response as a valid JSON object with the following structure:
{{
  "computational_methods": [
    {{
      "name": "string",
      "description": "string",
      "software_tools": ["string"],
      "programming_languages": ["string"],
      "parameters": {{"param": "value"}},
      "category": "string"
    }}
  ],
  "datasets": [
    {{
      "name": "string",
      "description": "string",
      "source": "string",
      "format": "string",
      "size": "string"
    }}
  ],
  "workflows": [
    {{
      "name": "string",
      "steps": ["string"],
      "input_data": ["string"],
      "output_data": ["string"],
      "dependencies": ["string"]
    }}
  ],
  "key_findings": ["string"],
  "reproducibility_notes": "string"
}}

Focus on extracting concrete, actionable information that could be used to reproduce the computational aspects of the research.
"""

    def _parse_extraction_result(self, result_text: str) -> ExtractedMethods:
        try:
            json_start = result_text.find('{')
            json_end = result_text.rfind('}') + 1
            
            if json_start == -1 or json_end == 0:
                raise ValueError("No JSON found in response")
            
            json_str = result_text[json_start:json_end]
            data = json.loads(json_str)
            
            computational_methods = [
                ComputationalMethod(**method) 
                for method in data.get('computational_methods', [])
            ]
            
            datasets = [
                Dataset(**dataset) 
                for dataset in data.get('datasets', [])
            ]
            
            workflows = [
                Workflow(**workflow) 
                for workflow in data.get('workflows', [])
            ]
            
            return ExtractedMethods(
                computational_methods=computational_methods,
                datasets=datasets,
                workflows=workflows,
                key_findings=data.get('key_findings', []),
                reproducibility_notes=data.get('reproducibility_notes', '')
            )
            
        except json.JSONDecodeError as e:
            raise Exception(f"Failed to parse JSON response: {e}")
        except Exception as e:
            raise Exception(f"Failed to parse extraction result: {e}")
    
    def save_extracted_methods(self, methods: ExtractedMethods, output_path: str):
        with open(output_path, 'w') as f:
            json.dump(asdict(methods), f, indent=2)
    
    def load_extracted_methods(self, input_path: str) -> ExtractedMethods:
        with open(input_path, 'r') as f:
            data = json.load(f)
        
        return ExtractedMethods(
            computational_methods=[
                ComputationalMethod(**method) 
                for method in data['computational_methods']
            ],
            datasets=[
                Dataset(**dataset) 
                for dataset in data['datasets']
            ],
            workflows=[
                Workflow(**workflow) 
                for workflow in data['workflows']
            ],
            key_findings=data['key_findings'],
            reproducibility_notes=data['reproducibility_notes']
        )