import os
import re
from typing import Dict, List, Optional
from dataclasses import dataclass
from anthropic import Anthropic
from dotenv import load_dotenv
from .method_extractor import ExtractedMethods, ComputationalMethod

load_dotenv()

@dataclass
class GeneratedCode:
    script_content: str
    language: str
    dependencies: List[str]
    description: str
    usage_instructions: str

class CodeGenerator:
    def __init__(self):
        api_key = os.getenv('ANTHROPIC_API_KEY')
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not found in environment variables")
        self.client = Anthropic(api_key=api_key)
        
        # Load code templates
        self.templates_dir = os.path.join(os.path.dirname(__file__), 'templates')
        self.templates = self._load_templates()
    
    def _load_templates(self) -> Dict[str, str]:
        templates = {}
        if os.path.exists(self.templates_dir):
            for filename in os.listdir(self.templates_dir):
                if filename.endswith('.py'):
                    template_name = filename[:-3]  # Remove .py extension
                    template_path = os.path.join(self.templates_dir, filename)
                    with open(template_path, 'r') as f:
                        templates[template_name] = f.read()
        return templates
    
    def generate_code_from_methods(self, extracted_methods: ExtractedMethods, 
                                 paper_title: str = "", 
                                 preferred_language: str = "python") -> List[GeneratedCode]:
        generated_scripts = []
        
        # Group methods by category
        methods_by_category = {}
        for method in extracted_methods.computational_methods:
            category = method.category
            if category not in methods_by_category:
                methods_by_category[category] = []
            methods_by_category[category].append(method)
        
        # Generate code for each category
        for category, methods in methods_by_category.items():
            script = self._generate_category_script(
                category, methods, extracted_methods, paper_title, preferred_language
            )
            if script:
                generated_scripts.append(script)
        
        return generated_scripts
    
    def _generate_category_script(self, category: str, methods: List[ComputationalMethod],
                                extracted_methods: ExtractedMethods, paper_title: str,
                                language: str) -> Optional[GeneratedCode]:
        
        # Get appropriate template
        template = self._get_template_for_category(category)
        
        # Create detailed prompt for code generation
        prompt = self._create_code_generation_prompt(
            category, methods, extracted_methods, paper_title, language, template
        )
        
        try:
            response = self.client.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=4000,
                messages=[{"role": "user", "content": prompt}]
            )
            
            result_text = response.content[0].text
            return self._parse_code_generation_result(result_text, category, language)
            
        except Exception as e:
            print(f"Failed to generate code for category {category}: {e}")
            return None
    
    def _get_template_for_category(self, category: str) -> str:
        category_mapping = {
            'statistical_analysis': 'statistical_analysis',
            'machine_learning': 'machine_learning',
            'bioinformatics': 'bioinformatics',
            'data_analysis': 'statistical_analysis',
            'image_analysis': 'statistical_analysis'  # Default fallback
        }
        
        template_name = category_mapping.get(category, 'statistical_analysis')
        return self.templates.get(template_name, "")
    
    def _create_code_generation_prompt(self, category: str, methods: List[ComputationalMethod],
                                     extracted_methods: ExtractedMethods, paper_title: str,
                                     language: str, template: str) -> str:
        
        methods_text = "\n".join([
            f"- {method.name}: {method.description}\n"
            f"  Tools: {', '.join(method.software_tools)}\n"
            f"  Parameters: {method.parameters}\n"
            for method in methods
        ])
        
        datasets_text = "\n".join([
            f"- {dataset.name}: {dataset.description} (Format: {dataset.format})"
            for dataset in extracted_methods.datasets
        ])
        
        workflows_text = "\n".join([
            f"- {workflow.name}:\n" + 
            "\n".join([f"  {i+1}. {step}" for i, step in enumerate(workflow.steps)])
            for workflow in extracted_methods.workflows
        ])
        
        return f"""
Generate a complete {language} script that reproduces the computational analysis from this research paper.

Paper Title: {paper_title}

Category: {category}

Methods to implement:
{methods_text}

Available Datasets:
{datasets_text}

Analysis Workflows:
{workflows_text}

Template (modify and extend as needed):
{template}

Requirements:
1. Create a complete, runnable script in {language}
2. Include all necessary imports and dependencies
3. Implement the specific methods mentioned above
4. Add data loading functions that work with the specified dataset formats
5. Include proper error handling and logging
6. Add visualization functions where appropriate
7. Include parameter settings that match those described in the paper
8. Add comprehensive comments explaining each step
9. Make the code modular and reusable
10. Include example usage at the bottom

Return your response in this format:

```{language}
[COMPLETE SCRIPT CODE HERE]
```

DEPENDENCIES:
[LIST ALL REQUIRED PACKAGES/LIBRARIES]

DESCRIPTION:
[BRIEF DESCRIPTION OF WHAT THE SCRIPT DOES]

USAGE:
[INSTRUCTIONS ON HOW TO RUN THE SCRIPT]

Focus on creating production-ready code that closely follows the methodology described in the paper.
"""

    def _parse_code_generation_result(self, result_text: str, category: str, language: str) -> GeneratedCode:
        # Extract code block
        code_pattern = f"```{language}(.*?)```"
        code_match = re.search(code_pattern, result_text, re.DOTALL | re.IGNORECASE)
        
        if not code_match:
            # Try generic code block
            code_match = re.search(r"```(.*?)```", result_text, re.DOTALL)
        
        script_content = code_match.group(1).strip() if code_match else result_text
        
        # Extract dependencies
        deps_pattern = r"DEPENDENCIES:\s*(.*?)(?=DESCRIPTION:|$)"
        deps_match = re.search(deps_pattern, result_text, re.DOTALL | re.IGNORECASE)
        dependencies = []
        if deps_match:
            deps_text = deps_match.group(1).strip()
            dependencies = [dep.strip() for dep in deps_text.split('\n') if dep.strip()]
        
        # Extract description
        desc_pattern = r"DESCRIPTION:\s*(.*?)(?=USAGE:|$)"
        desc_match = re.search(desc_pattern, result_text, re.DOTALL | re.IGNORECASE)
        description = desc_match.group(1).strip() if desc_match else f"Generated {category} analysis script"
        
        # Extract usage instructions
        usage_pattern = r"USAGE:\s*(.*?)$"
        usage_match = re.search(usage_pattern, result_text, re.DOTALL | re.IGNORECASE)
        usage_instructions = usage_match.group(1).strip() if usage_match else "Run the script with your data files"
        
        return GeneratedCode(
            script_content=script_content,
            language=language,
            dependencies=dependencies,
            description=description,
            usage_instructions=usage_instructions
        )
    
    def save_generated_code(self, generated_code: GeneratedCode, 
                          output_dir: str, filename: str):
        os.makedirs(output_dir, exist_ok=True)
        
        # Save script
        script_path = os.path.join(output_dir, f"{filename}.py")
        with open(script_path, 'w') as f:
            f.write(generated_code.script_content)
        
        # Save metadata
        metadata_path = os.path.join(output_dir, f"{filename}_metadata.txt")
        with open(metadata_path, 'w') as f:
            f.write(f"Description: {generated_code.description}\n\n")
            f.write(f"Dependencies:\n")
            for dep in generated_code.dependencies:
                f.write(f"- {dep}\n")
            f.write(f"\nUsage Instructions:\n{generated_code.usage_instructions}")
        
        return script_path, metadata_path
    
    def generate_requirements_file(self, generated_scripts: List[GeneratedCode], 
                                 output_path: str):
        all_deps = set()
        for script in generated_scripts:
            all_deps.update(script.dependencies)
        
        with open(output_path, 'w') as f:
            for dep in sorted(all_deps):
                f.write(f"{dep}\n")
        
        return output_path