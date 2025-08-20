import os
import requests
import PyPDF2
import pdfplumber
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import re
import xml.etree.ElementTree as ET

@dataclass
class PaperMetadata:
    title: str
    authors: List[str]
    abstract: str
    doi: str
    pubmed_id: str
    journal: str
    year: str
    full_text: str

class PaperIngestionEngine:
    def __init__(self):
        self.pubmed_base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        text = ""
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
        except Exception as e:
            try:
                with open(pdf_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    for page in pdf_reader.pages:
                        text += page.extract_text() + "\n"
            except Exception as e2:
                raise Exception(f"Failed to extract text from PDF: {e2}")
        return text.strip()
    
    def fetch_pubmed_metadata(self, pubmed_id: str) -> Dict:
        try:
            url = f"{self.pubmed_base_url}efetch.fcgi"
            params = {
                'db': 'pubmed',
                'id': pubmed_id,
                'retmode': 'xml'
            }
            response = requests.get(url, params=params)
            response.raise_for_status()
            
            root = ET.fromstring(response.content)
            article = root.find('.//Article')
            
            if article is None:
                raise Exception("Article not found in PubMed response")
            
            title = article.find('.//ArticleTitle').text if article.find('.//ArticleTitle') is not None else ""
            abstract_elem = article.find('.//Abstract/AbstractText')
            abstract = abstract_elem.text if abstract_elem is not None else ""
            
            authors = []
            author_list = article.find('.//AuthorList')
            if author_list:
                for author in author_list.findall('.//Author'):
                    last_name = author.find('.//LastName')
                    first_name = author.find('.//ForeName')
                    if last_name is not None:
                        name = last_name.text
                        if first_name is not None:
                            name = f"{first_name.text} {name}"
                        authors.append(name)
            
            journal_elem = article.find('.//Journal/Title')
            journal = journal_elem.text if journal_elem is not None else ""
            
            year_elem = article.find('.//PubDate/Year')
            year = year_elem.text if year_elem is not None else ""
            
            doi_elem = article.find('.//ELocationID[@EIdType="doi"]')
            doi = doi_elem.text if doi_elem is not None else ""
            
            return {
                'title': title,
                'authors': authors,
                'abstract': abstract,
                'journal': journal,
                'year': year,
                'doi': doi
            }
        except Exception as e:
            raise Exception(f"Failed to fetch PubMed metadata: {e}")
    
    def search_pubmed_by_doi(self, doi: str) -> str:
        try:
            url = f"{self.pubmed_base_url}esearch.fcgi"
            params = {
                'db': 'pubmed',
                'term': f'"{doi}"[DOI]',
                'retmode': 'xml'
            }
            response = requests.get(url, params=params)
            response.raise_for_status()
            
            root = ET.fromstring(response.content)
            id_elem = root.find('.//Id')
            if id_elem is not None:
                return id_elem.text
            else:
                raise Exception("No PubMed ID found for the given DOI")
        except Exception as e:
            raise Exception(f"Failed to search PubMed by DOI: {e}")
    
    def process_pdf_upload(self, pdf_path: str) -> PaperMetadata:
        full_text = self.extract_text_from_pdf(pdf_path)
        
        doi_match = re.search(r'doi:?\s*(10\.\d+/[^\s]+)', full_text, re.IGNORECASE)
        doi = doi_match.group(1) if doi_match else ""
        
        pubmed_id = ""
        if doi:
            try:
                pubmed_id = self.search_pubmed_by_doi(doi)
            except:
                pass
        
        if pubmed_id:
            try:
                metadata = self.fetch_pubmed_metadata(pubmed_id)
                return PaperMetadata(
                    title=metadata['title'],
                    authors=metadata['authors'],
                    abstract=metadata['abstract'],
                    doi=metadata['doi'] or doi,
                    pubmed_id=pubmed_id,
                    journal=metadata['journal'],
                    year=metadata['year'],
                    full_text=full_text
                )
            except:
                pass
        
        title_match = re.search(r'^([^\n]{10,200})', full_text.strip())
        title = title_match.group(1).strip() if title_match else "Unknown Title"
        
        return PaperMetadata(
            title=title,
            authors=[],
            abstract="",
            doi=doi,
            pubmed_id=pubmed_id,
            journal="",
            year="",
            full_text=full_text
        )
    
    def process_pubmed_id(self, pubmed_id: str) -> PaperMetadata:
        metadata = self.fetch_pubmed_metadata(pubmed_id)
        return PaperMetadata(
            title=metadata['title'],
            authors=metadata['authors'],
            abstract=metadata['abstract'],
            doi=metadata['doi'],
            pubmed_id=pubmed_id,
            journal=metadata['journal'],
            year=metadata['year'],
            full_text=""  # PubMed doesn't provide full text
        )
    
    def process_doi(self, doi: str) -> PaperMetadata:
        try:
            pubmed_id = self.search_pubmed_by_doi(doi)
            return self.process_pubmed_id(pubmed_id)
        except:
            return PaperMetadata(
                title="Unknown Title",
                authors=[],
                abstract="",
                doi=doi,
                pubmed_id="",
                journal="",
                year="",
                full_text=""
            )