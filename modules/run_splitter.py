# File: enhanced_intelligent_document_processor.py
import json
import re
import fitz #type:ignore
from docx import Document #type:ignore
import requests
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from typing import List, Dict, Tuple
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
import time
import math
from collections import defaultdict
import multidocs

class EnhancedIntelligentDocumentProcessor:
    def __init__(self, model_name: str = "llama3"):
        """
        Initialize processor for keypoint extraction and optional PDF summarization
        """
        self.model_name = model_name
        self.min_compression = 0.15  # More aggressive for PDF summary
        self.max_compression = 0.25  # Tighter compression for PDF
        self.max_workers = 4
        self.complete_pdf_summary = None  # Store optional PDF summary

    @staticmethod
    def get_file_extension(file_path: str) -> str:
        """Get lowercase file extension"""
        return os.path.splitext(file_path)[1].lower()

    def extract_text_from_file(self, file_path: str) -> Tuple[str, Dict]:
        """
        Extract text from multiple file formats with position tracking
        Returns: (full_text, position_map)
        """
        ext = self.get_file_extension(file_path)
        
        if ext == '.pdf':
            return self.extract_text_with_page_tracking(file_path)
        elif ext == '.docx':
            return self._extract_text_from_docx(file_path)
        elif ext in ('.txt', '.md'):
            return self._extract_text_from_plaintext(file_path)
        else:
            raise ValueError(f"Unsupported file format: {ext}. Supported: PDF, DOCX, TXT, MD")

    def _extract_text_from_docx(self, docx_path: str) -> Tuple[str, Dict]:
        """Extract text from DOCX files with paragraph tracking"""
        try:
            print(f"📖 Opening DOCX: {docx_path}")
            doc = Document(docx_path)
            full_text = []
            position_map = {}
            current_pos = 0
            
            for para_num, paragraph in enumerate(doc.paragraphs, 1):
                text = paragraph.text.strip()
                if text:
                    full_text.append(text)
                    position_map[current_pos] = {
                        'page': 1,  # DOCX doesn't have pages
                        'line': para_num,
                        'page_line_key': f"Paragraph {para_num}",
                        'text_preview': text[:50] + "..." if len(text) > 50 else text
                    }
                    current_pos += len(text) + 2  # +2 for paragraph breaks
            
            result_text = "\n\n".join(full_text)
            print(f"✅ Extracted {len(result_text):,} characters from DOCX")
            return result_text, position_map
            
        except Exception as e:
            print(f"❌ Error reading DOCX: {e}")
            return "", {}

    def _extract_text_from_plaintext(self, file_path: str) -> Tuple[str, Dict]:
        """Extract text from plain text files with line tracking"""
        try:
            print(f"📖 Opening text file: {file_path}")
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            full_text = "".join(lines)
            position_map = {}
            current_pos = 0
            
            for line_num, line in enumerate(lines, 1):
                if line.strip():
                    position_map[current_pos] = {
                        'page': 1,
                        'line': line_num,
                        'page_line_key': f"Line {line_num}",
                        'text_preview': line[:50] + "..." if len(line) > 50 else line.strip()
                    }
                current_pos += len(line)
            
            print(f"✅ Extracted {len(full_text):,} characters from text file")
            return full_text.strip(), position_map
            
        except Exception as e:
            print(f"❌ Error reading text file: {e}")
            return "", {}
        
    def analyze_document_structure(self, text: str) -> Dict:
        """Analyze document to determine optimal processing strategy"""
        # Basic document analysis
        total_chars = len(text)
        total_words = len(text.split())
        paragraphs = text.split('\n\n')
        
        # Detect document type and complexity
        academic_keywords = ['abstract', 'introduction', 'methodology', 'conclusion', 'references', 'figure', 'table']
        technical_keywords = ['algorithm', 'implementation', 'system', 'architecture', 'framework']
        
        academic_score = sum(1 for keyword in academic_keywords if keyword.lower() in text.lower())
        technical_score = sum(1 for keyword in technical_keywords if keyword.lower() in text.lower())
        
        # Determine document complexity
        avg_para_length = sum(len(p) for p in paragraphs) / len(paragraphs) if paragraphs else 100
        complexity_score = (academic_score + technical_score) / 10
        
        # Calculate optimal compression for JSON keypoints (preserve more detail)
        json_target_compression = 0.60  # Keep 60% of content for keypoints
        
        # Calculate aggressive compression for optional PDF summary
        if complexity_score > 0.7:  # High complexity (academic/technical)
            pdf_target_compression = 0.20  # 20% - very compressed
        elif complexity_score > 0.4:  # Medium complexity
            pdf_target_compression = 0.18  # 18% - aggressive
        else:  # Lower complexity
            pdf_target_compression = 0.15  # 15% - most aggressive
        
        analysis = {
            'total_chars': total_chars,
            'total_words': total_words,
            'paragraphs': len(paragraphs),
            'avg_paragraph_length': avg_para_length,
            'complexity_score': complexity_score,
            'academic_score': academic_score,
            'technical_score': technical_score,
            'json_target_compression': json_target_compression,
            'pdf_target_compression': pdf_target_compression,
            'estimated_json_length': int(total_chars * json_target_compression),
            'estimated_pdf_summary_length': int(total_chars * pdf_target_compression)
        }
        
        print(f"📊 Document Analysis:")
        print(f"   • Length: {total_chars:,} chars, {total_words:,} words")
        print(f"   • Complexity: {complexity_score:.2f} (Academic: {academic_score}, Technical: {technical_score})")
        print(f"   • JSON keypoints target: {json_target_compression:.1%} (~{analysis['estimated_json_length']:,} chars)")
        print(f"   • PDF summary target: {pdf_target_compression:.1%} (~{analysis['estimated_pdf_summary_length']:,} chars)")
        
        return analysis

    def extract_text_with_page_tracking(self, pdf_path: str) -> Tuple[str, Dict]:
        """Enhanced text extraction with page and line number tracking"""
        doc = None
        try:
            if not os.path.exists(pdf_path):
                print(f"❌ PDF file not found: {pdf_path}")
                return "", {}
            
            print(f"📖 Opening PDF: {pdf_path}")
            doc = fitz.open(pdf_path)
            
            if doc.is_closed or doc.page_count == 0:
                print("❌ PDF document invalid or empty")
                return "", {}
            
            print(f"📄 Processing {doc.page_count} pages with tracking...")
            full_text = ""
            page_map = {}  # Maps character positions to page/line info
            current_char_pos = 0
            
            for page_num in range(doc.page_count):
                try:
                    page = doc.load_page(page_num)
                    page_text = page.get_text()
                    
                    if page_text.strip():
                        lines = page_text.split('\n')
                        
                        for line_num, line in enumerate(lines, 1):
                            if line.strip():  # Only track non-empty lines
                                line_start_pos = current_char_pos
                                
                                # Store mapping for this line
                                page_map[line_start_pos] = {
                                    'page': page_num + 1,
                                    'line': line_num,
                                    'page_line_key': f"Page {page_num + 1}, Line {line_num}",
                                    'text_preview': line[:50] + "..." if len(line) > 50 else line
                                }
                            
                            current_char_pos += len(line) + 1  # +1 for newline
                        
                        full_text += page_text + "\n"
                        
                except Exception as page_error:
                    print(f"⚠️  Error processing page {page_num + 1}: {page_error}")
                    continue
            
            print(f"✅ Extracted {len(full_text):,} characters from {doc.page_count} pages with position tracking")
            return full_text.strip(), page_map
            
        except Exception as e:
            print(f"❌ Error reading PDF: {e}")
            return self._try_alternative_extraction(pdf_path)
            
        finally:
            if doc and not doc.is_closed:
                try:
                    doc.close()
                except:
                    pass

    def _try_alternative_extraction(self, pdf_path: str) -> Tuple[str, Dict]:
        """Alternative extraction methods with basic page tracking"""
        try:
            print("🔄 Trying alternative extraction...")
            with fitz.open(pdf_path) as doc:
                full_text = ""
                page_map = {}
                current_char_pos = 0
                
                for page_num in range(len(doc)):
                    page = doc[page_num]
                    text = page.get_text("text")
                    
                    if text.strip():
                        # Basic line tracking for alternative method
                        lines = text.split('\n')
                        for line_num, line in enumerate(lines, 1):
                            if line.strip():
                                page_map[current_char_pos] = {
                                    'page': page_num + 1,
                                    'line': line_num,
                                    'page_line_key': f"Page {page_num + 1}, Line {line_num}",
                                    'text_preview': line[:50] + "..." if len(line) > 50 else line
                                }
                            current_char_pos += len(line) + 1
                        
                        full_text += text + "\n"
                
                if full_text.strip():
                    print(f"✅ Alternative method: {len(full_text):,} characters with tracking")
                    return full_text.strip(), page_map
        except Exception as e:
            print(f"❌ Alternative extraction failed: {e}")
        return "", {}

    def find_page_line_for_position(self, char_pos: int, page_map: Dict) -> str:
        """Find the best page/line reference for a character position"""
        if not page_map:
            return "Unknown Location"
        
        # Find the closest preceding position in our map
        best_pos = 0
        for pos in page_map:
            if pos <= char_pos and pos > best_pos:
                best_pos = pos
        
        if best_pos in page_map:
            return page_map[best_pos]['page_line_key']
        
        # Fallback: use the first available mapping
        first_key = min(page_map.keys())
        return page_map[first_key]['page_line_key']

    def create_contextual_chunks_with_tracking(self, text: str, page_map: Dict, analysis: Dict, target_sections: int = 25) -> List[Dict]:
        """Create context-aware chunks for keypoint extraction"""
        total_chars = len(text)
        
        # Use target_sections parameter for precise control
        optimal_chunks = target_sections
        base_chunk_size = total_chars // optimal_chunks if optimal_chunks > 0 else total_chars
        
        print(f"🧩 Creating {optimal_chunks} contextual chunks for keypoint extraction (~{base_chunk_size:,} chars each)")
        
        # Smart splitting strategies with section awareness
        section_patterns = [
            r'\n(?=(?:Abstract|Introduction|Methodology|Methods|Results|Discussion|Conclusion|References|Bibliography|Appendix|Chapter \d+|Section \d+|\d+\.|\w+\.\s*\w+))',
            r'\n(?=\d+\.\s+[A-Z])',  # Numbered sections
            r'\n(?=[A-Z][^.]*\n)',   # Titles
            r'\n\s*\n(?=[A-Z])',     # Paragraph breaks before capitals
        ]
        
        # Try different splitting strategies
        best_splits = None
        for pattern in section_patterns:
            try:
                splits = re.split(pattern, text, flags=re.IGNORECASE)
                if len(splits) >= optimal_chunks // 2:
                    best_splits = splits
                    break
            except:
                continue
        
        # Fallback to paragraph splitting
        if not best_splits:
            best_splits = re.split(r'\n\s*\n', text)
        
        # Group paragraphs into appropriately sized chunks
        chunks = []
        current_chunk = ""
        current_paras = []
        chunk_id = 1
        current_start_pos = 0
        
        for para in best_splits:
            para = para.strip()
            if not para:
                continue
            
            para_start_pos = text.find(para, current_start_pos)
            if para_start_pos == -1:
                para_start_pos = current_start_pos
            
            # Check if adding this paragraph exceeds optimal chunk size
            if (len(current_chunk) + len(para) > base_chunk_size * 1.3 and 
                current_chunk and len(current_chunk) > base_chunk_size * 0.7):
                
                # Create chunk for keypoint extraction
                chunk_start_pos = text.find(current_chunk.split('\n\n')[0]) if current_chunk else 0
                chunk_theme = self._identify_chunk_theme(current_paras)
                
                # Get page/line information for this chunk
                chunk_location = self.find_page_line_for_position(chunk_start_pos, page_map)
                
                chunks.append({
                    "id": chunk_id,
                    "text": current_chunk,
                    "length": len(current_chunk),
                    "paragraph_count": len(current_paras),
                    "theme": chunk_theme,
                    "context_type": self._classify_content_type(current_chunk),
                    "page_line_location": chunk_location,
                    "start_position": chunk_start_pos
                })
                
                current_chunk = para
                current_paras = [para]
                current_start_pos = para_start_pos
                chunk_id += 1
            else:
                if current_chunk:
                    current_chunk += "\n\n" + para
                else:
                    current_chunk = para
                    current_start_pos = para_start_pos
                current_paras.append(para)
        
        # Add final chunk
        if current_chunk.strip():
            chunk_start_pos = text.find(current_chunk.split('\n\n')[0]) if current_chunk else 0
            chunk_theme = self._identify_chunk_theme(current_paras)
            chunk_location = self.find_page_line_for_position(chunk_start_pos, page_map)
            
            chunks.append({
                "id": chunk_id,
                "text": current_chunk,
                "length": len(current_chunk),
                "paragraph_count": len(current_paras),
                "theme": chunk_theme,
                "context_type": self._classify_content_type(current_chunk),
                "page_line_location": chunk_location,
                "start_position": chunk_start_pos
            })
        
        # Sort chunks by their position in the document
        chunks.sort(key=lambda x: x['start_position'])
        
        # Reassign sequential IDs after sorting
        for i, chunk in enumerate(chunks, 1):
            chunk['id'] = i
        
        print(f"📋 Created {len(chunks)} contextual chunks for keypoint extraction")
        for chunk in chunks:
            print(f"   Chunk {chunk['id']}: {chunk['theme']} at {chunk['page_line_location']} ({chunk['length']:,} chars)")
        
        return chunks

    def _identify_chunk_theme(self, paragraphs: List[str]) -> str:
        """Identify the main theme/topic of a chunk"""
        combined_text = " ".join(paragraphs).lower()
        
        themes = {
            'Abstract/Summary': ['abstract', 'summary', 'overview', 'this paper', 'we present'],
            'Introduction': ['introduction', 'background', 'motivation', 'problem', 'challenge'],
            'Methodology': ['method', 'approach', 'algorithm', 'technique', 'procedure', 'implementation'],
            'Results/Analysis': ['results', 'analysis', 'findings', 'evaluation', 'performance', 'experiment'],
            'Discussion': ['discussion', 'implications', 'significance', 'interpretation', 'limitations'],
            'Conclusion': ['conclusion', 'summary', 'future work', 'contribution', 'in conclusion'],
            'Technical Details': ['architecture', 'system', 'framework', 'model', 'design', 'structure'],
            'Literature Review': ['previous', 'prior work', 'related work', 'literature', 'studies show'],
            'Figures/Tables': ['figure', 'table', 'graph', 'chart', 'diagram', 'illustration']
        }
        
        best_theme = 'General Content'
        max_score = 0
        
        for theme, keywords in themes.items():
            score = sum(1 for keyword in keywords if keyword in combined_text)
            if score > max_score:
                max_score = score
                best_theme = theme
        
        return best_theme

    def _classify_content_type(self, text: str) -> str:
        """Classify the type of content for appropriate keypoint extraction"""
        text_lower = text.lower()
        
        if any(word in text_lower for word in ['figure', 'table', 'graph', 'chart']):
            return 'visual_content'
        elif any(word in text_lower for word in ['algorithm', 'procedure', 'step', 'method']):
            return 'procedural'
        elif any(word in text_lower for word in ['result', 'finding', 'data', 'analysis']):
            return 'analytical'
        elif any(word in text_lower for word in ['definition', 'concept', 'theory']):
            return 'conceptual'
        else:
            return 'narrative'

    def extract_keypoints_from_chunk(self, chunk: Dict) -> Tuple[int, str, str, str]:
        """Extract keypoints, numerical values, and takeaways from chunk - NOT summary"""
        theme = chunk['theme']
        content_type = chunk['context_type']
        location = chunk['page_line_location']
        
        # Customize prompts for keypoint extraction based on content type
        if content_type == 'analytical':
            extraction_focus = "Extract key findings, all numerical values, percentages, statistics, data points, and quantitative results. Include specific numbers, measurements, and analytical conclusions."
        elif content_type == 'procedural':
            extraction_focus = "Extract key steps, procedures, algorithms, technical specifications, and methodological details. Include specific technical parameters and process steps."
        elif content_type == 'conceptual':
            extraction_focus = "Extract key concepts, definitions, theories, important terms, and conceptual frameworks. Include specific terminology and theoretical foundations."
        elif content_type == 'visual_content':
            extraction_focus = "Extract data from figures, tables, graphs, and charts. Include specific values, trends, comparisons, and visual evidence."
        else:
            extraction_focus = "Extract key points, important facts, main arguments, and significant details. Include specific information and critical takeaways."
        
        prompt = f"""Extract KEYPOINTS and IMPORTANT DATA from this {theme.lower()} section. DO NOT SUMMARIZE - extract actual content.

{extraction_focus}

Location: {location}

Content to extract from:
{chunk['text'][:12000]}

EXTRACT (not summarize):
- KEY POINTS: Important facts and statements
- NUMERICAL VALUES: All numbers, percentages, statistics, measurements
- KEY TAKEAWAYS: Critical insights and conclusions
- TECHNICAL DETAILS: Specific technical information
- IMPORTANT FACTS: Significant information points

Format as structured keypoint extraction, preserving original wording where important.

KEYPOINT EXTRACTION:"""
        
        try:
            start_time = time.time()
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.1,  # Low temperature for accurate extraction
                        "top_p": 0.95,
                        "num_predict": 1500,  # Generous space for keypoints
                    }
                },
                timeout=120
            )
            
            processing_time = time.time() - start_time
            response.raise_for_status()
            keypoints = response.json().get('response', 'ERROR: No response').strip()
            
            print(f"Chunk {chunk['id']} ({theme}) at {location}: {len(keypoints):,} chars extracted in {processing_time:.1f}s")
            return chunk['id'], keypoints, theme, location
            
        except Exception as e:
            return chunk['id'], f"ERROR extracting keypoints from {theme}: {str(e)[:200]}", theme, location

    def generate_complete_pdf_summary(self, text: str, analysis: Dict) -> str:
        """Generate complete PDF summary - stored in local variable only"""
        target_length = analysis['estimated_pdf_summary_length']
        
        prompt = f"""Create an EXTREMELY CONCISE and INTELLIGENT complete document summary. Target: {target_length} characters.

REQUIREMENTS:
- MAXIMUM {target_length} characters (strict limit)
- Focus on KEY TAKEAWAYS only
- Include CRITICAL numerical values and findings
- Highlight MAIN CONCLUSIONS and IMPLICATIONS
- Preserve ESSENTIAL technical details
- Skip redundant information
- Use precise, dense language

Document content (first part):
{text[:20000]}

Create ultra-compressed intelligent summary covering entire document:

COMPLETE DOCUMENT SUMMARY:"""
        
        try:
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.2,
                        "top_p": 0.9,
                        "num_predict": target_length // 2
                    }
                },
                timeout=180
            )
            
            response.raise_for_status()
            summary = response.json().get('response', 'ERROR generating complete summary').strip()
            
            # Enforce strict length limit
            if len(summary) > target_length:
                summary = summary[:target_length-3] + "..."
            
            print(f"✅ Complete PDF summary generated: {len(summary):,} chars (target: {target_length:,})")
            return summary
            
        except Exception as e:
            return f"ERROR creating complete PDF summary: {e}"

    def save_enhanced_json(self, chunk_keypoints: List[Dict], analysis: Dict, stats: Dict, 
                          original_pdf_name: str, output_path: str):
        """Save JSON with keypoints (not summaries) and location tracking"""
        try:
            # Prepare contextual keypoints with proper ordering and location info
            contextual_keypoints = []
            
            # Sort by chunk ID to maintain document order
            ordered_keypoints = sorted([k for k in chunk_keypoints if not k['keypoints'].startswith('ERROR')], 
                                     key=lambda x: x['id'])
            
            for i, keypoint_data in enumerate(ordered_keypoints, 1):
                contextual_keypoints.append({
                    "section_number": i,
                    "title": keypoint_data['location'],  # Title is now page/line location
                    "location": keypoint_data['location'],
                    "theme": keypoint_data['theme'],
                    "keypoints": keypoint_data['keypoints'],  # Actual keypoints, not summary
                    "original_chunk_id": keypoint_data['id'],
                    "keypoints_length": len(keypoint_data['keypoints'])
                })
            
            # Create JSON structure focused on keypoints for RAG
            json_data = {
                "metadata": {
                    "source_pdf": original_pdf_name,
                    "model_used": self.model_name,
                    "processing_date": __import__('datetime').datetime.now().isoformat(),
                    "total_sections": len(contextual_keypoints),
                    "processing_version": "Keypoint Extraction v1.0",
                    "data_type": "keypoints_and_takeaways"
                },
                "document_analysis": {
                    **analysis,
                    "sections_created": len(contextual_keypoints),
                    "extraction_strategy": "Keypoint and numerical value extraction"
                },
                "processing_statistics": {
                    **stats,
                    "sections_successfully_processed": len(contextual_keypoints),
                    "total_keypoints_length": sum(len(k['keypoints']) for k in contextual_keypoints),
                    "average_keypoint_section_length": sum(len(k['keypoints']) for k in contextual_keypoints) // len(contextual_keypoints) if contextual_keypoints else 0
                },
                "contextual_keypoints": contextual_keypoints,  # Changed from summaries to keypoints
                "rag_ready": {
                    "total_sections": len(contextual_keypoints),
                    "sections_by_theme": {theme: len([k for k in contextual_keypoints if k['theme'] == theme]) 
                                        for theme in set(k['theme'] for k in contextual_keypoints)},
                    "data_structure": "Each section contains keypoints, numerical values, and takeaways for RAG processing"
                }
            }
            
            # Save JSON with proper formatting
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(json_data, f, indent=2, ensure_ascii=False)
            
            print(f"✅ Keypoint extraction JSON saved: {output_path}")
            print(f"   • {len(contextual_keypoints)} sections with keypoints and numerical data")
            print(f"   • Ready for RAG integration and Q&A systems")
            
        except Exception as e:
            print(f"❌ Error creating JSON: {e}")

    def process_document_for_rag(self, file_path: str, target_sections: int = 25, 
                                generate_pdf_summary: bool = False) -> Tuple[List[Dict], Dict, Dict]:
        """Main processing pipeline for RAG-ready keypoint extraction
        Now supports: PDF, DOCX, TXT, MD files
        """
        start_time = time.time()
        
        print("="*60)
        print(f"RAG-READY KEYPOINT EXTRACTION PROCESSOR ({self.get_file_extension(file_path)} FILE)")
        print("="*60)
        
        # Step 1: Extract text with position tracking
        print(f"\n📄 Extracting text from {os.path.basename(file_path)} with location tracking...")
        text, page_map = self.extract_text_from_file(file_path)
        if not text:
            return [], {}, {}
        
        # Step 2: Analyze document
        print("\n🧠 Analyzing document for keypoint extraction...")
        analysis = self.analyze_document_structure(text)
        
        # Step 3: Create chunks for keypoint extraction
        print(f"\n🧩 Creating {target_sections} contextual chunks for keypoint extraction...")
        chunks = self.create_contextual_chunks_with_tracking(text, page_map, analysis, target_sections)
        
        # Step 4: Extract keypoints (not summaries) in parallel
        print(f"\n🔍 Extracting keypoints from {len(chunks)} chunks with Llama3...")
        chunk_keypoints = []
        successful_keypoints = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_chunk = {executor.submit(self.extract_keypoints_from_chunk, chunk): chunk 
                             for chunk in chunks}
            
            for future in tqdm(as_completed(future_to_chunk), total=len(chunks), 
                             desc="Keypoint extraction"):
                try:
                    chunk_id, keypoints, theme, location = future.result()
                    chunk_keypoints.append({
                        'id': chunk_id,
                        'keypoints': keypoints,
                        'theme': theme,
                        'location': location
                    })
                    
                    if not keypoints.startswith('ERROR'):
                        successful_keypoints.append({
                            'keypoints': keypoints, 
                            'theme': theme,
                            'location': location
                        })
                        
                except Exception as e:
                    print(f"⚠️ Keypoint extraction failed: {e}")
        
        # Sort by chunk ID to maintain document order
        chunk_keypoints.sort(key=lambda x: x['id'])
        
        # Step 5: Generate optional complete PDF summary (stored in local variable)
        if generate_pdf_summary:
            print("\n📋 Generating optional complete document summary...")
            self.complete_pdf_summary = self.generate_complete_pdf_summary(text, analysis)
            print(f"✅ Complete document summary stored in memory: {len(self.complete_pdf_summary):,} chars")
        
        # Calculate final statistics
        total_time = time.time() - start_time
        total_keypoints_length = sum(len(k['keypoints']) for k in chunk_keypoints 
                                   if not k['keypoints'].startswith('ERROR'))
        
        stats = {
            'original_length': len(text),
            'keypoints_length': total_keypoints_length,
            'keypoint_extraction_ratio': total_keypoints_length / len(text) if len(text) > 0 else 0,
            'total_time': total_time,
            'successful_chunks': len(successful_keypoints),
            'total_chunks': len(chunks),
            'pdf_summary_generated': generate_pdf_summary,
            'pdf_summary_length': len(self.complete_pdf_summary) if self.complete_pdf_summary else 0
        }
        
        return chunk_keypoints, analysis, stats

    def get_complete_pdf_summary(self) -> str:
        """Get the complete PDF summary from local variable"""
        return self.complete_pdf_summary if self.complete_pdf_summary else "No PDF summary generated"

    def clear_pdf_summary(self):
        """Clear the stored PDF summary"""
        self.complete_pdf_summary = None


def main():
    """Main function for RAG-ready keypoint extraction"""
    # --- START of new condition block ---
    
    # Define the directory to check and the target extensions
    target_dir = "./database/"  # Checks the current directory where the script is run
    target_extensions = ('.pdf', '.docx', '.txt')
    
    # Find all files with the target extensions
    processable_files = []
    for filename in os.listdir(target_dir):
        if filename.lower().endswith(target_extensions):
            processable_files.append(filename)
            
    # Check if the number of processable files is exactly one
    if len(processable_files) != 1:
        #idhar a
        multidocs.multi(f"./database/{processable_files[0]}", f"./database/{processable_files[1]}")
        return # Exit the function

    # --- END of new condition block ---

    # Configuration is now set based on the single file found
    FILE_PATH = f"./database/{processable_files[0]}"
    MODEL_NAME = "llama3"
    TARGET_SECTIONS = 25
    GENERATE_PDF_SUMMARY = True  # Optional feature
    
    print(f"RAG-Ready Configuration:")
    print(f"• File Found: {FILE_PATH}")
    print(f"• Model: {MODEL_NAME}")
    print(f"• Target sections: {TARGET_SECTIONS}")
    print(f"• Generate summary: {GENERATE_PDF_SUMMARY}")
    
    # Test Ollama with llama3
    try:
        test_response = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": MODEL_NAME, "prompt": "Test", "stream": False},
            timeout=10
        )
        if test_response.status_code != 200:
            print(f"❌ Ollama connection failed: {test_response.status_code}")
            return
        print("✅ Ollama connection verified with Llama3")
    except Exception as e:
        print(f"❌ Ollama not accessible: {e}")
        return
    
    # Process document for RAG
    processor = EnhancedIntelligentDocumentProcessor(model_name=MODEL_NAME)
    chunk_keypoints, analysis, stats = processor.process_document_for_rag(
        FILE_PATH, TARGET_SECTIONS, GENERATE_PDF_SUMMARY
    )
    
    if chunk_keypoints and stats:
        # Generate output filenames
        base_name = os.path.splitext(os.path.basename(FILE_PATH))[0]
        timestamp = __import__('datetime').datetime.now().strftime('%Y%m%d_%H%M%S')
        
        json_path = f"{base_name}_rag_keypoints_{timestamp}.json"
        
        # Save keypoint extraction JSON for RAG
        processor.save_enhanced_json(chunk_keypoints, analysis, stats, FILE_PATH, json_path)
        
        # Results summary
        print("\n" + "="*60)
        print("RAG-READY KEYPOINT EXTRACTION COMPLETE")
        print("="*60)
        print(f"📊 Original: {stats['original_length']:,} chars")
        print(f"🔍 Keypoints: {stats['keypoints_length']:,} chars")
        if stats['original_length'] > 0:
             print(f"📈 Extraction ratio: {stats['keypoint_extraction_ratio']:.1%}")
        print(f"⏱️  Processing time: {stats['total_time']:.1f}s")
        print(f"✅ Success rate: {stats['successful_chunks']}/{stats['total_chunks']}")
        print(f"📁 JSON Output: {json_path}")
        
        # Summary info
        if GENERATE_PDF_SUMMARY:
            summary = processor.get_complete_pdf_summary()
            print(f"📋 Complete document summary: {len(summary):,} chars (stored in memory)")
            if stats['original_length'] > 0:
                print(f"🗜️  Summary compression: {stats['pdf_summary_length'] / stats['original_length']:.1%}")
            print("   Use processor.get_complete_pdf_summary() to access")
        
        # RAG readiness info
        successful_sections = stats['successful_chunks']
        print(f"🤖 RAG Integration Ready:")
        print(f"   • {successful_sections} sections with keypoints and numerical data")
        print(f"   • Location tracking for precise retrieval")
        print(f"   • Structured format for embedding systems")
        
        # Theme distribution
        if chunk_keypoints:
            themes = {}
            for chunk in chunk_keypoints:
                if not chunk['keypoints'].startswith('ERROR'):
                    theme = chunk['theme']
                    themes[theme] = themes.get(theme, 0) + 1
            
            print(f"📊 Content distribution:")
            for theme, count in sorted(themes.items()):
                print(f"   • {theme}: {count} section{'s' if count > 1 else ''}")
        
        # Usage examples
        print(f"\n💡 Usage Examples:")
        print(f"   • Load JSON for RAG: json.load(open('{json_path}'))")
        print(f"   • Access summary: processor.get_complete_pdf_summary()")
        print(f"   • Clear memory: processor.clear_pdf_summary()")
            
    else:
        print("❌ Processing failed. Check error messages above.")


if __name__ == "__main__":
    main()