# File: enhanced_intelligent_document_processor.py
import json
import re
import fitz
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

class EnhancedIntelligentDocumentProcessor:
    def __init__(self, model_name: str = "llama3"):
        """
        Initialize processor with intelligent compression and page tracking
        """
        self.model_name = model_name
        self.min_compression = 0.25  # 25% minimum
        self.max_compression = 0.40  # 40% maximum
        self.max_workers = 4
        
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
        
        # Calculate optimal compression based on content
        if complexity_score > 0.7:  # High complexity (academic/technical)
            target_compression = 0.35  # 35% - preserve more detail
        elif complexity_score > 0.4:  # Medium complexity
            target_compression = 0.30  # 30% - balanced
        else:  # Lower complexity
            target_compression = 0.25  # 25% - more aggressive
        
        # Adjust based on document length
        if total_chars > 100000:  # Very long documents
            target_compression = min(target_compression + 0.05, self.max_compression)
        
        analysis = {
            'total_chars': total_chars,
            'total_words': total_words,
            'paragraphs': len(paragraphs),
            'avg_paragraph_length': avg_para_length,
            'complexity_score': complexity_score,
            'academic_score': academic_score,
            'technical_score': technical_score,
            'target_compression': target_compression,
            'estimated_summary_length': int(total_chars * target_compression)
        }
        
        print(f"📊 Document Analysis:")
        print(f"   • Length: {total_chars:,} chars, {total_words:,} words")
        print(f"   • Complexity: {complexity_score:.2f} (Academic: {academic_score}, Technical: {technical_score})")
        print(f"   • Target compression: {target_compression:.1%}")
        print(f"   • Expected summary: ~{analysis['estimated_summary_length']:,} chars")
        
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
                        page_start_pos = current_char_pos
                        
                        for line_num, line in enumerate(lines, 1):
                            line_start_pos = current_char_pos
                            line_end_pos = current_char_pos + len(line)
                            
                            # Store mapping for this line
                            page_map[line_start_pos] = {
                                'page': page_num + 1,
                                'line': line_num,
                                'page_line_key': f"Page {page_num + 1}, Line {line_num}",
                                'text_preview': line[:50] + "..." if len(line) > 50 else line
                            }
                            
                            current_char_pos = line_end_pos + 1  # +1 for newline
                        
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

    def create_contextual_chunks_with_tracking(self, text: str, page_map: Dict, analysis: Dict) -> List[Dict]:
        """Create context-aware chunks with proper page/line tracking"""
        target_compression = analysis['target_compression']
        total_chars = len(text)
        
        # Calculate optimal number of chunks based on content and target compression
        target_summary_chars = int(total_chars * target_compression)
        
        # Determine appropriate number of chunks based on document length and complexity
        if total_chars < 20000:  # Small documents
            optimal_chunks = max(3, min(6, target_summary_chars // 1000))
        elif total_chars < 50000:  # Medium documents
            optimal_chunks = max(6, min(12, target_summary_chars // 1200))
        elif total_chars < 100000:  # Large documents
            optimal_chunks = max(8, min(16, target_summary_chars // 1500))
        else:  # Very large documents
            optimal_chunks = max(12, min(20, target_summary_chars // 1800))
        
        base_chunk_size = total_chars // optimal_chunks
        
        print(f"🧩 Creating {optimal_chunks} contextual chunks (~{base_chunk_size:,} chars each)")
        
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
                
                # Create chunk with tracking information
                chunk_start_pos = text.find(current_chunk.split('\n\n')[0]) if current_chunk else 0
                chunk_theme = self._identify_chunk_theme(current_paras)
                target_summary_length = max(800, int(len(current_chunk) * target_compression * 1.2))
                
                # Get page/line information for this chunk
                chunk_location = self.find_page_line_for_position(chunk_start_pos, page_map)
                
                chunks.append({
                    "id": chunk_id,
                    "text": current_chunk,
                    "length": len(current_chunk),
                    "paragraph_count": len(current_paras),
                    "theme": chunk_theme,
                    "target_summary_length": target_summary_length,
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
            target_summary_length = max(800, int(len(current_chunk) * target_compression * 1.2))
            chunk_location = self.find_page_line_for_position(chunk_start_pos, page_map)
            
            chunks.append({
                "id": chunk_id,
                "text": current_chunk,
                "length": len(current_chunk),
                "paragraph_count": len(current_paras),
                "theme": chunk_theme,
                "target_summary_length": target_summary_length,
                "context_type": self._classify_content_type(current_chunk),
                "page_line_location": chunk_location,
                "start_position": chunk_start_pos
            })
        
        # Sort chunks by their position in the document
        chunks.sort(key=lambda x: x['start_position'])
        
        # Reassign sequential IDs after sorting
        for i, chunk in enumerate(chunks, 1):
            chunk['id'] = i
        
        print(f"📋 Created {len(chunks)} contextual chunks with location tracking")
        for chunk in chunks:
            print(f"   Chunk {chunk['id']}: {chunk['theme']} at {chunk['page_line_location']} ({chunk['length']:,} chars → ~{chunk['target_summary_length']} chars)")
        
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
        """Classify the type of content for appropriate summarization"""
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

    def summarize_chunk_contextually(self, chunk: Dict) -> Tuple[int, str, str, str]:
        """Create context-aware summaries preserving important information"""
        theme = chunk['theme']
        content_type = chunk['context_type']
        target_length = chunk['target_summary_length']
        location = chunk['page_line_location']
        
        # Customize prompts based on content type and theme
        if content_type == 'analytical':
            prompt_focus = "Focus on key findings, data points, results, and their significance. Preserve specific numbers, percentages, and quantitative results."
        elif content_type == 'procedural':
            prompt_focus = "Detail the methods, procedures, algorithms, and technical approaches. Maintain step-by-step processes and technical specifications."
        elif content_type == 'conceptual':
            prompt_focus = "Explain key concepts, definitions, theories, and their relationships. Preserve technical terminology and conceptual frameworks."
        elif content_type == 'visual_content':
            prompt_focus = "Describe figures, tables, graphs and their key insights. Include data trends, comparisons, and visual evidence."
        else:
            prompt_focus = "Capture main ideas, arguments, and supporting evidence. Maintain logical flow and key supporting details."
        
        prompt = f"""Create a comprehensive summary of this {theme.lower()} section. {prompt_focus}

Target length: {target_length} characters (approximately {target_length//5} words)
Location: {location}

Content to summarize:
{chunk['text'][:12000]}

Requirements:
- Maintain approximately {target_length} characters
- Preserve key information and context
- Keep technical terms and important details
- Use clear, structured language
- Focus on substantive content over style

CONTEXTUAL SUMMARY:"""
        
        try:
            start_time = time.time()
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.2,  # More deterministic for consistency
                        "top_p": 0.9,
                        "num_predict": target_length // 3,  # Allow sufficient space
                    }
                },
                timeout=120
            )
            
            processing_time = time.time() - start_time
            response.raise_for_status()
            summary = response.json().get('response', 'ERROR: No response').strip()
            
            # Ensure reasonable length (allow some flexibility)
            if len(summary) > target_length * 1.3:
                summary = summary[:int(target_length * 1.2)] + "..."
            
            print(f"Chunk {chunk['id']} ({theme}) at {location}: {len(summary):,} chars in {processing_time:.1f}s")
            return chunk['id'], summary, theme, location
            
        except Exception as e:
            return chunk['id'], f"ERROR processing {theme}: {str(e)[:200]}", theme, location

    def create_executive_summary(self, chunk_summaries: List[Dict], analysis: Dict) -> str:
        """Create intelligent executive summary"""
        # Group summaries by theme for better organization
        theme_groups = defaultdict(list)
        for summary in chunk_summaries:
            if not summary['summary'].startswith('ERROR'):
                theme = summary['theme']
                theme_groups[theme].append(summary['summary'])
        
        # Create organized summary text
        organized_content = []
        for theme, summaries in theme_groups.items():
            organized_content.append(f"{theme}:\n" + "\n".join(summaries))
        
        combined_summaries = "\n\n".join(organized_content)
        target_exec_length = max(2000, int(analysis['total_chars'] * 0.08))  # 8% for executive summary
        
        prompt = f"""Create a comprehensive executive summary from these section summaries. Target length: {target_exec_length} characters.

Section Summaries by Theme:
{combined_summaries[:15000]}

Create an executive summary with:
1. DOCUMENT OVERVIEW (purpose, scope, type)
2. KEY CONTRIBUTIONS (main findings, innovations, arguments)
3. METHODOLOGY/APPROACH (if applicable)
4. SIGNIFICANT RESULTS (important data, conclusions)
5. IMPLICATIONS (significance, applications, future directions)

Target: {target_exec_length} characters. Be comprehensive but concise.

EXECUTIVE SUMMARY:"""
        
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
                        "num_predict": target_exec_length // 3
                    }
                },
                timeout=150
            )
            
            response.raise_for_status()
            summary = response.json().get('response', 'ERROR generating executive summary').strip()
            
            if len(summary) > target_exec_length * 1.2:
                summary = summary[:int(target_exec_length * 1.1)] + "..."
            
            return summary
            
        except Exception as e:
            return f"ERROR creating executive summary: {e}"

    def save_enhanced_pdf(self, chunk_summaries: List[Dict], executive_summary: str, 
                         output_path: str, original_pdf_name: str, analysis: Dict, stats: Dict):
        """Save comprehensive PDF with proper section ordering"""
        try:
            doc = SimpleDocTemplate(output_path, pagesize=letter, 
                                  topMargin=0.8*inch, bottomMargin=0.8*inch,
                                  leftMargin=0.8*inch, rightMargin=0.8*inch)
            
            styles = getSampleStyleSheet()
            story = []
            
            # Professional styles
            title_style = ParagraphStyle(
                'Title', parent=styles['Title'],
                fontSize=18, spaceAfter=20, alignment=1, textColor='darkblue'
            )
            
            section_style = ParagraphStyle(
                'Section', parent=styles['Heading1'],
                fontSize=14, spaceAfter=12, textColor='darkblue', spaceBefore=16
            )
            
            subsection_style = ParagraphStyle(
                'Subsection', parent=styles['Heading2'],
                fontSize=12, spaceAfter=8, textColor='blue', spaceBefore=12
            )
            
            body_style = ParagraphStyle(
                'Body', parent=styles['Normal'],
                fontSize=10, spaceAfter=8, leading=12, alignment=4  # Justified
            )
            
            stats_style = ParagraphStyle(
                'Stats', parent=styles['Normal'],
                fontSize=9, spaceAfter=6, textColor='gray'
            )
            
            location_style = ParagraphStyle(
                'Location', parent=styles['Normal'],
                fontSize=8, spaceAfter=4, textColor='darkgray', fontName='Helvetica-Oblique'
            )
            
            # Title page
            story.append(Paragraph(f"Enhanced Document Analysis", title_style))
            story.append(Paragraph(f"{os.path.basename(original_pdf_name)}", subsection_style))
            story.append(Spacer(1, 20))
            
            # Processing statistics
            compression_achieved = stats['compression_ratio']
            stats_text = f"""Processing Summary: {stats['total_time']:.1f}s processing time • 
                          {compression_achieved:.1%} compression ratio • 
                          {stats['successful_chunks']}/{stats['total_chunks']} sections processed • 
                          Document complexity: {analysis['complexity_score']:.2f}"""
            story.append(Paragraph(stats_text, stats_style))
            story.append(Spacer(1, 20))
            
            # Executive Summary
            story.append(Paragraph("EXECUTIVE SUMMARY", section_style))
            story.append(Paragraph(executive_summary, body_style))
            story.append(PageBreak())
            
            # Section-wise summaries in proper order
            story.append(Paragraph("DETAILED SECTION ANALYSIS", section_style))
            story.append(Paragraph("Sections are presented in document order with location references.", stats_style))
            story.append(Spacer(1, 12))
            
            # Sort summaries by chunk ID (which corresponds to document order)
            ordered_summaries = sorted([s for s in chunk_summaries if not s['summary'].startswith('ERROR')], 
                                     key=lambda x: x['id'])
            
            for i, summary_data in enumerate(ordered_summaries, 1):
                # Section header with proper numbering
                section_title = f"Section {i}: {summary_data['theme']}"
                story.append(Paragraph(section_title, subsection_style))
                
                # Location information
                location_text = f"Location: {summary_data['location']}"
                story.append(Paragraph(location_text, location_style))
                story.append(Spacer(1, 4))
                
                # Summary content
                story.append(Paragraph(summary_data['summary'], body_style))
                story.append(Spacer(1, 12))
            
            # Document analysis appendix
            story.append(PageBreak())
            story.append(Paragraph("DOCUMENT ANALYSIS", section_style))
            
            analysis_text = f"""
            <b>Document Characteristics:</b><br/>
            • Total length: {analysis['total_chars']:,} characters ({analysis['total_words']:,} words)<br/>
            • Paragraphs: {analysis['paragraphs']}<br/>
            • Average paragraph length: {analysis['avg_paragraph_length']:.0f} characters<br/>
            • Content complexity score: {analysis['complexity_score']:.2f}/1.0<br/>
            • Academic indicators: {analysis['academic_score']} • Technical indicators: {analysis['technical_score']}<br/>
            <br/>
            <b>Compression Analysis:</b><br/>
            • Target compression ratio: {analysis['target_compression']:.1%}<br/>
            • Achieved compression ratio: {compression_achieved:.1%}<br/>
            • Original document: {stats['original_length']:,} characters<br/>
            • Summary length: {stats['summary_length']:,} characters<br/>
            • Information preservation: {"Excellent" if compression_achieved >= 0.25 else "Good" if compression_achieved >= 0.15 else "Aggressive"}
            """
            
            story.append(Paragraph(analysis_text, body_style))
            
            doc.build(story)
            print(f"✅ Enhanced PDF summary saved: {output_path}")
            
        except Exception as e:
            print(f"❌ Error creating PDF: {e}")

    def save_enhanced_json(self, chunk_summaries: List[Dict], executive_summary: str, 
                          analysis: Dict, stats: Dict, original_pdf_name: str, output_path: str):
        """Save enhanced JSON with proper structure and location tracking"""
        try:
            # Prepare contextual summaries with proper ordering and location info
            contextual_summaries = []
            
            # Sort by chunk ID to maintain document order
            ordered_summaries = sorted([s for s in chunk_summaries if not s['summary'].startswith('ERROR')], 
                                     key=lambda x: x['id'])
            
            for i, summary_data in enumerate(ordered_summaries, 1):
                contextual_summaries.append({
                    "section_number": i,
                    "title": f"Section {i}: {summary_data['theme']}",
                    "location": summary_data['location'],
                    "theme": summary_data['theme'],
                    "summary": summary_data['summary'],
                    "original_chunk_id": summary_data['id'],
                    "summary_length": len(summary_data['summary'])
                })
            
            # Create comprehensive JSON structure
            json_data = {
                "metadata": {
                    "source_pdf": original_pdf_name,
                    "model_used": self.model_name,
                    "processing_date": __import__('datetime').datetime.now().isoformat(),
                    "total_sections": len(contextual_summaries),
                    "processing_version": "Enhanced v2.0"
                },
                "document_analysis": {
                    **analysis,
                    "optimal_sections_created": len(contextual_summaries),
                    "section_creation_strategy": "Context-aware with page tracking"
                },
                "processing_statistics": {
                    **stats,
                    "sections_successfully_processed": len(contextual_summaries),
                    "average_section_compression": sum(len(s['summary']) for s in contextual_summaries) / sum(stats['original_length'] / len(contextual_summaries) for _ in contextual_summaries) if contextual_summaries else 0
                },
                "executive_summary": {
                    "content": executive_summary,
                    "length": len(executive_summary),
                    "compression_ratio": len(executive_summary) / analysis['total_chars']
                },
                "contextual_summaries": contextual_summaries,
                "summary_statistics": {
                    "total_sections": len(contextual_summaries),
                    "total_summary_length": sum(len(s['summary']) for s in contextual_summaries),
                    "average_section_length": sum(len(s['summary']) for s in contextual_summaries) // len(contextual_summaries) if contextual_summaries else 0,
                    "sections_by_theme": {theme: len([s for s in contextual_summaries if s['theme'] == theme]) 
                                        for theme in set(s['theme'] for s in contextual_summaries)}
                }
            }
            
            # Save JSON with proper formatting
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(json_data, f, indent=2, ensure_ascii=False)
            
            print(f"✅ Enhanced JSON analysis saved: {output_path}")
            print(f"   • {len(contextual_summaries)} contextual summaries with location tracking")
            print(f"   • Sections properly ordered and numbered")
            
        except Exception as e:
            print(f"❌ Error creating JSON: {e}")

    def process_document_enhanced(self, pdf_path: str) -> Tuple[List[Dict], str, Dict, Dict]:
        """Enhanced processing pipeline with location tracking"""
        start_time = time.time()
        
        print("="*60)
        print("ENHANCED INTELLIGENT DOCUMENT PROCESSOR")
        print("="*60)
        
        # Step 1: Extract text with page tracking
        print("\n📄 Extracting text with location tracking...")
        text, page_map = self.extract_text_with_page_tracking(pdf_path)
        if not text:
            return [], "", {}, {}
        
        # Step 2: Analyze document
        print("\n🧠 Analyzing document structure and complexity...")
        analysis = self.analyze_document_structure(text)
        
        # Step 3: Create intelligent chunks with tracking
        print("\n🧩 Creating contextual chunks with location tracking...")
        chunks = self.create_contextual_chunks_with_tracking(text, page_map, analysis)
        
        # Step 4: Process chunks in parallel with location info
        print(f"\n🤖 Processing {len(chunks)} chunks contextually with Llama3...")
        chunk_summaries = []
        successful_summaries = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_chunk = {executor.submit(self.summarize_chunk_contextually, chunk): chunk 
                             for chunk in chunks}
            
            for future in tqdm(as_completed(future_to_chunk), total=len(chunks), 
                             desc="Enhanced processing"):
                try:
                    chunk_id, summary, theme, location = future.result()
                    chunk_summaries.append({
                        'id': chunk_id,
                        'summary': summary,
                        'theme': theme,
                        'location': location
                    })
                    
                    if not summary.startswith('ERROR'):
                        successful_summaries.append({
                            'summary': summary, 
                            'theme': theme,
                            'location': location
                        })
                        
                except Exception as e:
                    print(f"⚠️ Task failed: {e}")
        
        # Sort by chunk ID to maintain document order
        chunk_summaries.sort(key=lambda x: x['id'])
        
        # Step 5: Create executive summary
        print("\n📋 Creating executive summary...")
        executive_summary = self.create_executive_summary(successful_summaries, analysis)
        
        # Calculate final statistics
        total_time = time.time() - start_time
        total_summary_length = sum(len(s['summary']) for s in chunk_summaries 
                                 if not s['summary'].startswith('ERROR')) + len(executive_summary)
        compression_achieved = total_summary_length / len(text) if len(text) > 0 else 0
        
        stats = {
            'original_length': len(text),
            'summary_length': total_summary_length,
            'compression_ratio': compression_achieved,
            'target_compression': analysis['target_compression'],
            'total_time': total_time,
            'successful_chunks': len(successful_summaries),
            'total_chunks': len(chunks)
        }
        
        return chunk_summaries, executive_summary, analysis, stats


def main():
    """Enhanced main function with improved processing and outputs"""
    # Configuration
    PDF_PATH = "./database/sample_document.pdf"
    MODEL_NAME = "llama3"  # Updated to llama3
    
    print(f"Enhanced Configuration:")
    print(f"• PDF: {PDF_PATH}")
    print(f"• Model: {MODEL_NAME}")
    print(f"• Intelligent compression: 25-40% (adaptive)")
    print(f"• Location tracking: Enabled")
    print(f"• Section ordering: Enhanced")
    
    # Validate setup
    if not os.path.exists(PDF_PATH):
        print(f"❌ PDF file '{PDF_PATH}' not found!")
        return
    
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
    
    # Process document with enhanced features
    processor = EnhancedIntelligentDocumentProcessor(model_name=MODEL_NAME)
    chunk_summaries, executive_summary, analysis, stats = processor.process_document_enhanced(PDF_PATH)
    
    if chunk_summaries and stats:
        # Generate output filenames
        base_name = os.path.splitext(os.path.basename(PDF_PATH))[0]
        timestamp = __import__('datetime').datetime.now().strftime('%Y%m%d_%H%M%S')
        
        pdf_path = f"./database/{base_name}_enhanced_summary_{timestamp}.pdf"
        json_path = f"./database/{base_name}_enhanced_analysis_{timestamp}.json"
        
        # Save enhanced PDF with proper section ordering
        processor.save_enhanced_pdf(chunk_summaries, executive_summary, pdf_path, 
                                   PDF_PATH, analysis, stats)
        
        # Save enhanced JSON with location tracking and proper structure
        processor.save_enhanced_json(chunk_summaries, executive_summary, analysis, 
                                   stats, PDF_PATH, json_path)
        
        # Results summary
        print("\n" + "="*60)
        print("ENHANCED PROCESSING COMPLETE")
        print("="*60)
        print(f"📊 Original: {stats['original_length']:,} chars")
        print(f"📄 Summary: {stats['summary_length']:,} chars")
        print(f"🧠 Target compression: {analysis['target_compression']:.1%}")
        print(f"🗜️  Achieved compression: {stats['compression_ratio']:.1%}")
        print(f"📈 Complexity score: {analysis['complexity_score']:.2f}")
        print(f"⏱️  Processing time: {stats['total_time']:.1f}s")
        print(f"✅ Success rate: {stats['successful_chunks']}/{stats['total_chunks']}")
        print(f"📁 PDF Output: {pdf_path}")
        print(f"📁 JSON Output: {json_path}")
        
        # Enhanced quality assessment
        successful_sections = stats['successful_chunks']
        if 0.25 <= stats['compression_ratio'] <= 0.40:
            print("🎯 Compression ratio optimal - excellent information preservation!")
        elif stats['compression_ratio'] < 0.25:
            print("⚠️  Compression higher than optimal - some detail may be lost")
        else:
            print("📝 Compression lower than target - very detailed summary")
        
        print(f"📚 Created {successful_sections} contextual summaries with location tracking")
        print(f"🔢 Sections properly ordered and numbered in outputs")
        
        # Theme distribution
        if chunk_summaries:
            themes = {}
            for chunk in chunk_summaries:
                if not chunk['summary'].startswith('ERROR'):
                    theme = chunk['theme']
                    themes[theme] = themes.get(theme, 0) + 1
            
            print(f"📊 Content distribution:")
            for theme, count in sorted(themes.items()):
                print(f"   • {theme}: {count} section{'s' if count > 1 else ''}")
            
    else:
        print("❌ Processing failed. Check error messages above.")


if __name__ == "__main__":
    main()