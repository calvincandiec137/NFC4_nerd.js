import json
import os
import fitz  # PyMuPDF
import re
from typing import List, Dict, Tuple, Optional
from datetime import datetime
import difflib

class EnhancedRAGPDFHighlighter:
    """
    Enhanced PDF highlighter that works with your RAG system's JSON output
    Focuses on robust text matching and easy integration
    """
    
    def __init__(self, pdf_path: str, session_json_path: str = None):
        self.pdf_path = pdf_path
        self.session_json_path = session_json_path
        self.pdf_doc = None
        self.session_data = None
        self.highlight_history = []
        self.load_resources()
    
    def load_resources(self):
        """Load PDF and session JSON data"""
        try:
            # Load PDF
            if os.path.exists(self.pdf_path):
                self.pdf_doc = fitz.open(self.pdf_path)
                print(f"✅ PDF loaded: {self.pdf_path} ({self.pdf_doc.page_count} pages)")
            else:
                print(f"❌ PDF not found: {self.pdf_path}")
                return False
            
            # Load session JSON if provided
            if self.session_json_path and os.path.exists(self.session_json_path):
                with open(self.session_json_path, 'r', encoding='utf-8') as f:
                    self.session_data = json.load(f)
                print(f"✅ Session JSON loaded: {self.session_json_path}")
                return True
            elif self.session_json_path:
                print(f"❌ Session JSON not found: {self.session_json_path}")
                return False
            
            return True
                
        except Exception as e:
            print(f"❌ Error loading resources: {e}")
            return False
    
    def load_session_from_file(self, session_json_path: str) -> bool:
        """Load session data from a specific file"""
        self.session_json_path = session_json_path
        try:
            with open(session_json_path, 'r', encoding='utf-8') as f:
                self.session_data = json.load(f)
            print(f"✅ Session JSON loaded: {session_json_path}")
            return True
        except Exception as e:
            print(f"❌ Error loading session JSON: {e}")
            return False
    
    def extract_page_and_line_info(self, location: str) -> Tuple[Optional[int], Optional[int]]:
        """Extract page and line information from location string"""
        page_match = re.search(r"Page\s+(\d+)", location, re.IGNORECASE)
        line_match = re.search(r"Line\s+(\d+)", location, re.IGNORECASE)
        
        page_num = int(page_match.group(1)) if page_match else None
        line_num = int(line_match.group(1)) if line_match else None
        
        return page_num, line_num
    
    def find_text_with_multiple_strategies(self, search_text: str, target_page: int = None) -> List[Dict]:
        """
        Find text using multiple search strategies for better accuracy
        """
        if not self.pdf_doc or not search_text.strip():
            return []
        
        matches = []
        search_text_clean = self.clean_text_for_search(search_text)
        
        # Determine pages to search
        pages_to_search = [target_page - 1] if target_page and target_page <= self.pdf_doc.page_count else range(self.pdf_doc.page_count)
        
        for page_num in pages_to_search:
            page = self.pdf_doc[page_num]
            page_matches = []
            
            # Strategy 1: Exact match
            exact_matches = page.search_for(search_text_clean)
            for match_rect in exact_matches:
                page_matches.append({
                    'page': page_num + 1,
                    'rect': match_rect,
                    'matched_text': search_text_clean,
                    'confidence': 1.0,
                    'method': 'exact_match'
                })
            
            # Strategy 2: Word-by-word search if no exact match
            if not exact_matches:
                word_matches = self.find_by_words(page, search_text_clean)
                page_matches.extend(word_matches)
            
            # Strategy 3: Fuzzy matching for partial matches
            if not page_matches:
                fuzzy_matches = self.find_fuzzy_matches(page, search_text_clean, page_num + 1)
                page_matches.extend(fuzzy_matches)
            
            matches.extend(page_matches)
        
        # Sort by confidence and return best matches
        matches.sort(key=lambda x: x['confidence'], reverse=True)
        return matches
    
    def find_by_words(self, page, search_text: str) -> List[Dict]:
        """Find text by searching for individual words and combining results"""
        words = search_text.split()
        if len(words) < 2:
            return []
        
        word_positions = {}
        for word in words:
            if len(word) > 2:  # Skip very short words
                positions = page.search_for(word)
                if positions:
                    word_positions[word] = positions
        
        # If we found most words, try to find regions where they cluster
        if len(word_positions) >= len(words) * 0.6:  # At least 60% of words found
            return self.cluster_word_matches(word_positions, page.number + 1, search_text)
        
        return []
    
    def cluster_word_matches(self, word_positions: Dict, page_num: int, original_text: str) -> List[Dict]:
        """Cluster nearby word matches into probable text regions"""
        matches = []
        
        # Simple clustering: find words that are close to each other
        all_rects = []
        for word, rects in word_positions.items():
            for rect in rects:
                all_rects.append((rect, word))
        
        if len(all_rects) >= 2:
            # Find the most likely cluster (simplified approach)
            # In a real implementation, you might use more sophisticated clustering
            
            # For now, just take the first few matches and create a bounding box
            sorted_rects = sorted(all_rects, key=lambda x: (x[0].y0, x[0].x0))
            
            if len(sorted_rects) >= 2:
                first_rect = sorted_rects[0][0]
                last_rect = sorted_rects[-1][0]
                
                # Create a bounding box around the matches
                combined_rect = fitz.Rect(
                    min(first_rect.x0, last_rect.x0),
                    min(first_rect.y0, last_rect.y0),
                    max(first_rect.x1, last_rect.x1),
                    max(first_rect.y1, last_rect.y1)
                )
                
                matches.append({
                    'page': page_num,
                    'rect': combined_rect,
                    'matched_text': original_text,
                    'confidence': 0.7,
                    'method': 'word_clustering'
                })
        
        return matches
    
    def find_fuzzy_matches(self, page, search_text: str, page_num: int, threshold: float = 0.6) -> List[Dict]:
        """Find fuzzy text matches using similarity comparison"""
        matches = []
        page_text = page.get_text()
        
        if not page_text.strip():
            return matches
        
        # Split into sentences and paragraphs for better matching
        text_blocks = self.extract_text_blocks(page_text)
        
        for block in text_blocks:
            similarity = self.calculate_similarity(search_text, block['text'])
            if similarity >= threshold:
                # Try to find approximate location of this text block
                block_words = block['text'].split()[:5]  # First 5 words
                search_phrase = ' '.join(block_words)
                
                locations = page.search_for(search_phrase)
                if locations:
                    matches.append({
                        'page': page_num,
                        'rect': locations[0],
                        'matched_text': block['text'][:100] + "..." if len(block['text']) > 100 else block['text'],
                        'confidence': similarity,
                        'method': 'fuzzy_match'
                    })
        
        return matches
    
    def extract_text_blocks(self, text: str) -> List[Dict]:
        """Extract meaningful text blocks from page text"""
        # Split by sentences first
        sentences = re.split(r'[.!?]+', text)
        blocks = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 20:  # Only meaningful sentences
                blocks.append({'text': sentence, 'type': 'sentence'})
        
        # Also try splitting by paragraphs
        paragraphs = text.split('\n\n')
        for para in paragraphs:
            para = ' '.join(para.split())  # Normalize whitespace
            if len(para) > 50:  # Only substantial paragraphs
                blocks.append({'text': para, 'type': 'paragraph'})
        
        return blocks
    
    def calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate text similarity using multiple methods"""
        # Method 1: Word overlap
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        jaccard_similarity = len(intersection) / len(union) if union else 0
        
        # Method 2: Character-based similarity using difflib
        char_similarity = difflib.SequenceMatcher(None, text1.lower(), text2.lower()).ratio()
        
        # Combine both methods
        combined_similarity = (jaccard_similarity * 0.7) + (char_similarity * 0.3)
        return combined_similarity
    
    def clean_text_for_search(self, text: str) -> str:
        """Clean and normalize text for better searching"""
        if not text:
            return ""
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Remove problematic characters but keep punctuation
        text = re.sub(r'[^\w\s\.\,\!\?\-\:\;\(\)]', ' ', text)
        
        # Remove multiple spaces
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def highlight_from_session_data(self, session_data: Dict = None, min_confidence: float = 0.5) -> Dict:
        """
        Main method to highlight PDF based on session data
        Works with your JSON structure containing chunks/contextual_keypoints
        """
        if session_data:
            self.session_data = session_data
        
        if not self.session_data:
            print("❌ No session data available")
            return {}
        
        highlight_results = {
            'timestamp': datetime.now().isoformat(),
            'query': self.session_data.get('query', 'Unknown query'),
            'rag_response': self.session_data.get('rag_response', ''),
            'highlights_created': [],
            'failed_highlights': [],
            'total_matches': 0
        }
        
        # Extract chunks from different possible structures
        chunks = []
        if 'chunks' in self.session_data:
            chunks = self.session_data['chunks']
        elif 'contextual_keypoints' in self.session_data:
            chunks = self.session_data['contextual_keypoints']
        elif isinstance(self.session_data, list):
            chunks = self.session_data
        
        print(f"\n🎯 Processing {len(chunks)} chunks for highlighting")
        
        for i, chunk in enumerate(chunks):
            # Handle different chunk structures
            chunk_text = chunk.get('keypoints', chunk.get('text', chunk.get('content', '')))
            location = chunk.get('location', chunk.get('title', f'Chunk {i+1}'))
            
            if not chunk_text or not chunk_text.strip():
                continue
            
            print(f"\n📍 Processing chunk {i+1}: {location}")
            print(f"   Text preview: {chunk_text[:100]}...")
            
            # Extract page information
            target_page, _ = self.extract_page_and_line_info(location)
            
            # Find text in PDF
            matches = self.find_text_with_multiple_strategies(chunk_text, target_page)
            
            # Filter by confidence
            good_matches = [m for m in matches if m['confidence'] >= min_confidence]
            
            if good_matches:
                best_match = good_matches[0]  # Take the best match
                
                highlight_info = {
                    'location': location,
                    'text_preview': chunk_text[:200] + "..." if len(chunk_text) > 200 else chunk_text,
                    'matches': len(good_matches),
                    'best_match': best_match,
                    'pages': [m['page'] for m in good_matches],
                    'confidence': best_match['confidence'],
                    'method': best_match['method']
                }
                
                highlight_results['highlights_created'].append(highlight_info)
                highlight_results['total_matches'] += len(good_matches)
                
                print(f"  ✅ Found {len(good_matches)} matches (best: {best_match['confidence']:.2f} confidence)")
            else:
                failed_info = {
                    'location': location,
                    'text_preview': chunk_text[:100] + "..." if len(chunk_text) > 100 else chunk_text,
                    'reason': f'No matches above confidence threshold ({min_confidence})'
                }
                highlight_results['failed_highlights'].append(failed_info)
                print(f"  ⚠️ No reliable matches found")
        
        # Store in history
        self.highlight_history.append(highlight_results)
        
        print(f"\n📊 Summary: {len(highlight_results['highlights_created'])} successful, {len(highlight_results['failed_highlights'])} failed")
        
        return highlight_results
    
    def create_highlighted_pdf(self, output_path: str, highlight_color: Tuple[float, float, float] = (1, 1, 0)) -> bool:
        """Create highlighted PDF with all matches from history"""
        if not self.pdf_doc or not self.highlight_history:
            print("❌ No PDF loaded or no highlights to apply")
            return False
        
        try:
            # Create a copy for highlighting
            highlighted_doc = fitz.open(self.pdf_path)
            total_highlights = 0
            
            # Apply highlights from all sessions
            for session_result in self.highlight_history:
                for highlight_info in session_result['highlights_created']:
                    best_match = highlight_info['best_match']
                    page = highlighted_doc[best_match['page'] - 1]
                    
                    # Create highlight annotation
                    highlight_annot = page.add_highlight_annot(best_match['rect'])
                    highlight_annot.set_colors(stroke=highlight_color)
                    
                    # Add metadata to the annotation
                    content = f"Location: {highlight_info['location']}\nConfidence: {best_match['confidence']:.2f}\nMethod: {best_match['method']}"
                    highlight_annot.set_info(content=content)
                    highlight_annot.update()
                    
                    total_highlights += 1
            
            # Save highlighted PDF
            highlighted_doc.save(output_path)
            highlighted_doc.close()
            
            print(f"✅ Highlighted PDF saved: {output_path}")
            print(f"📊 Total highlights applied: {total_highlights}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error creating highlighted PDF: {e}")
            return False
    
    def highlight_from_json_file(self, json_file_path: str, output_pdf_path: str = None, 
                                min_confidence: float = 0.5) -> bool:
        """
        Complete workflow: load JSON, highlight PDF, save result
        This is your main entry point for easy integration
        """
        print(f"🚀 Starting PDF highlighting workflow")
        print(f"   JSON file: {json_file_path}")
        print(f"   PDF file: {self.pdf_path}")
        
        # Load session data
        if not self.load_session_from_file(json_file_path):
            return False
        
        # Perform highlighting
        result = self.highlight_from_session_data(min_confidence=min_confidence)
        
        if not result or not result['highlights_created']:
            print("❌ No highlights were created")
            return False
        
        # Create output PDF if path provided
        if output_pdf_path:
            success = self.create_highlighted_pdf(output_pdf_path)
            if success:
                print(f"✅ Complete! Highlighted PDF saved to: {output_pdf_path}")
                return True
        
        return len(result['highlights_created']) > 0
    
    def get_highlight_summary(self) -> Dict:
        """Get comprehensive summary of highlighting results"""
        if not self.highlight_history:
            return {'message': 'No highlights created yet'}
        
        total_highlights = sum(len(h['highlights_created']) for h in self.highlight_history)
        total_failed = sum(len(h['failed_highlights']) for h in self.highlight_history)
        
        # Calculate average confidence
        all_confidences = []
        for session in self.highlight_history:
            for highlight in session['highlights_created']:
                all_confidences.append(highlight['confidence'])
        
        avg_confidence = sum(all_confidences) / len(all_confidences) if all_confidences else 0
        
        return {
            'total_sessions': len(self.highlight_history),
            'total_successful_highlights': total_highlights,
            'total_failed_highlights': total_failed,
            'success_rate': total_highlights / (total_highlights + total_failed) if (total_highlights + total_failed) > 0 else 0,
            'average_confidence': avg_confidence,
            'last_session': self.highlight_history[-1]['timestamp'] if self.highlight_history else None,
            'methods_used': self.get_methods_summary()
        }
    
    def get_methods_summary(self) -> Dict:
        """Get summary of highlighting methods used"""
        methods = {}
        for session in self.highlight_history:
            for highlight in session['highlights_created']:
                method = highlight['method']
                methods[method] = methods.get(method, 0) + 1
        return methods
    
    def close(self):
        """Clean up resources"""
        if self.pdf_doc:
            self.pdf_doc.close()


# Example usage and testing functions
def main():
    """Example usage of the enhanced highlighter"""
    
    # Configuration
    pdf_path = "./database/sample_document.pdf"
    json_path = "./datatbase/last_query_result.json"
    output_path = "./database/highlighted_output.pdf"
    
    # Initialize highlighter
    highlighter = EnhancedRAGPDFHighlighter(pdf_path)
    
    # Method 1: Complete workflow (recommended for your integration)
    success = highlighter.highlight_from_json_file(
        json_file_path=json_path,
        output_pdf_path=output_path,
        min_confidence=0.4  # Lower threshold for better coverage
    )
    
    if success:
        # Get summary
        summary = highlighter.get_highlight_summary()
        print(f"\n📈 Highlighting Summary:")
        print(f"   Success rate: {summary['success_rate']:.2%}")
        print(f"   Average confidence: {summary['average_confidence']:.2f}")
        print(f"   Methods used: {summary['methods_used']}")
    
    # Method 2: Step-by-step approach (for more control)
    # highlighter.load_session_from_file(json_path)
    # result = highlighter.highlight_from_session_data(min_confidence=0.4)
    # highlighter.create_highlighted_pdf(output_path)
    
    highlighter.close()


if __name__ == "__main__":
    main()
