import json
import os
import fitz  # PyMuPDF
import re
from typing import List, Dict, Tuple, Optional
import numpy as np
from datetime import datetime

class RAGPDFHighlighter:
    """
    Integrates RAG responses with PDF highlighting functionality
    Bridges the gap between your RAG system and PDF visualization
    """
    
    def __init__(self, pdf_path: str, json_data_path: str):
        self.pdf_path = pdf_path
        self.json_data_path = json_data_path
        self.pdf_doc = None
        self.json_data = None
        self.highlight_history = []
        self.load_resources()
    
    def load_resources(self):
        """Load PDF and JSON data"""
        try:
            # Load PDF
            if os.path.exists(self.pdf_path):
                self.pdf_doc = fitz.open(self.pdf_path)
                print(f"✅ PDF loaded: {self.pdf_path} ({self.pdf_doc.page_count} pages)")
            else:
                print(f"❌ PDF not found: {self.pdf_path}")
                return
            
            # Load JSON data
            if os.path.exists(self.json_data_path):
                with open(self.json_data_path, 'r', encoding='utf-8') as f:
                    self.json_data = json.load(f)
                print(f"✅ JSON data loaded: {len(self.json_data.get('contextual_keypoints', []))} sections")
            else:
                print(f"❌ JSON not found: {self.json_data_path}")
                return
                
        except Exception as e:
            print(f"❌ Error loading resources: {e}")
    
    def find_text_in_pdf(self, search_text: str, similarity_threshold: float = 0.7) -> List[Dict]:
        """
        Find text locations in PDF with fuzzy matching
        Returns list of matches with page numbers and coordinates
        """
        if not self.pdf_doc:
            return []
        
        matches = []
        search_text_clean = self.clean_text_for_search(search_text)
        
        for page_num in range(self.pdf_doc.page_count):
            page = self.pdf_doc[page_num]
            text_instances = page.search_for(search_text_clean)
            
            # If exact match found
            if text_instances:
                for inst in text_instances:
                    matches.append({
                        'page': page_num + 1,
                        'rect': inst,
                        'matched_text': search_text_clean,
                        'similarity': 1.0,
                        'method': 'exact_match'
                    })
            else:
                # Try fuzzy matching
                page_text = page.get_text()
                fuzzy_matches = self.fuzzy_search_in_text(search_text_clean, page_text, similarity_threshold)
                
                for match in fuzzy_matches:
                    # Find approximate coordinates for fuzzy matches
                    match_instances = page.search_for(match['matched_substring'])
                    if match_instances:
                        matches.append({
                            'page': page_num + 1,
                            'rect': match_instances[0],
                            'matched_text': match['matched_substring'],
                            'similarity': match['similarity'],
                            'method': 'fuzzy_match'
                        })
        
        return matches
    
    def clean_text_for_search(self, text: str) -> str:
        """Clean text for better search matching"""
        # Remove excessive whitespace and normalize
        text = ' '.join(text.split())
        # Remove common punctuation that might interfere
        text = re.sub(r'[^\w\s\.\,\!\?\-]', ' ', text)
        return text.strip()
    
    def fuzzy_search_in_text(self, search_text: str, full_text: str, threshold: float = 0.7) -> List[Dict]:
        """Perform fuzzy text matching"""
        search_words = search_text.split()
        if len(search_words) < 3:  # For short queries, be more strict
            return []
        
        # Create sliding window of text
        full_words = full_text.split()
        matches = []
        
        window_size = len(search_words) + 2  # Allow some flexibility
        
        for i in range(len(full_words) - window_size + 1):
            window_text = ' '.join(full_words[i:i + window_size])
            similarity = self.calculate_text_similarity(search_text, window_text)
            
            if similarity >= threshold:
                matches.append({
                    'matched_substring': window_text,
                    'similarity': similarity,
                    'start_index': i
                })
        
        return matches
    
    def calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two text strings"""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        if not union:
            return 0.0
        
        return len(intersection) / len(union)
    
    def highlight_from_rag_response(self, rag_response: str, retrieved_chunks: List[Dict]) -> Dict:
        """
        Main method to highlight PDF based on RAG response and retrieved chunks
        """
        highlight_results = {
            'timestamp': datetime.now().isoformat(),
            'rag_response': rag_response,
            'highlights_created': [],
            'failed_highlights': [],
            'total_matches': 0
        }
        
        print(f"\n🎯 Highlighting PDF based on RAG response and {len(retrieved_chunks)} retrieved chunks")
        
        for i, chunk in enumerate(retrieved_chunks):
            chunk_text = chunk.get('text', '')
            location = chunk.get('location', f'Unknown Location {i+1}')
            similarity_score = chunk.get('similarity', 0)
            
            print(f"\n📍 Processing chunk {i+1}: {location} (similarity: {similarity_score:.2f})")
            
            # Extract key sentences from the chunk for highlighting
            key_sentences = self.extract_key_sentences_from_chunk(chunk_text, rag_response)
            
            for sentence in key_sentences:
                matches = self.find_text_in_pdf(sentence)
                
                if matches:
                    highlight_info = {
                        'location': location,
                        'sentence': sentence,
                        'matches': len(matches),
                        'pages': [m['page'] for m in matches],
                        'chunk_similarity': similarity_score
                    }
                    highlight_results['highlights_created'].append(highlight_info)
                    highlight_results['total_matches'] += len(matches)
                    
                    print(f"  ✅ Found {len(matches)} matches for sentence")
                else:
                    highlight_results['failed_highlights'].append({
                        'location': location,
                        'sentence': sentence,
                        'reason': 'No matches found in PDF'
                    })
                    print(f"  ⚠️ No matches found for sentence")
        
        # Store in history
        self.highlight_history.append(highlight_results)
        
        return highlight_results
    
    def extract_key_sentences_from_chunk(self, chunk_text: str, rag_response: str) -> List[str]:
        """
        Extract the most relevant sentences from a chunk for highlighting
        Focus on sentences that appear related to the RAG response
        """
        sentences = self.split_into_sentences(chunk_text)
        rag_keywords = self.extract_keywords(rag_response)
        
        scored_sentences = []
        for sentence in sentences:
            if len(sentence.strip()) < 20:  # Skip very short sentences
                continue
            
            score = self.score_sentence_relevance(sentence, rag_keywords)
            if score > 0.3:  # Threshold for relevance
                scored_sentences.append((sentence, score))
        
        # Sort by relevance and return top sentences
        scored_sentences.sort(key=lambda x: x[1], reverse=True)
        
        # Return top 3 most relevant sentences from this chunk
        return [sent[0] for sent in scored_sentences[:3]]
    
    def split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        # Simple sentence splitting - can be improved with NLTK if needed
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from text"""
        # Remove common stop words and extract meaningful terms
        stop_words = {'the', 'is', 'at', 'which', 'on', 'and', 'or', 'but', 'in', 'with', 'a', 'an', 'as', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'should', 'could', 'can', 'may', 'might', 'must', 'to', 'of', 'for', 'from', 'by', 'this', 'that', 'these', 'those'}
        
        words = re.findall(r'\b\w+\b', text.lower())
        keywords = [w for w in words if len(w) > 3 and w not in stop_words]
        
        # Return unique keywords
        return list(set(keywords))
    
    def score_sentence_relevance(self, sentence: str, keywords: List[str]) -> float:
        """Score how relevant a sentence is based on keywords"""
        sentence_words = set(re.findall(r'\b\w+\b', sentence.lower()))
        keyword_set = set(keywords)
        
        matches = sentence_words.intersection(keyword_set)
        
        if not sentence_words:
            return 0.0
        
        return len(matches) / len(sentence_words)
    
    def create_highlighted_pdf(self, output_path: str, highlight_color: Tuple[float, float, float] = (1, 1, 0)) -> bool:
        """
        Create a new PDF with highlights based on the highlight history
        """
        if not self.pdf_doc or not self.highlight_history:
            print("❌ No PDF loaded or no highlights to apply")
            return False
        
        try:
            # Create a copy of the PDF document
            highlighted_doc = fitz.open(self.pdf_path)
            
            total_highlights = 0
            
            # Apply all highlights from history
            for result in self.highlight_history:
                for highlight_info in result['highlights_created']:
                    sentence = highlight_info['sentence']
                    matches = self.find_text_in_pdf(sentence)
                    
                    for match in matches:
                        page = highlighted_doc[match['page'] - 1]
                        
                        # Add highlight annotation
                        highlight_annot = page.add_highlight_annot(match['rect'])
                        highlight_annot.set_colors(stroke=highlight_color)
                        highlight_annot.set_info(content=f"RAG Highlight: {highlight_info['location']}")
                        highlight_annot.update()
                        
                        total_highlights += 1
            
            # Save the highlighted PDF
            highlighted_doc.save(output_path)
            highlighted_doc.close()
            
            print(f"✅ Highlighted PDF saved: {output_path}")
            print(f"📊 Total highlights applied: {total_highlights}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error creating highlighted PDF: {e}")
            return False
    
    def generate_web_compatible_data(self) -> Dict:
        """
        Generate data compatible with your HTML PDF highlighter
        """
        if not self.json_data:
            return {}
        
        # Extract location data for the web interface
        contextual_keypoints = []
        
        for keypoint in self.json_data.get('contextual_keypoints', []):
            contextual_keypoints.append({
                'section_number': keypoint.get('section_number'),
                'title': keypoint.get('location', keypoint.get('title', 'Unknown')),
                'location': keypoint.get('location', 'Unknown Location'),
                'theme': keypoint.get('theme', 'General Content'),
                'keypoints': keypoint.get('keypoints', '')
            })
        
        return {
            'contextual_keypoints': contextual_keypoints,
            'metadata': {
                'total_sections': len(contextual_keypoints),
                'source_pdf': self.pdf_path,
                'generated_for_highlighting': True
            }
        }
    
    def save_web_data(self, output_path: str):
        """Save data in format compatible with your HTML highlighter"""
        web_data = self.generate_web_compatible_data()
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(web_data, f, indent=2, ensure_ascii=False)
            
            print(f"✅ Web-compatible data saved: {output_path}")
            
        except Exception as e:
            print(f"❌ Error saving web data: {e}")
    
    def get_highlight_summary(self) -> Dict:
        """Get summary of all highlighting activities"""
        if not self.highlight_history:
            return {'message': 'No highlights created yet'}
        
        total_highlights = sum(len(h['highlights_created']) for h in self.highlight_history)
        total_failed = sum(len(h['failed_highlights']) for h in self.highlight_history)
        
        return {
            'total_highlight_sessions': len(self.highlight_history),
            'total_successful_highlights': total_highlights,
            'total_failed_highlights': total_failed,
            'success_rate': total_highlights / (total_highlights + total_failed) if (total_highlights + total_failed) > 0 else 0,
            'last_session': self.highlight_history[-1]['timestamp'] if self.highlight_history else None
        }
    
    def close(self):
        """Clean up resources"""
        if self.pdf_doc:
            self.pdf_doc.close()
