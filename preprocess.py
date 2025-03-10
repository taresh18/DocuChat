import re
import unicodedata
import fitz 
from spacy.lang.en import English

class PDFPreprocessor:
    def __init__(self, config):
        """
        Initialize the PDFPreprocessor.

        :param config: Configuration object containing settings for the application.
        """
        self.pdf_path = config.pdf_path
        self.ignore_pages = config.ignore_pages
        self.minimum_sentence_length = config.minimum_sentence_length
        
        # Set up a basic spaCy pipeline using the English model.
        self.nlp = English()
        
        # Add common abbreviations as special cases to avoid erroneous sentence splits.
        special_cases = [
            "Dr.", "Mr.", "Mrs.", "Ms.", "Prof.", "Inc.", "Ltd.", "Co.", "Corp.",
            "e.g.", "i.e.", "etc.", "vs.", "Fig.", "Figs.", "No.", "Vol.", "Ed.",
            "Jr.", "Sr.", "al."
        ]
        for case in special_cases:
            self.nlp.tokenizer.add_special_case(case, [{"ORTH": case}])
        
        # Add the sentencizer with custom punctuation.
        config = {"punct_chars": [".", "!", "?"]}
        self.nlp.add_pipe("sentencizer", config=config)
    
    def decode_if_bytes(self, raw_input):
        """
        If raw_input is of type bytes, decode using UTF-8.
        """
        if isinstance(raw_input, bytes):
            return raw_input.decode('utf-8')
        return raw_input

    def clean_pdf_text(self, raw_text):
        """
        Clean and normalize extracted text from a PDF.
        For chapter pages with a header block, extract only the chapter title and remove everything else.
        Then apply general cleanup.
        """
        # Ensure input is a Unicode string.
        text = self.decode_if_bytes(raw_text)
        
        # Step 1: Unicode normalization (NFC is usually a good choice)
        text = unicodedata.normalize('NFC', text)
        
        # Step 2: Remove copyright header block.
        copyright_pattern = (
            r'\n\s*Programming Massively Parallel Processors\. DOI:\s*https?://[^\n]+\s*'
            r'\n\s*©\s*\d{4}\s*Elsevier Inc\. All rights reserved\.'
        )
        text = re.sub(copyright_pattern, '', text, flags=re.IGNORECASE)
        
        # Step 3: Remove the chapter header block with only the chapter title.
        # This pattern assumes the header starts with "CHAPTER", a chapter number, then the chapter title,
        # then "Chapter Outline" and ends with either "Exercises" or "References".
        chapter_pattern = r'(?s)^CHAPTER\s*\n\s*\d+\s*\n(.*?)\n\s*Chapter Outline.*\n\s*(?:Exercises|References|Future outlook)\s*\n'

        text = re.sub(chapter_pattern, '', text, flags=re.IGNORECASE)
        
        # Step 4: Remove trailing page headers (pattern: newline, digits, newline, then header text).
        text = re.sub(r'\n\d+\n.*$', '', text, flags=re.DOTALL)
        
        # Step 5: Remove lines that are mainly punctuation (e.g., lines consisting mostly of dots and numbers).
        lines = text.splitlines()
        clean_lines = [line for line in lines if not re.fullmatch(r'[\.\s\d]+', line)]
        text = "\n".join(clean_lines)
        
        # Step 6: Remove figure references (e.g., "FIGURE 4.8").
        text = re.sub(r'\bFIGURE\s+\d+(\.\d+)?\b', '', text, flags=re.IGNORECASE)
        
        # Step 7: Fix hyphenated line breaks (e.g., "hy-\nphenated" becomes "hyphenated").
        text = re.sub(r'(\w)-\s*\n\s*(\w)', r'\1\2', text)
        
        # Step 8: Remove extra newlines, tabs, and multiple spaces.
        text = re.sub(r'[\r\n\t]+', ' ', text)
        text = re.sub(r' +', ' ', text).strip()
        
        # Step 9: Remove unwanted special characters while preserving punctuation,
        # the multiplication sign (×) and the percent sign (%).
        text = re.sub(r'[^\w\s\.,:;!?()\-\u00D7%]', '', text)
        
        # Step 10: Remove lines that contain only numbers.
        text = "\n".join(line for line in text.splitlines() if not re.fullmatch(r'\s*\d+\s*', line))
        
        return text

    def get_sentences(self, text):
        """
        Split the cleaned text into a list of sentences using spaCy.
        Only sentences with length greater than or equal to self.minimum_sentence_length are returned.
        """
        doc = self.nlp(text)
        sentences = [sent.text.strip() for sent in doc.sents if len(sent.text.strip()) >= self.minimum_sentence_length]
        return sentences

    def extract_chapters_info(self):
        """
        Extract chapter information from the PDF.
        
        For each page, check if it contains a chapter header using a regex pattern.
        If so, record the chapter number and title as the start of a chapter.
        Then, accumulate text from subsequent pages until an "Exercises" or "References"
        marker is found, at which point that chapter is finalized.
        
        Returns a list of dictionaries, where each dictionary contains:
            - 'chapter_number': The chapter number as a string.
            - 'chapter_title': The extracted chapter title.
            - 'start_page': The page number (0-indexed) where the chapter starts.
            - 'end_page': The page number where the chapter content ends.
            - 'combined_text': The cleaned text from the start page up to (but not including) text after the marker.
        """
        doc = fitz.open(self.pdf_path)
        chapters = []
        current_chapter = None
        current_text = ""
        
        # Regex to capture chapter header: captures chapter number and title.
        chapter_header_pattern = r'(?s)^CHAPTER\s*\n\s*(\d+)\s*\n(.*?)\n\s*Chapter Outline'
        # Pattern to detect end-of-chapter marker (Exercises or References).
        marker_pattern = r'\n\s*(Exercises|References)\s*\n'
        
        for page_num in range(len(doc)):
            raw_text = doc[page_num].get_text()
            
            marker_match = re.search(marker_pattern, raw_text, flags=re.IGNORECASE)
            # Check if the page contains a chapter header.
            chap_match = re.search(chapter_header_pattern, raw_text, flags=re.IGNORECASE)
            if chap_match:
                # If a chapter is already in progress, finalize it.
                if current_chapter is not None:
                    current_chapter['end_page'] = page_num - 1
                    current_chapter['combined_text'] = current_text.strip()
                    current_chapter['sentences'] = self.get_sentences(current_text.strip())
                    chapters.append(current_chapter)
                    current_chapter = None
                    current_text = ""
                
                # Start a new chapter.
                chapter_number = chap_match.group(1).strip()
                chapter_title = re.sub(r'\s+', ' ', chap_match.group(2)).strip()
                # Remove "With special contributions" and everything after if present
                if "With special contributions" in chapter_title:
                    chapter_title = chapter_title.split("With special contributions")[0].strip()
                current_chapter = {
                    'chapter_number': chapter_number,
                    'chapter_title': chapter_title,
                    'start_page': page_num,
                    'end_page': None,
                    'combined_text': ''
                }
                current_text += self.clean_pdf_text(raw_text) + " "
            # if "Exercises" comes at the starting, we skip processing the page
            elif raw_text.lstrip().startswith("Exercises"):
                current_chapter['end_page'] = page_num-1
                current_chapter['combined_text'] = current_text.strip()
                current_chapter['sentences'] = self.get_sentences(current_text.strip())
                chapters.append(current_chapter)
                current_chapter = None
                current_text = ""
            # execute logic where "Exercises" appear within a page
            elif marker_match and current_chapter is not None:
                cutoff = marker_match.start()
                page_body = raw_text[:cutoff]
                current_text += self.clean_pdf_text(page_body) + " "
                current_chapter['end_page'] = page_num
                current_chapter['combined_text'] = current_text.strip()
                current_chapter['sentences'] = self.get_sentences(current_text.strip())
                chapters.append(current_chapter)
                current_chapter = None
                current_text = ""
            else:
                # If inside a chapter, accumulate page text.
                if current_chapter is not None:
                    current_text += self.clean_pdf_text(raw_text) + " "
        
        # Finalize any chapter still in progress.
        if current_chapter is not None:
            current_chapter['end_page'] = len(doc) - 1
            current_chapter['combined_text'] = current_text.strip()
            current_chapter['sentences'] = self.get_sentences(current_text.strip())
            chapters.append(current_chapter)
        
        return chapters


if __name__ == "__main__":
    from config import Config
    config = Config()
    # Specify pages to ignore (0-indexed) if needed.
    preprocessor = PDFPreprocessor(config)
    # Extract chapter information.
    chapters_info = preprocessor.extract_chapters_info(config.pdf_path)

