import logging
from config import Config  # Import the Config class
from llm_infer import LLMInference
from retreival import Retrieval
from chunker import SemanticChunker
from embeddings import GenerateEmbeddings
from preprocess import PDFPreprocessor
from tqdm import tqdm
import sys
import json  # Import json for saving and loading the chapters_info

# Set up logging
def setup_logging(log_file, log_level):
    logging.basicConfig(filename=log_file, level=getattr(logging, log_level), 
                        format='%(asctime)s - %(levelname)s - %(message)s')
    
def preprocessing_steps(config, logging):
    pdf_preprocessor = PDFPreprocessor(config=config)
    chapters_info = pdf_preprocessor.extract_chapters_info()
    logging.info(f"Extracted {len(chapters_info)} chapters from the PDF.")

    semantic_chunker = SemanticChunker(config=config)
    chapters_info = semantic_chunker.process_all_chapters(chapters_info)

    logging.info(f"Saving chapters_info to: {config.chapters_info_path}")
    with open(config.chapters_info_path, 'w') as f:  # Specify the file name
        json.dump(chapters_info, f, indent=4)  # Save the dict as JSON
    
    embedding_generator = GenerateEmbeddings(config=config)
    embedding_generator.generate_embeddings(chapters_info)


def main():
    config = Config()  # Create a config object
    setup_logging(config.log_file, config.log_level)  # Set up logging

    logging.info("Starting the RAG application.")

    if config.do_preprocessing_steps:
        preprocessing_steps(config, logging)

    # Load chapters_info from the file
    with open(config.chapters_info_path, 'r') as f:  # Read the file
        chapters_info = json.load(f)  # Load the dict from JSON
    
    # Initialize retrieval
    retrieval = Retrieval(config=config, chapter_info=chapters_info)
    
    # Example query
    query = "explain thread, block and grid in cuda"
    similar_chunks = retrieval.find_similar_embeddings(query)
    
    # Generate response from context
    context_items = [chunk['sentence_chunk'] for chunk in similar_chunks]
    logging.info(f"Context items: {context_items}")
    llm_inference = LLMInference(config=config)
    response = llm_inference.generate_response_from_context(query, context_items=context_items)
    
    # Output the response
    logging.info("Response:\n\n" + response[0])

if __name__ == "__main__":
    main()
