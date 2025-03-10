import logging
import re
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from vllm import LLM, TokensPrompt


class SemanticChunker:
    def __init__(self, config):
        """
        Initializes the SemanticChunker class with configuration settings.

        :param config: Configuration object containing settings for the application.
        :param llm: The language model to use for chunking.
        :param tokenizer: The tokenizer to use for processing text.
        """
        logging.info("Initializing SemanticChunker.")
        self.llm = LLM(model=config.embedding_model_id, task="embed", trust_remote_code=True)
        self.tokenizer = self.llm.llm_engine.get_tokenizer_group()
        self.buffer_size = config.buffer_size
        self.threshold_percentile = config.threshold_percentile
        self.context_window = config.context_window

        self.sentences = None
        logging.info("SemanticChunker initialized successfully.")

    def preprocess_sentences(self, sentences):
        """
        Preprocess the sentences to create a sentences_dict and combined_sentences.
        :param sentences: List of sentences to preprocess.
        :return: List of dictionaries containing sentence and index.
        """
        logging.debug("Preprocessing sentences.")
        # Create sentences_dict
        sentences_dict = [{'sentence': x, 'index': i} for i, x in enumerate(sentences)]
        
        # Combine sentences into combined_sentences
        combined_sentences = self.combine_sentences(sentences_dict)
        
        logging.debug("Sentences preprocessed successfully.")
        return combined_sentences

    def combine_sentences(self, sentences):
        """
        Combine sentences into a single string based on a buffer size.
        :param sentences: List of sentence dictionaries.
        :return: List of sentences with combined sentences added.
        """
        logging.debug("Combining sentences based on buffer size.")
        for i in range(len(sentences)):
            combined_sentence = ''

            # Add sentences before the current one, based on the buffer size.
            for j in range(i - self.buffer_size, i):
                if j >= 0:
                    combined_sentence += sentences[j]['sentence'] + ' '

            # Add the current sentence
            combined_sentence += sentences[i]['sentence']

            # Add sentences after the current one, based on the buffer size
            for j in range(i + 1, i + 1 + self.buffer_size):
                if j < len(sentences):
                    combined_sentence += ' ' + sentences[j]['sentence']

            # Store the combined sentence in the current sentence dict
            sentences[i]['combined_sentence'] = combined_sentence

        logging.debug("Sentences combined successfully.")
        return sentences

    def split_into_chunks(self, text):
        """
        Split text into chunks that fit within the context window.
        """
        sentences = re.split(r'(?<=[.!?]) +', text)  # Split by sentence boundaries
        chunks = []
        current_chunk = []

        for sentence in sentences:
            current_chunk.append(sentence)
            current_length = sum(len(self.tokenizer.encode(s)) for s in current_chunk)

            if current_length > self.context_window:
                # If the current chunk exceeds the context window, finalize the chunk
                chunks.append(' '.join(current_chunk[:-1]))  # Add all but the last sentence
                current_chunk = [sentence]  # Start a new chunk with the last sentence

        if current_chunk:
            chunks.append(' '.join(current_chunk))  # Add any remaining sentences as a chunk

        return chunks

    def generate_embeddings(self):
        """
        Generate embeddings for the combined sentences, ensuring they fit within the context window.
        """
        logging.info("Generating embeddings for combined sentences.")
        for combined_sentence in tqdm(self.sentences):
            prompt = combined_sentence['combined_sentence']
            prompt_token_ids = self.tokenizer.encode(prompt=prompt)
            combined_sentence['token_ids'] = prompt_token_ids
            combined_sentence['token_id_length'] = len(prompt_token_ids)

            # Split into chunks if the token length exceeds the context window
            if combined_sentence['token_id_length'] > self.context_window:
                chunks = self.split_into_chunks(prompt)
                combined_sentence['token_chunks'] = chunks
                combined_sentence['token_id_chunks'] = [self.tokenizer.encode(chunk) for chunk in chunks]
            else:
                combined_sentence['token_chunks'] = [prompt]
                combined_sentence['token_id_chunks'] = [prompt_token_ids]

        # Generate embeddings for each chunk
        all_embeddings = []
        for combined_sentence in self.sentences:
            token_id_chunks = combined_sentence['token_id_chunks']
            embeddings = self.llm.encode([TokensPrompt(prompt_token_ids=chunk) for chunk in token_id_chunks])
            all_embeddings.extend([embedding.outputs.data for embedding in embeddings])  # Ensure correct extraction

        for i, combined_sentence in enumerate(self.sentences):
            combined_sentence['token_embeddings'] = all_embeddings[i]

        logging.info("Embeddings generated successfully.")

    def calculate_cosine_distances(self):
        """
        Calculate cosine distances between consecutive sentence embeddings.
        """
        distances = []
        for i in range(len(self.sentences) - 1):
            embedding_current = self.sentences[i]['token_embeddings']
            embedding_next = self.sentences[i + 1]['token_embeddings']
            similarity = cosine_similarity([embedding_current], [embedding_next])[0][0]
            distance = 1 - similarity
            distances.append(distance)
            self.sentences[i]['distance_to_next'] = distance

        return distances

    def chunk_sentences(self):
        """
        Chunk sentences based on calculated distances and a defined threshold.
        """
        distances = self.calculate_cosine_distances()
        breakpoint_distance_threshold = np.percentile(distances, self.threshold_percentile)

        chunks = []
        start_index = 0
        indices_above_thresh = [i for i, x in enumerate(distances) if x > breakpoint_distance_threshold]
        for index in indices_above_thresh:
            end_index = index
            group = self.sentences[start_index:end_index + 1]
            combined_text = ' '.join([d['sentence'] for d in group])
            chunks.append(combined_text)
            start_index = index + 1

        if start_index < len(self.sentences):
            combined_text = ' '.join([d['sentence'] for d in self.sentences[start_index:]])
            chunks.append(combined_text)

        return chunks

    def finalize_chunks(self, chunks):
        """
        Further split the final chunks if their token length exceeds the context window.
        :param chunks: List of chunks produced by chunk_sentences.
        :return: List of finalized chunks that fit within the context window.
        """
        finalized_chunks = []
        
        for chunk in chunks:
            # Check the token length of the chunk
            token_ids = self.tokenizer.encode(chunk)
            token_length = len(token_ids)

            if token_length > self.context_window:
                # If the chunk exceeds the context window, split it
                sub_chunks = self.split_into_chunks(chunk)
                finalized_chunks.extend(sub_chunks)
            else:
                finalized_chunks.append(chunk)

        return finalized_chunks

    def process_all_chapters(self, chapters_info):
        """
        Execute the chunking process for all chapters in chapters_info.
        :param chapters_info: List of dictionaries containing chapter information.
        :return: List of finalized chunks for all chapters.
        """        
        
        for chapter in chapters_info:
            # Extract sentences from the chapter
            sentences = chapter['sentences']
            # Initialize the chunker for the current chapter
            self.sentences = self.preprocess_sentences(sentences)
            # Process the current chapter
            self.generate_embeddings()
            chunks = self.chunk_sentences()
            finalized_chunks = self.finalize_chunks(chunks)
            chapter['finalized_chunks'] = finalized_chunks
        
        return chapters_info