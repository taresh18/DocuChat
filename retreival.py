import logging
from vllm import LLM

import torch
from torch.nn.functional import cosine_similarity

import numpy as np


class Retrieval:
    def __init__(self, config, chapter_info: list[dict]):
        """
        Initializes the Retrieval class with chapter information and configuration settings.

        :param config: Configuration object containing settings for the application.
        :param chapter_info: A list of dictionaries containing chapter information with sentence chunks.
        """
        logging.info("Initializing Retrieval.")
        self.chapter_info = chapter_info
        self.embeddings = torch.load(config.embeddings_save_path)  
        self.llm = LLM(model=config.embedding_model_id, task="embed", trust_remote_code=True)
        self.sentence_chunks = self._extract_sentence_chunks()  # Extract sentence chunks from chapters_info
        self.query_template = config.query_template
        self.top_k = config.top_k
        logging.info("Retrieval initialized successfully.")

    def _extract_sentence_chunks(self) -> list[str]:
        """
        Extracts sentence chunks from chapters_info.

        :return: A flat list of sentence chunks.
        """
        logging.debug("Extracting sentence chunks from chapters.")
        chunks = []
        for chapter in self.chapter_info:
            chunks.extend(chapter.get('finalized_chunks', []))  # Get finalized chunks from each chapter
        logging.debug("Sentence chunks extracted successfully.")
        return chunks

    def find_similar_embeddings(self, query: str) -> list[dict]:
        """
        Takes a query, generates its embedding, and finds the nearest embedding vectors.

        :param query: The input query string.
        :param top_k: The number of top similar embeddings to return.
        :return: A list of dictionaries containing the nearest sentence chunks and their similarity scores.
        """
        logging.info(f"Finding similar embeddings for query: '{query}'")
        query_prompt = self.query_template.format(query=query)
        logging.info(f"Query prompt: {query_prompt}")
        query_embedding = self.llm.encode(query_prompt)
        
        all_embeddings = [embedding.outputs.data for embedding in query_embedding]
        query_embedding = torch.tensor(np.array(all_embeddings))
        logging.info(f"Query embedding shape: {query_embedding.shape}, {query_embedding.dtype}, {query_embedding}")

        # Calculate cosine similarity between the query embedding and the stored embeddings
        similarities = cosine_similarity(query_embedding, self.embeddings)
        logging.info(f"Similarities shape: {similarities.shape}, {similarities.dtype}, {similarities}")

        # Get the top k similar embeddings (for example, top 5)
        top_k_indices = torch.topk(similarities, k=self.top_k).indices
        logging.info(f"Top k indices: {top_k_indices}")

        # Retrieve the corresponding sentence chunks from chapter_info
        similar_chunks = [{"sentence_chunk": self.sentence_chunks[i], "similarity": similarities[i].item()} for i in top_k_indices]
        logging.info(f"Found {len(similar_chunks)} similar chunks.")
        return similar_chunks
