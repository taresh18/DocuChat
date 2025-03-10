import logging
import torch
from vllm import LLM
from tqdm import tqdm
import numpy as np


class GenerateEmbeddings:
    def __init__(self, config):
        """
        Initialize the GenerateEmbeddings class with the specified model.

        :param config: Configuration object containing settings for the application.
        """
        logging.info("Initializing GenerateEmbeddings.")
        self.llm = LLM(model=config.embedding_model_id, task="embed", trust_remote_code=True)
        self.prompt_template = config.embedding_generation_template
        self.embeddings_save_path = config.embeddings_save_path
        logging.info("GenerateEmbeddings initialized successfully.")

    def generate_embeddings(self, chapters_info):
        """
        Generate embeddings for each chunk in chapters_info.
        :param chapters_info: List of dictionaries containing chapter information with finalized chunks.
        :return: List of embeddings for all chunks.
        """
        logging.info("Generating embeddings for chapters.")
        all_embeddings = []

        for chapter in tqdm(chapters_info, desc="Generating embeddings"):
            chapter_title = chapter.get('chapter_title', 'Untitled Chapter')  # Get chapter title or default
            prompts = [
                    self.prompt_template.format(chapter_name=chapter_title, text_chunk=chunk)
                    for chunk in chapter['finalized_chunks']
                ]
            embeddings = self.llm.encode(prompts)
            for embedding in embeddings:
                all_embeddings.append(embedding.outputs.data)

        # Convert the list of embeddings to a PyTorch tensor
        embeddings_tensor = torch.tensor(np.array(all_embeddings))
        # Save the embeddings tensor to disk
        torch.save(embeddings_tensor, self.embeddings_save_path)
        logging.info("Embeddings generated and saved successfully.")
        return embeddings_tensor 