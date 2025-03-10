import logging
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer


class LLMInference:
    def __init__(self, config):
        """
        Initializes the LLMInference class with configuration settings.

        :param config: Configuration object containing settings for the application.
        """
        logging.info("Initializing LLMInference.")
        # Initialize the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(config.llm_model_id)
        
        # Pass the default decoding hyperparameters
        self.sampling_params = SamplingParams(
            temperature=config.temperature,
            top_p=config.top_p,
            repetition_penalty=config.repetition_penalty,
            max_tokens=config.max_tokens
        )
        
        # Initialize the LLM
        self.llm = LLM(model=config.llm_model_id)

        self.llm_base_prompt = config.llm_base_prompt
        logging.info("LLMInference initialized successfully.")

    
    def prompt_formatter(self, query: str, context_items: list[str]) -> str:
        """
        Augments query with text-based context from context_items.
        """
        logging.debug("Formatting prompt for query.")
        # Join context items into one dotted paragraph
        context = "- " + "\n- ".join(context_items)

        # Update base prompt with context items and query   
        base_prompt = self.llm_base_prompt.format(context=context, query=query)

        # Create prompt template for instruction-tuned model
        dialogue_template = [
            {"role": "user", "content": base_prompt}
        ]

        # Apply the chat template
        prompt = self.tokenizer.apply_chat_template(conversation=dialogue_template,
                                                    tokenize=False,
                                                    add_generation_prompt=True)
        return prompt

    def generate_response_from_context(self, query: str, context_items: list[str]):
        """
        Generates a response based on the provided query and context items.

        :param query: The input query string.
        :param context_items: A list of context strings to augment the query.
        :return: The generated response from the LLM.
        """
        logging.info("Generating response from context.")
        # Format the prompt using the provided context and query
        prompt = self.prompt_formatter(query, context_items)

        logging.info(f"llm infer Prompt: {prompt}")
        # Generate outputs
        outputs = self.llm.generate([prompt], self.sampling_params)

        # Collect the generated text from the outputs
        generated_responses = []
        for output in outputs:
            generated_text = output.outputs[0].text.strip()  # Ensure no leading/trailing whitespace
            generated_responses.append(generated_text)
        logging.info("Response generated successfully.")
        return generated_responses  # Return the list of generated responses