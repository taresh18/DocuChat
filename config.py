class Config:
    # Model configuration
    llm_model_id = "Qwen/Qwen2.5-1.5B-Instruct"  # Name of the LLM model
    embedding_model_id = "intfloat/multilingual-e5-large-instruct"  # ID of the embedding model
    
    # PDF Preprocessing settings
    pdf_path = "data/ppmp.pdf"  # Path to the PDF file
    ignore_pages = [0, 1, 2]  # Pages to ignore during extraction
    minimum_sentence_length = 20  # Minimum length of sentences to keep
    
    # Chunking settings
    buffer_size = 1  # Buffer size for sentence chunking
    threshold_percentile = 95  # Percentile threshold for chunking
    context_window = 450  # Maximum context window size for embeddings

    chapters_info_path = "data/chapters_info.json"  # Path to save chapters info
    do_preprocessing_steps = False  # Whether to generate chapters info
    
    # Retrieval settings
    embeddings_save_path = "data/embeddings.pt"  # Path to save embeddings
    top_k = 5  # Number of top similar embeddings to return
    
    # Logging settings
    log_file = "app.log"  # Log file name
    log_level = "INFO"  # Logging level

    # LLM sampling parameters
    temperature = 0.7  # Temperature for sampling
    top_p = 0.8  # Top-p sampling
    repetition_penalty = 1.05  # Repetition penalty
    max_tokens = 512  # Maximum tokens to generate

    # Other settings can be added here as needed 
    embedding_generation_template = (
        "Generate a dense semantic representation for the following excerpt. "
        "Ensure the embedding captures key technical themes, context, and narrative details.\n\n"
        "Chapter: {chapter_name}\n"
        "{text_chunk}\n"
    )

    query_template = (
        "Generate a dense semantic representation for the following query ensuring that the embedding captures the technical intent.\n\n"
        "Query: {query}\n"
    )

    llm_base_prompt = """Based on the following context items, please answer the query.
    Give yourself room to think by extracting relevant passages from the context before answering the query.
    Don't return the thinking, only return the answer.
    Make sure your answers are as explanatory as possible.
    Use the following examples as reference for the ideal answer style.
    \nExample 1:
    Query: What are the fat-soluble vitamins?
    Answer: The fat-soluble vitamins include Vitamin A, Vitamin D, Vitamin E, and Vitamin K. These vitamins are absorbed along with fats in the diet and can be stored in the body's fatty tissue and liver for later use. Vitamin A is important for vision, immune function, and skin health. Vitamin D plays a critical role in calcium absorption and bone health. Vitamin E acts as an antioxidant, protecting cells from damage. Vitamin K is essential for blood clotting and bone metabolism.
    \nExample 2:
    Query: What are the causes of type 2 diabetes?
    Answer: Type 2 diabetes is often associated with overnutrition, particularly the overconsumption of calories leading to obesity. Factors include a diet high in refined sugars and saturated fats, which can lead to insulin resistance, a condition where the body's cells do not respond effectively to insulin. Over time, the pancreas cannot produce enough insulin to manage blood sugar levels, resulting in type 2 diabetes. Additionally, excessive caloric intake without sufficient physical activity exacerbates the risk by promoting weight gain and fat accumulation, particularly around the abdomen, further contributing to insulin resistance.
    \nExample 3:
    Query: What is the importance of hydration for physical performance?
    Answer: Hydration is crucial for physical performance because water plays key roles in maintaining blood volume, regulating body temperature, and ensuring the transport of nutrients and oxygen to cells. Adequate hydration is essential for optimal muscle function, endurance, and recovery. Dehydration can lead to decreased performance, fatigue, and increased risk of heat-related illnesses, such as heat stroke. Drinking sufficient water before, during, and after exercise helps ensure peak physical performance and recovery.
    \nNow use the following context items to answer the user query:
    {context}
    \nRelevant passages: <extract relevant passages from the context here>
    User query: {query}
    Answer:"""