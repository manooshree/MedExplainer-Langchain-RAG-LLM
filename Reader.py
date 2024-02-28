from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# Initialize and configure the reader model
READER_MODEL_NAME = "HuggingFaceH4/zephyr-7b-beta"

def initialize_reader(model_name: str):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype="bfloat16",
    )
    model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_config)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    reader_llm = pipeline(
        model=model,
        tokenizer=tokenizer,
        task="text-generation",
        do_sample=True,
        temperature=0.2,
        repetition_penalty=1.1,
        return_full_text=False,
        max_new_tokens=500,
    )
    return reader_llm

# Function to generate an answer using the reader model
def generate_answer(reader, prompt: str):
    answer = reader(prompt)[0]["generated_text"]
    return answer

# Example usage
if __name__ == "__main__":
    reader_llm = initialize_reader(READER_MODEL_NAME)
    test_prompt = "What is the capital of France? Answer:"
    answer = generate_answer(reader_llm, test_prompt)
    print("Generated Answer:", answer)
