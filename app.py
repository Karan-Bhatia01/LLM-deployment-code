import gradio as gr
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load model and tokenizer
model_name = "gpt2"  # You can change this to any model like "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF"
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Define function for text generation
def generate_text(input_text):
    inputs = tokenizer.encode(input_text, return_tensors="pt")
    outputs = model.generate(inputs, max_length=100, num_return_sequences=1)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text

# Create Gradio interface
iface = gr.Interface(fn=generate_text, inputs="text", outputs="text", title="LLM Text Generation")

# Launch the interface
iface.launch()
