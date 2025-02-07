import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM

# Initialize FastAPI app
app = FastAPI()

# Load model only once on startup
print("Loading SQLCoder model... (This may take a while on CPU)")
model_name = "defog/sqlcoder-7b-2"
tokenizer = AutoTokenizer.from_pretrained(model_name)

#  Fix: Add a new [PAD] token to prevent padding errors
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,  # Lower precision for CPU efficiency
    device_map="cpu"
)
model.resize_token_embeddings(len(tokenizer))  #  Ensure the model knows about the new token

# Store conversation history
chat_history = []

class QueryRequest(BaseModel):
    user_input: str

@app.post("/generate")
def generate_sql(query: QueryRequest):
    """
    Generates an SQL query based on user input while keeping conversation history.
    """
    global chat_history

    # Maintain chat history for context
    formatted_input = "Chat History:\n" + "\n".join(chat_history) + f"\nUser: {query.user_input}\nAI:"

    # Tokenize input
    inputs = tokenizer(
        formatted_input, 
        return_tensors="pt", 
        padding=True,  # Allow padding
        truncation=True, 
        max_length=2048
    )

    with torch.no_grad():  # Disable gradient computation for efficiency
        output = model.generate(
            inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=200,  # Limit response length
            pad_token_id=tokenizer.pad_token_id,  #  Fix: Uses new [PAD] token
            do_sample=False,  # Ensure deterministic output
            temperature=0,  # More stable SQL queries
            top_p=1.0
        )

    response = tokenizer.decode(output[0], skip_special_tokens=True)

    # Update chat history
    chat_history.append(f"User: {query.user_input}")
    chat_history.append(f"AI: {response}")
    chat_history = chat_history[-10:]  # Keep last 10 exchanges for memory efficiency

    return {"response": response}
