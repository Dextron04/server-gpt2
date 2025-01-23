from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForCausalLM
from flask_cors import CORS
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Load GPT-2 model
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token  # 

@app.route("/gpt2/generate", methods=["POST"])
def generate():
    data = request.json
    prompt = data.get("prompt", "")
    logging.info(f"Received prompt: {prompt}")

    if not prompt.strip():
        logging.error("Empty prompt received")
        return jsonify({"error": "Prompt cannot be empty"}), 400
    
    if len(prompt) > 1000:
        logging.error("Prompt is too long")
        return jsonify({"error": "Prompt is too long"}), 400

    try:
        inputs = tokenizer(
            prompt, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=512
        )
        outputs = model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=100,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id
        )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        logging.info(f"Generated response: {response}")
        return jsonify({"response": response})
    except Exception as e:
        logging.error(f"Error generating response: {e}")
        return jsonify({"error": "An error occurred while generating the response"}), 500

@app.route("/gpt2/", methods=["GET"])
def hello():
    return jsonify({"message": "Oh So you found this section of my server well it has greater purpose than you think"})

@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "ok", "model": model_name})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
