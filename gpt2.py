from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForCausalLM

# Initialize Flask app
app = Flask(__name__)

# Allow CORS for frontend access
from flask_cors import CORS
CORS(app)

model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

@app.route("/generate", methods=["POST"])
def generate():
    # Get the input text from the frontend
    data = request.json
    prompt = data.get("prompt", "")
    # Generate a response
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(inputs["input_ids"], max_length=100, num_return_sequences=1)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Return the response
    return jsonify({"response": response})

@app.route("/", methods=["GET"])
def hello():
    return jsonify({"message": "Oh So you found this section of my server well it has greater purpose than you think"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000) 
