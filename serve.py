from flask import Flask, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer

app = Flask(__name__)
tokenizer = AutoTokenizer.from_pretrained("./models/tinyllama")
model = AutoModelForCausalLM.from_pretrained("./models/tinyllama")

@app.route("/generate", methods=["POST"])
def generate():
    text = request.json["prompt"]
    inputs = tokenizer(text, return_tensors="pt")
    output = model.generate(**inputs, max_new_tokens=100)
    return jsonify({"response": tokenizer.decode(output[0])})

app.run(port=8080)


