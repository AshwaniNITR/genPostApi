from flask import Flask, request, jsonify
from flask_cors import CORS  
from diffusers import StableDiffusionPipeline
import torch

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": ["http://localhost:3000", "https://myapp-lac-nine.vercel.app"]}})

# ✅ Load the model ONCE when the server starts
pipe = StableDiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-1",
    torch_dtype=torch.float16
).to("cuda")

@app.route("/generate", methods=["POST"])
def generate_image():
    data = request.json
    prompt = data.get("prompt", "A futuristic cityscape at night")
    num_steps = data.get("num_steps", 50)
    guidance = data.get("guidance", 7.5)

    # 🔥 Generate the image
    image = pipe(prompt=prompt, num_inference_steps=num_steps, guidance_scale=guidance).images[0]

    # Save the image
    image_path = "output.png"
    image.save(image_path)

    return jsonify({"message": "Image generated!", "image_url": f"http://your-server.com/{image_path}"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
