import onnxruntime as ort
import numpy as np
import cv2
import argparse
import os

def test_inference(model_path, image_path, output_path, scale=2):
    # Load ONNX model
    session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    
    # Load and preprocess image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image {image_path}")
        return
        
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # CRITICAL: Real-CUGAN expects 0-255 range, so we don't divide by 255.0
    img = img.astype(np.float32)
    img = np.transpose(img, (2, 0, 1)) # HWC to CHW
    img = np.expand_dims(img, axis=0)  # BCHW
    
    # Run inference
    inputs = {session.get_inputs()[0].name: img}
    outputs = session.run(None, inputs)
    output = outputs[0][0]
    
    # Post-process
    output = np.transpose(output, (1, 2, 0)) # CHW to HWC
    output = np.clip(output, 0, 255).astype(np.uint8)
    output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
    
    # Save result
    cv2.imwrite(output_path, output)
    print(f"Inference completed! Result saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test ONNX model with Real-CUGAN (0-255 range)")
    parser.add_argument("--model", type=str, required=True, help="Path to ONNX model")
    parser.add_argument("--input", type=str, required=True, help="Path to input image")
    parser.add_argument("--output", type=str, default="output.png", help="Path to output image")
    parser.add_argument("--scale", type=int, default=2, help="Upscale factor")
    
    args = parser.parse_args()
    
    if os.path.exists(args.model) and os.path.exists(args.input):
        test_inference(args.model, args.input, args.output, args.scale)
    else:
        print("Model or input file not found.")
