#!/usr/bin/env python3
"""
Script to download and test the google/flan-t5-large model for question paper generation
"""

import os
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

def download_model():
    """Download the flan-t5-large model and tokenizer"""
    print("🔄 Downloading google/flan-t5-large model...")
    
    # Model name
    model_name = "google/flan-t5-large"
    
    # Create cache directory
    cache_dir = "./model_cache"
    os.makedirs(cache_dir, exist_ok=True)
    
    try:
        # Download tokenizer
        print("📥 Downloading tokenizer...")
        tokenizer = T5Tokenizer.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            local_files_only=False  # Set to True if you want to use cached version only
        )
        
        # Download model
        print("📥 Downloading model...")
        model = T5ForConditionalGeneration.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            torch_dtype=torch.float32,  # Use float32 for CPU compatibility
            device_map="cpu"  # Use CPU to avoid MPS issues
        )
        
        print("✅ Model downloaded successfully!")
        print(f"📁 Model saved to: {cache_dir}")
        
        return model, tokenizer
        
    except Exception as e:
        print(f"❌ Error downloading model: {e}")
        return None, None

def test_model(model, tokenizer):
    """Test the model with a simple question generation prompt"""
    print("\n🧪 Testing model with sample prompt...")
    
    # Sample text for question generation
    sample_text = """
    Photosynthesis is the process by which plants convert light energy into chemical energy. 
    During this process, plants use sunlight, water, and carbon dioxide to produce glucose and oxygen. 
    The process occurs in the chloroplasts of plant cells, specifically in the thylakoid membranes.
    """
    
    # Create a prompt for question generation
    prompt = f"Given the following text, generate 3 multiple choice questions with 4 options each:\n\n{sample_text}"
    
    try:
        # Tokenize input
        inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
        
        # Generate response
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                max_length=500,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Decode response
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        print("📝 Generated Questions:")
        print(response)
        
    except Exception as e:
        print(f"❌ Error testing model: {e}")

def main():
    """Main function to download and test the model"""
    print("🚀 Starting flan-t5-large model download...")
    
    # Download model
    model, tokenizer = download_model()
    
    if model and tokenizer:
        # Test the model
        test_model(model, tokenizer)
        
        print("\n🎉 Setup complete! You can now use the model for question paper generation.")
        print("\n💡 Usage tips:")
        print("- The model is saved in ./model_cache/")
        print("- Use torch.float16 for memory efficiency")
        print("- Adjust temperature (0.1-1.0) to control creativity")
        print("- Use max_length parameter to control response length")
    else:
        print("❌ Failed to download model. Please check your internet connection and try again.")

if __name__ == "__main__":
    main() 