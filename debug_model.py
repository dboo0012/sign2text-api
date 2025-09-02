"""
Debug script to test model loading separately
"""
import sys
import os

# Add the model directory to path
model_dir = os.path.join(os.path.dirname(__file__), 'model')
sys.path.insert(0, model_dir)

print(f"Model directory: {model_dir}")
print(f"Python path: {sys.path[:3]}")

try:
    from models.trans_model_inter_vn import TransBaseModel
    print("✓ Successfully imported TransBaseModel")
except ImportError as e:
    print(f"✗ Failed to import TransBaseModel: {e}")
    print("Available modules in models:")
    models_dir = os.path.join(model_dir, 'models')
    if os.path.exists(models_dir):
        print(os.listdir(models_dir))
    else:
        print("Models directory not found")

try:
    from utils.easydict import EasyDict
    print("✓ Successfully imported EasyDict")
except ImportError as e:
    print(f"✗ Failed to import EasyDict: {e}")
    print("Available modules in utils:")
    utils_dir = os.path.join(model_dir, 'utils')
    if os.path.exists(utils_dir):
        print(os.listdir(utils_dir))
    else:
        print("Utils directory not found")

# Test file paths
weights_path = "model/how2sign/vn_model/glofe_vn_how2sign_0224.pt"
tokenizer_path = "model/notebooks/how2sign/how2sign-bpe25000-tokenizer-uncased"

print(f"\nFile checks:")
print(f"Weights exist: {os.path.exists(weights_path)} - {weights_path}")
print(f"Tokenizer exists: {os.path.exists(tokenizer_path)} - {tokenizer_path}")

# Test tokenizer loading
try:
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    print("✓ Tokenizer loaded successfully")
except Exception as e:
    print(f"✗ Tokenizer loading failed: {e}")
