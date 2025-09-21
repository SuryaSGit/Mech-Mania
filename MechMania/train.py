import sys
sys.path.append('.')  # Add current directory to path

from main.py import Soccer4v4MLStrategy  # Replace with your actual filename

def main():
    print("Extracting weights from TensorFlow model...")
    
    # Create the original ML strategy
    ml_strategy = Soccer4v4MLStrategy()
    
    # Try to load the trained model
    if ml_strategy.load_model('soccer_4v4_model'):
        print("✅ Model loaded successfully")
        
        # Extract weights using TensorFlow
        import tensorflow as tf
        model = ml_strategy.model
        
        # Extract weights and biases
        weights = []
        biases = []
        
        for layer in model.layers:
            if hasattr(layer, 'get_weights') and len(layer.get_weights()) > 0:
                layer_weights = layer.get_weights()
                if len(layer_weights) == 2:  # Dense layer with weights and biases
                    weights.append(layer_weights[0])  # Weight matrix
                    biases.append(layer_weights[1])   # Bias vector
        
        # Save weights as NumPy format
        import numpy as np
        np.savez('soccer_weights.npz', 
                weights=weights, 
                biases=biases)
        
        print(f"✅ Extracted {len(weights)} weight matrices")
        print("✅ Saved to soccer_weights.npz")
        print("You can now remove TensorFlow dependency!")
        
    else:
        print("❌ Could not load model. Make sure soccer_4v4_model.keras exists")

if __name__ == "__main__":
    main()