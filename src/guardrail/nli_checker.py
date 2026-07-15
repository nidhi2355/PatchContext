import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class HallucinationGuard:
    def __init__(self, model_name: str = "MoritzLaurer/DeBERTa-v3-base-mnli"):
        print(f"Loading NLI Guardrail Model ({model_name})...")
        
        # Load the tokenizer and model. 
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        
        # Dynamically load the label mapping directly from the model's configuration
        # This ensures compatibility regardless of how the author indexed the labels
        self.label_map = self.model.config.id2label

    def verify_claim(self, premise: str, hypothesis: str) -> str:
        """
        Checks if the hypothesis (LLM answer) is logically supported by the premise (retrieved context).
        """
        # Tokenize the input pair. We truncate to ensure it fits the model's 512 token limit.
        inputs = self.tokenizer(
            text=premise, 
            text_pair=hypothesis, 
            return_tensors="pt", 
            truncation=True, 
            max_length=512
        )
        
        # Run the model without calculating gradients to save memory and time
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        # Extract the highest probability prediction
        prediction_idx = torch.argmax(outputs.logits, dim=-1).item()
        
        return self.label_map.get(prediction_idx, "UNKNOWN")

if __name__ == "__main__":
    guard = HallucinationGuard()
    
    print("\n--- Running NLI Verification Tests ---")
    
    # Simulated context from a software repository
    retrieved_context = "The JSONResponse class was introduced in PR #45 by the maintainer to optimize the serialization of dictionary objects, providing a 20% speedup over standard formatting."
    
    # Test 1: Supported Claim
    good_answer = "JSONResponse was added to optimize dictionary serialization, which improved speed."
    result_1 = guard.verify_claim(retrieved_context, good_answer)
    print(f"\nTest 1 (Factual): {good_answer}")
    print(f"Result: {result_1}")
    
    # Test 2: Hallucinated Claim
    bad_answer = "JSONResponse was created because the developers wanted to stop supporting XML entirely."
    result_2 = guard.verify_claim(retrieved_context, bad_answer)
    print(f"\nTest 2 (Hallucinated): {bad_answer}")
    print(f"Result: {result_2}")