from typing import Tuple

class CostCalculator:
    """Utility class for calculating token counts and costs for LLM API usage."""
    
    # Cost constants
    COST_PER_1000_TOKENS_GEMINI_BASE = 0.01
    COST_PER_1000_TOKENS_GEMINI_PRO = 0.05
    
    @staticmethod
    def estimate_tokens(text: str) -> int:
        """
        Provides a rough estimation of token count for a given text.
        
        Args:
            text: Input text to estimate tokens for
            
        Returns:
            int: Estimated number of tokens
            
        Note:
            This is a rough approximation assuming 1 token ≈ 4 characters for English text.
            For accurate token counts, use the model's actual tokenizer.
        """
        return len(text) // 4
    
    @classmethod
    def calculate_cost(
        cls,
        model: str,
        input_text: str,
        output_text: str
    ) -> Tuple[float, int]:
        """
        Calculates estimated cost and token count for model usage.
        
        Args:
            model: Model name ('gemini-base' or 'gemini-pro')
            input_text: Input prompt text
            output_text: Generated output text
            
        Returns:
            Tuple[float, int]: (Estimated cost in USD, Total token count)
        """
        input_tokens = cls.estimate_tokens(input_text)
        output_tokens = cls.estimate_tokens(output_text)
        total_tokens = input_tokens + output_tokens
        
        if model == "gemini-base":
            cost = (total_tokens / 1000) * cls.COST_PER_1000_TOKENS_GEMINI_BASE
        elif model == "gemini-pro":
            cost = (total_tokens / 1000) * cls.COST_PER_1000_TOKENS_GEMINI_PRO
        else:
            cost = 0
            
        return cost, total_tokens
