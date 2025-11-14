from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch


class LocalLLM:
    def __init__(self, model_name='microsoft/phi-2'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model with lower precision to save memory
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        )

        # Create pipeline with proper generation config
        self.pipe = pipeline(
            'text-generation',
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.7,
            pad_token_id=self.tokenizer.eos_token_id
        )

    def generate(self, prompt: str) -> str:
        response = self.pipe(prompt, max_new_tokens=256, temperature=0.7)
        return response[0]['generated_text']
