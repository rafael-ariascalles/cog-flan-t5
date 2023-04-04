from typing import List, Optional
from huggingface_hub import HfFolder
from cog import BasePredictor, Input
import torch
from peft import PeftModel, PeftConfig
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


CACHE_DIR = 'weights'
SEP = "<sep>"

class Predictor(BasePredictor):
    def setup(self):

        HfFolder().save_token(token="")

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        peft_model_id = "rjac/flan-t5-xxl-senza-LoRA-qa"
        config = PeftConfig.from_pretrained(peft_model_id)
        model = AutoModelForSeq2SeqLM.from_pretrained(config.base_model_name_or_path,torch_dtype=torch.float16, load_in_8bit=True,  device_map={'':0})
        self.tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path,device_map={'':0})
        self.model = PeftModel.from_pretrained(model, peft_model_id,device_map={'':0})
        #print(torch.__version__)
        

    def predict(
        self,
        prompt: str = Input(description=f"Prompt to send to FLAN-T5."),
        n: int = Input(description="Number of output sequences to generate", default=1, ge=1, le=5),
        max_length: int = Input(
            description="Maximum number of tokens to generate. A word is generally 2-3 tokens",
            ge=1,
            default=350
        ),
        temperature: float = Input(
            description="Adjusts randomness of outputs, greater than 1 is random and 0 is deterministic, 0.75 is a good starting value.",
            ge=0.01,
            le=5,
            default=0.75,
        ),
        ) -> List[str]:

        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")

        outputs = self.model.generate(
            input_ids=input_ids,
            num_return_sequences=n,
            max_length=max_length,
            do_sample=True,
            temperature=temperature
        )
        out = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return out
        