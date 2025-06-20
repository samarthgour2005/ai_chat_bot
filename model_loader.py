import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelLoader:
    def __init__(self, model_name="google/flan-t5-base", max_length=256):
        self.model_name = model_name
        self.max_length = max_length
        self.tokenizer = None
        self.model = None
        self.generator = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Initialized ModelLoader with device: {self.device}")

    def load_model(self):
        try:
            logger.info(f"Loading model: {self.model_name}")

            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None
            )

            self.generator = pipeline(
                "text2text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if self.device == "cuda" else -1
            )

            logger.info("Model loaded successfully!")
            return True

        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return False

    def generate_response(self, user_input):
        if not self.generator:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        try:
            prompt = f"Answer this question: {user_input}"
            outputs = self.generator(
                prompt,
                max_new_tokens=128,
                num_return_sequences=1
            )
            response = outputs[0]['generated_text'].strip()
            return self._clean_response(response)
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return "Sorry, I encountered an error while generating a response."

    def _clean_response(self, response):
        response = response.strip()
        for sep in ['\n', '<|endoftext|>', '</s>', '<|im_end|>']:
            if sep in response:
                response = response.split(sep)[0].strip()
        if not response:
            response = "I'm not sure how to answer that."
        return response

    def get_model_info(self):
        return {
            "model_name": self.model_name,
            "device": self.device,
            "max_length": self.max_length,
            "loaded": self.model is not None
        }

RECOMMENDED_MODELS = [
    "google/flan-t5-base",
    "google/flan-t5-large",
    "mistralai/Mistral-7B-Instruct-v0.1",
    "tiiuae/falcon-7b-instruct"
]

def get_recommended_model():
    return RECOMMENDED_MODELS[0]
