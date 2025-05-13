# Load model directly
from transformers import AutoTokenizer, ViViTForVideoClassification

tokenizer = AutoTokenizer.from_pretrained("google/vivit-b-16x2-kinetics400")
model = ViViTForVideoClassification.from_pretrained("google/vivit-b-16x2-kinetics400")

