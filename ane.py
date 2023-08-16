from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import numpy as np
from ane_transformers.huggingface import distilbert as ane_distilbert


# Can we get a transfomer to work on ane with coremltools?



model_checkpoint = "apple/ane-distilbert-base-uncased-finetuned-sst-2-english"
baseline_model = AutoModelForSequenceClassification.from_pretrained(
    model_checkpoint, return_dict=False, torchscript=True, trust_remote_code=True
).eval()


optimized_model = ane_distilbert.DistilBertForSequenceClassification(baseline_model.config).eval()
optimized_model.load_state_dict(baseline_model.state_dict())

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
tokenized = tokenizer(["This is a test"], return_tensors="pt", max_length=128)


with torch.no_grad():
    output = optimized_model(**tokenized)


print(output)



import coremltools as ct

mlmodel = ct.models.MLModel("weights/DistilBERT_fp16.mlpackage/Data/com.apple.CoreML/model.mlmodel")

input = tokenizer(
    ["This is a ANE test"],
    return_tensors="np",
    max_length=128,
    padding="max_length"
)

output_coreml = mlmodel.predict({
    "input_ids": input["input_ids"].astype(np.int32),
    "attention_mask": input["attention_mask"].astype(np.int32),
})
