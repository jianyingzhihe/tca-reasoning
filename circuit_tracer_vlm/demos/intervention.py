
import sys
sys.path.append("/work/nvme/bewu/jyang28/vlm-tracing/third_party/TransformerLens")

from collections import namedtuple
from functools import partial

import torch 

from circuit_tracer import ReplacementModel

# display functions
from utils import display_topk_token_predictions, display_generations_comparison, print_topk_token_predictions
from utils import *

from circuit_tracer.utils.hf_utils import load_transcoder_from_hub

transcoder, config = load_transcoder_from_hub(
        "tianhux2/lichang1",
        dtype=torch.bfloat16,
    )

model = ReplacementModel.from_pretrained_and_transcoders("google/gemma-3-4b-it", transcoder, dtype=torch.bfloat16)

Feature = namedtuple('Feature', ['layer', 'pos', 'feature_idx'])

# a display function that needs the model's tokenizer
display_topk_token_predictions = partial(display_topk_token_predictions, tokenizer=model.set_tokenizer)

suppress_features = [
    Feature(layer=24, pos=-1, feature_idx=155477),
]

# --- features to amplify (set to a big positive value)
amplify_features = [
    Feature(layer=28, pos=-1, feature_idx=53763),
    Feature(layer=24, pos=-1, feature_idx=91192),
]

# build intervention tuples directly
intervention_tuples = []

# suppress
intervention_tuples += [(f.layer, f.pos, f.feature_idx, 0.0) for f in suppress_features]

# amplify — pick an arbitrary strong target value (try 8.0–12.0)
BOOST = 8.0
intervention_tuples += [(f.layer, f.pos, f.feature_idx, BOOST) for f in amplify_features]

from PIL import Image
image = Image.open("/work/nvme/bewu/jyang28/vlm-tracing/images/mars.png")
prompt = "<start_of_image> This is planet"

# Build batch so we can pick the last *text* position (ignoring image tokens)
batch = model.processor(text=prompt, images=image, return_tensors="pt")

with torch.inference_mode():
    original_logits = model([prompt], [[image]])           # [1,T,V]
    new_logits, _ = model.feature_intervention((prompt, image), intervention_tuples)

'''
print_topk_token_predictions(
    prompt,
    original_logits,
    new_logits,
    tokenizer=model.processor.tokenizer,
    k=5,
    input_ids=batch["input_ids"][0],                       # lets us auto-pick last text token
    # position=...                                         # (optional) override position explicitly
)'''

print_topk_token_predictions(prompt, original_logits, new_logits, model.processor.tokenizer)