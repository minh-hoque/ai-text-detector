import os

if os.path.exists("/streamlit/app/module-ai-text-detector/"):
    cache_dir = "/streamlit/app/module-ai-text-detector/"
    os.environ["TRANSFORMERS_CACHE"] = cache_dir
else:
    cache_dir = None

from typing import Any, Dict, List
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import streamlit as st


class LanguageModel:
    def __init__(self, model_name_or_path="gpt2-large"):
        super(LanguageModel, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder = AutoTokenizer.from_pretrained(model_name_or_path, cache_dir=cache_dir)
        self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path, cache_dir=cache_dir)
        print(self.device)
        self.model.to(self.device)
        self.model.eval()
        self.start_token = self.encoder(self.encoder.bos_token, return_tensors="pt").data["input_ids"][0]
        self.topk = self.model.config.task_specific_params["text-generation"].get("top_k", None)
        print(f"Loaded {model_name_or_path} model!")

    def check_probabilities(self, in_text, default_topk=50):

        # Set topk
        if self.topk is not None:
            topk = self.topk
        else:
            topk = default_topk

        # Process input
        token_ids = self.encoder(in_text, return_tensors="pt").data["input_ids"][0]
        token_ids = torch.concat([self.start_token, token_ids])

        # Forward through the model
        output = self.model(token_ids.to(self.device))
        all_logits = output.logits[:-1].detach().squeeze()

        # Get all token probabilities
        all_probs = torch.softmax(all_logits, dim=1)

        # Remove the first token (bos)
        y = token_ids[1:]

        # Sort the predictions for each timestep
        sorted_preds = torch.argsort(all_probs, dim=1, descending=True).cpu()

        # Get the real token positions
        real_topk_pos = list([int(np.where(sorted_preds[i] == y[i].item())[0][0]) for i in range(y.shape[0])])

        # Get the real token probabilities
        real_topk_probs = all_probs[np.arange(0, y.shape[0], 1), y].data.cpu().numpy().tolist()

        # Round the probabilities
        real_topk_probs = list(map(lambda x: round(x, 5), real_topk_probs))

        # [(token_idx, prob), ...]
        real_topk = list(zip(real_topk_pos, real_topk_probs))

        # Create a list of the real BPE tokens [str, str, ...]
        bpe_strings = self.encoder.convert_ids_to_tokens(token_ids[:])

        # Postprocess the BPE tokens
        bpe_strings = [self.postprocess(s) for s in bpe_strings]

        # Get the topk predictions for each timestep. The shape of both tensors is (input_token_length, topk)
        topk_prob_values, topk_prob_idxs = torch.topk(all_probs, k=topk, dim=1)

        # Create a nested list. Each inner list contains the topk idxs and probs [(token_idx, prob), ...]
        # Shape of pred_topk is (input_token_length, topk, 2)
        pred_topk = [
            list(
                zip(
                    self.encoder.convert_ids_to_tokens(topk_prob_idxs[i]),
                    topk_prob_values[i].data.cpu().numpy().tolist(),
                )
            )
            for i in range(y.shape[0])
        ]
        pred_topk = [[(self.postprocess(t[0]), t[1]) for t in pred] for pred in pred_topk]

        payload = {"bpe_strings": bpe_strings, "real_topk": real_topk, "pred_topk": pred_topk}

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return payload

    def postprocess(self, token):
        with_space = False
        with_break = False
        if token.startswith("Ġ"):
            with_space = True
            token = token[1:]
            # print(token)
        elif token.startswith("â"):
            token = " "
        elif token.startswith("Ċ"):
            token = " "
            with_break = True

        token = "-" if token.startswith("â") else token
        token = "“" if token.startswith("ľ") else token
        token = "”" if token.startswith("Ŀ") else token
        token = "'" if token.startswith("Ļ") else token

        if with_space:
            token = "\u0120" + token
        if with_break:
            token = "\u010A" + token

        return token

    def check_if_ai_generated(self, text: str):

        FOUND = False
        found_in_topk = 0
        true_probs = []

        payload = self.check_probabilities(text)
        real_topk_tuple, pred_topk_tuple, bpe_strings = (
            payload["real_topk"],
            payload["pred_topk"],
            payload["bpe_strings"],
        )

        bpe_strings = bpe_strings[1:]
        pred_length = len(bpe_strings)

        # Start index is the middle of the input text. We assumed first half is the prompt.
        start_idx = -pred_length // 2

        # Get the real probabilities for each token. Shape is (input_token_length,1)
        real_probs = np.array(real_topk_tuple)[:, 1]

        # Get max probability for each token prediction. Shape is (input_token_length,1)
        pred_max_probs = np.array(pred_topk_tuple)[:, :, 1].astype(float).max(axis=1)

        assert real_probs.shape == pred_max_probs.shape, "real_prob and pred_prob should have the same shape"
        label_score = (real_probs[start_idx:] / pred_max_probs[start_idx:]).mean()

        # Iterate over the next true characters
        for idx, bpe_string in enumerate(bpe_strings[start_idx:]):

            # Iterate over the topk predictions. Pred is a tuple (token, prob)
            for pred in pred_topk_tuple[start_idx + idx]:
                # If character in topk predictions, save the probability
                if bpe_string in pred:
                    found_in_topk += 1
                    FOUND = True
                    break

        # Total number of tokens in input text
        tokens_len = -1 * start_idx

        # Normalize the number of found tokens in topk by the total number of tokens
        found_in_topk_normalized = found_in_topk / tokens_len

        # Calculate the detection score
        detection_score = found_in_topk_normalized * label_score

        # If the detection score is greater than 0.6, the text is AI generated
        ai_generated_bool = detection_score > 0.6

        # st.write(f"Input token length: {pred_length}")
        # st.write(f"Found in topk: {found_in_topk}")
        # st.write(f"found_in_topk_normalized: {found_in_topk_normalized:.2f}")
        st.subheader(f"Detection Score")
        st.progress(detection_score)
        if ai_generated_bool:
            st.error(f"AI Generated: {ai_generated_bool}")
        else:
            st.info(f"AI Generated: {ai_generated_bool}")
