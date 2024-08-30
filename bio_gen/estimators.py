from typing import Optional, Callable

from langchain.chains.base import Chain
from transformers import (
    AutoTokenizer,
    AutoModel,
)
import torch
from scipy.spatial.distance import cosine

from .consts import ENCODER_ID


class Estimator(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(ENCODER_ID)
        self.model = AutoModel.from_pretrained(ENCODER_ID)

        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = torch.device("cpu")
        print(f"Current device: {self.device}")
        self.model.to(self.device)

        self.model.eval()
        self.eval()

    @torch.no_grad()
    def compute_similarity(self, text_1: str, text_2: str) -> float:
        batch_tokens = self.tokenizer([text_1, text_2],
                                      padding=True,
                                      truncation=True,
                                      return_attention_mask=True,
                                      return_tensors="pt")
        batch_tokens.to(self.device)
        last_hidden_state = self.model(**batch_tokens,
                                       output_hidden_states=True,
                                       return_dict=True).last_hidden_state

        weights: torch.Tensor = (
            torch.arange(start=1, end=last_hidden_state.shape[1] + 1)
            .unsqueeze(0)
            .unsqueeze(-1)
            .expand(last_hidden_state.size())
            .float().to(last_hidden_state.device)
        )

        # Get attn mask of shape [bs, seq_len, hid_dim]
        input_mask_expanded = (
            batch_tokens["attention_mask"]
            .unsqueeze(-1)
            .expand(last_hidden_state.size())
            .float()
        )

        # Perform weighted mean pooling across seq_len: bs, seq_len, hidden_dim -> bs, hidden_dim
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded * weights, dim=1)
        sum_mask = torch.sum(input_mask_expanded * weights, dim=1)

        embeddings = sum_embeddings / sum_mask

        cosine_dist = cosine(embeddings[0].detach().cpu(), embeddings[1].detach().cpu())

        return torch.add(1, - cosine_dist).item()


class ImportanceEstimator(Estimator):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def __call__(self, text: str, summarized: str, *args, **kwargs) -> float:
        return self.compute_similarity(text, summarized)


class RedundancyEstimator(Estimator):
    def __init__(self, summarize_func: Callable, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.summarize_func = summarize_func

    def __call__(self, text: str, num_words: Optional[int] = 7, *args, **kwargs) -> float:
        self_summarized = self.summarize_func(text, num_words=num_words)

        return self.compute_similarity(text, self_summarized)


# TODO add elaborateness characteristic for a biography chunks
class ElaboratenessEstimator(Estimator): ...
