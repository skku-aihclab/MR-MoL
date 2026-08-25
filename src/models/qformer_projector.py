"""BERT Q-Former projector, checkpoint-compatible with MolCA.

Our code. The architecture mirrors MolCA (Liu et al., EMNLP 2023) closely enough
to load its released Stage-1 weights:
    - SciBERT backbone with cross-attention every `cross_attention_freq` layers
    - `num_query_tokens` learnable query embeddings
    - LayerNorm on graph node features (matches MolCA `ln_graph`)
    - Linear `llm_proj` from Q-Former hidden dim (768) to LLM hidden dim (4096)

The module is designed so that the `Qformer + query_tokens + ln_graph` parameters
can be initialized from MolCA's `stage1.ckpt`, while `llm_proj` stays random-init.
"""

from typing import Optional, Dict

import torch
from torch import nn

from .lavis_qformer import BertConfig, BertLMHeadModel


SCIBERT_NAME = "allenai/scibert_scivocab_uncased"


_STRIPPED_KEY_PREFIXES = (
    "Qformer.cls.",
    "Qformer.bert.embeddings.word_embeddings.",
    "Qformer.bert.embeddings.position_embeddings.",
)


def _is_stripped_layer_ffn_key(key: str) -> bool:
    if not key.startswith("Qformer.bert.encoder.layer."):
        return False
    parts = key.split(".", 6)
    if len(parts) < 6:
        return False
    sub = parts[5]
    return sub in ("intermediate", "output")


class QFormerProjector(nn.Module):
    def __init__(
        self,
        graph_dim: int = 300,
        llm_dim: int = 4096,
        num_query_tokens: int = 8,
        cross_attention_freq: int = 2,
    ):
        super().__init__()
        self.num_query_tokens = num_query_tokens

        self.ln_graph = nn.LayerNorm(graph_dim)

        encoder_config = BertConfig.from_pretrained(SCIBERT_NAME)
        encoder_config.encoder_width = graph_dim
        encoder_config.add_cross_attention = True
        encoder_config.cross_attention_freq = cross_attention_freq
        encoder_config.query_length = num_query_tokens

        self.Qformer = BertLMHeadModel.from_pretrained(
            SCIBERT_NAME, config=encoder_config
        )

        state_dict = self.Qformer.state_dict()
        for name, param in self.Qformer.named_parameters():
            if "_query" in name:
                key_orig = name.replace("_query", "")
                param.data.copy_(state_dict[key_orig])

        self.Qformer.cls = None
        self.Qformer.bert.embeddings.word_embeddings = None
        self.Qformer.bert.embeddings.position_embeddings = None
        for layer in self.Qformer.bert.encoder.layer:
            layer.intermediate = None
            layer.output = None

        self.query_tokens = nn.Parameter(
            torch.zeros(1, num_query_tokens, encoder_config.hidden_size)
        )
        self.query_tokens.data.normal_(mean=0.0, std=encoder_config.initializer_range)

        self.llm_proj = nn.Linear(encoder_config.hidden_size, llm_dim)

    def forward(
        self,
        graph_node_feats: torch.Tensor,
        graph_node_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            graph_node_feats: (B, N, graph_dim) node embeddings
            graph_node_mask:  (B, N) 1 for real nodes, 0 for padding
        Returns:
            (B, num_query_tokens, llm_dim) — projector output for LLM
        """
        graph_embeds = self.ln_graph(graph_node_feats)
        if graph_node_mask is None:
            graph_node_mask = torch.ones(
                graph_embeds.size()[:-1],
                dtype=torch.long,
                device=graph_embeds.device,
            )

        query_tokens = self.query_tokens.expand(graph_embeds.size(0), -1, -1)

        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=graph_embeds,
            encoder_attention_mask=graph_node_mask,
            return_dict=True,
        )
        projected = self.llm_proj(query_output.last_hidden_state)
        return projected

    def load_molca_weights(self, stripped_state_dict: Dict[str, torch.Tensor]) -> None:
        """Load Q-Former + query_tokens + ln_graph weights from a MolCA checkpoint.

        `stripped_state_dict` must already have the MolCA top-level prefix removed
        (so keys look like `Qformer.*`, `query_tokens`, `ln_graph.*`).
        """
        namespace_filtered = {
            k: v
            for k, v in stripped_state_dict.items()
            if k.startswith("Qformer.") or k == "query_tokens" or k.startswith("ln_graph.")
        }
        if not namespace_filtered:
            raise RuntimeError(
                "load_molca_weights: no keys matched Qformer./query_tokens/ln_graph. "
                "Check the checkpoint prefix detection."
            )

        filtered = {
            k: v
            for k, v in namespace_filtered.items()
            if not k.startswith(_STRIPPED_KEY_PREFIXES)
            and not _is_stripped_layer_ffn_key(k)
        }
        dropped = len(namespace_filtered) - len(filtered)

        missing, unexpected = self.load_state_dict(filtered, strict=False)
        expected_missing_prefix = "llm_proj."
        loaded = len(filtered) - len(unexpected)
        print(
            f"[QFormerProjector] loaded {loaded}/{len(filtered)} MolCA keys "
            f"(dropped_stripped={dropped}, unexpected={len(unexpected)}, missing={len(missing)})"
        )
        for k in missing:
            if not k.startswith(expected_missing_prefix):
                print(f"  [missing non-llm_proj key] {k}")
        for k in unexpected[:10]:
            print(f"  [unexpected] {k}")
