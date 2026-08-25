"""Main MolecularLLM (Q-Former variant): MolCA GIN + BERT Q-Former + Llama LLM."""

import contextlib
import os
import torch
import torch.nn as nn
from typing import Dict, List, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType

from .qformer_projector import QFormerProjector
from .encoder_wrapper import MolCAGraphEncoder

from configs.model_config import MOL_TOKEN_START, MOL_TOKEN_EMBED, MOL_TOKEN_END


class MolecularLLM(nn.Module):
    """Multimodal Molecular LLM (Q-Former variant).

    Architecture:
        1. MolCA fine-tuned GIN graph encoder (300D node embeddings)
        2. BERT Q-Former projector (8 query tokens -> llm_dim via cross-attention)
        3. Llama-3.1-8B-Instruct (frozen by default; LoRA optional)

    The MolCA checkpoint seeds `graph_encoder + Qformer + query_tokens + ln_graph`
    once at Stage 1 init. `llm_proj` stays random-init.
    """

    def __init__(
        self,
        llm_name: str = "meta-llama/Llama-3.1-8B-Instruct",
        graph_dim: int = 300,
        llm_dim: int = 4096,
        num_query_tokens: int = 8,
        molca_ckpt_path: Optional[str] = None,
        dropout: float = 0.1,
        use_lora: bool = False,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
        lora_target_modules: List[str] = None,
        freeze_graph_encoder: bool = False,
        freeze_projector: bool = False,
        device: str = "cuda",
    ):
        super().__init__()

        self.device = device
        self.llm_name = llm_name
        self.llm_dim = llm_dim
        self.graph_dim = graph_dim
        self.num_mol_tokens = num_query_tokens
        self.use_lora = use_lora
        self.freeze_graph_encoder = freeze_graph_encoder
        self.freeze_projector = freeze_projector

        self.tokenizer = AutoTokenizer.from_pretrained(llm_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.mol_token_start = MOL_TOKEN_START
        self.mol_token_embed = MOL_TOKEN_EMBED
        self.mol_token_end = MOL_TOKEN_END
        self.mol_token = MOL_TOKEN_EMBED

        self.tokenizer.add_special_tokens({
            "additional_special_tokens": [
                self.mol_token_start, self.mol_token_embed, self.mol_token_end,
            ]
        })
        self.mol_token_start_id = self.tokenizer.convert_tokens_to_ids(self.mol_token_start)
        self.mol_token_embed_id = self.tokenizer.convert_tokens_to_ids(self.mol_token_embed)
        self.mol_token_end_id = self.tokenizer.convert_tokens_to_ids(self.mol_token_end)
        self.mol_token_id = self.mol_token_embed_id

        self._load_llm(llm_name)

        for param in self.llm.parameters():
            param.requires_grad = False

        if use_lora:
            self._apply_lora(lora_r, lora_alpha, lora_dropout, lora_target_modules)

        self.graph_encoder = MolCAGraphEncoder(freeze=freeze_graph_encoder, device=device)

        self.projector = QFormerProjector(
            graph_dim=graph_dim,
            llm_dim=llm_dim,
            num_query_tokens=num_query_tokens,
        ).to(device)

        if molca_ckpt_path is not None:
            self._load_molca(molca_ckpt_path)

        if freeze_projector:
            for param in self.projector.parameters():
                param.requires_grad = False

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_graph_encoder:
            self.graph_encoder.eval()
        if self.freeze_projector:
            self.projector.eval()
        return self

    def _load_llm(self, llm_name: str):
        self.llm = AutoModelForCausalLM.from_pretrained(
            llm_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="sdpa",
        )
        self.llm.resize_token_embeddings(len(self.tokenizer), mean_resizing=False)
        self.llm.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

    def _apply_lora(self, r, alpha, dropout, target_modules):
        if target_modules is None:
            target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=r,
            lora_alpha=alpha,
            lora_dropout=dropout,
            target_modules=target_modules,
            bias="none",
        )
        self.llm = get_peft_model(self.llm, lora_config)

    def _load_molca(self, molca_ckpt_path: str):
        """Load graph_encoder + Qformer + query_tokens + ln_graph from MolCA stage1.ckpt.

        Detected MolCA top-level prefix: `blip2qformer.`
        """
        if not os.path.exists(molca_ckpt_path):
            raise FileNotFoundError(f"MolCA checkpoint not found at {molca_ckpt_path}")

        print(f"[MolecularLLM] Loading MolCA checkpoint: {molca_ckpt_path}")
        raw = torch.load(molca_ckpt_path, map_location="cpu", weights_only=False)
        state_dict = raw.get("state_dict", raw) if isinstance(raw, dict) else raw

        prefixes = {k.split(".", 1)[0] for k in state_dict.keys()}
        if len(prefixes) == 1:
            prefix = next(iter(prefixes)) + "."
            stripped = {k[len(prefix):]: v for k, v in state_dict.items()}
            print(f"[MolecularLLM] Detected MolCA top-level prefix: '{prefix}'")
        else:
            stripped = dict(state_dict)
            print(f"[MolecularLLM] Multiple top-level prefixes {prefixes}; using raw keys")

        self.graph_encoder.load_molca_graph_state(stripped)
        self.projector.load_molca_weights(stripped)

    def encode_molecule(self, graphs: Dict[str, torch.Tensor]) -> torch.Tensor:
        edge_attr = graphs.get("edge_attr")
        if edge_attr is not None:
            edge_attr = edge_attr.to(self.device)

        graph_ctx = torch.no_grad() if self.freeze_graph_encoder else contextlib.nullcontext()
        with graph_ctx:
            graph_output, _ = self.graph_encoder(
                x=graphs["x"].to(self.device),
                edge_index=graphs["edge_index"].to(self.device),
                edge_attr=edge_attr,
                batch=graphs["batch"].to(self.device),
            )

        batch_device = graphs["batch"].to(self.device)
        batch_size = batch_device.max().item() + 1
        graph_output = self._reshape_graph_output(graph_output, batch_device)
        graph_mask = self._create_graph_mask(batch_device, batch_size)

        mol_tokens = self.projector(
            graph_node_feats=graph_output,
            graph_node_mask=graph_mask,
        )
        return mol_tokens

    def _reshape_graph_output(self, graph_output, batch):
        batch_size = batch.max().item() + 1
        max_nodes = 0
        for i in range(batch_size):
            num_nodes = (batch == i).sum().item()
            max_nodes = max(max_nodes, num_nodes)
        output = torch.zeros(
            batch_size, max_nodes, graph_output.shape[-1],
            device=graph_output.device, dtype=graph_output.dtype
        )
        for i in range(batch_size):
            mask = batch == i
            nodes = graph_output[mask]
            output[i, :nodes.shape[0]] = nodes
        return output

    def _create_graph_mask(self, batch, batch_size):
        max_nodes = 0
        for i in range(batch_size):
            num_nodes = (batch == i).sum().item()
            max_nodes = max(max_nodes, num_nodes)
        mask = torch.zeros(batch_size, max_nodes, device=batch.device)
        for i in range(batch_size):
            num_nodes = (batch == i).sum().item()
            mask[i, :num_nodes] = 1
        return mask

    def forward(self, input_ids, attention_mask, labels=None, graphs=None):
        batch_size = input_ids.shape[0]
        mol_tokens = self.encode_molecule(graphs) if graphs is not None else None

        text_embeds = self.llm.get_input_embeddings()(input_ids)

        if mol_tokens is not None:
            mol_token_mask = input_ids == self.mol_token_id
            combined_embeds = text_embeds.clone()
            mol_tokens_llm_dtype = mol_tokens.to(combined_embeds.dtype)
            for i in range(batch_size):
                mol_positions = mol_token_mask[i].nonzero(as_tuple=True)[0]
                if len(mol_positions) > 0:
                    start_pos = mol_positions[0].item()
                    end_pos = min(start_pos + self.num_mol_tokens, combined_embeds.shape[1])
                    num_tokens = end_pos - start_pos
                    combined_embeds[i, start_pos:end_pos] = mol_tokens_llm_dtype[i, :num_tokens]
        else:
            combined_embeds = text_embeds

        outputs = self.llm(
            inputs_embeds=combined_embeds,
            attention_mask=attention_mask,
            labels=labels,
            use_cache=False,
            return_dict=True,
        )
        result = {"logits": outputs.logits}
        if labels is not None:
            result["loss"] = outputs.loss
        return result

    def generate(
        self,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
        graphs=None,
        input_ids=None,
        attention_mask=None,
    ):
        mol_tokens = self.encode_molecule(graphs) if graphs is not None else None

        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        text_embeds = self.llm.get_input_embeddings()(input_ids)

        if mol_tokens is not None:
            batch_size = input_ids.shape[0]
            mol_token_mask = input_ids == self.mol_token_id
            combined_embeds = text_embeds.clone()
            mol_tokens_llm_dtype = mol_tokens.to(combined_embeds.dtype)
            for i in range(batch_size):
                mol_positions = mol_token_mask[i].nonzero(as_tuple=True)[0]
                if len(mol_positions) > 0:
                    start_pos = mol_positions[0].item()
                    end_pos = min(start_pos + self.num_mol_tokens, combined_embeds.shape[1])
                    num_tokens = end_pos - start_pos
                    combined_embeds[i, start_pos:end_pos] = mol_tokens_llm_dtype[i, :num_tokens]
        else:
            combined_embeds = text_embeds

        outputs = self.llm.generate(
            inputs_embeds=combined_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

    def get_trainable_parameters(self):
        params = []
        if not self.freeze_projector:
            params.extend(p for p in self.projector.parameters() if p.requires_grad)
        if not self.freeze_graph_encoder:
            params.extend(p for p in self.graph_encoder.parameters() if p.requires_grad)
        if self.use_lora:
            for name, param in self.llm.named_parameters():
                if "lora" in name.lower():
                    params.append(param)
        return params

    def save_pretrained(self, save_path: str):
        os.makedirs(save_path, exist_ok=True)
        save_dict = {
            "projector": self.projector.state_dict(),
            "graph_encoder": self.graph_encoder.state_dict(),
        }
        torch.save(save_dict, os.path.join(save_path, "trainable_components.pt"))

        if self.use_lora:
            self.llm.save_pretrained(os.path.join(save_path, "lora"))

        self.tokenizer.save_pretrained(save_path)

    def load_pretrained(self, load_path: str):
        ckpt = torch.load(
            os.path.join(load_path, "trainable_components.pt"),
            map_location=self.device,
        )
        self.projector.load_state_dict(ckpt["projector"])
        if ckpt.get("graph_encoder") is not None:
            self.graph_encoder.load_state_dict(ckpt["graph_encoder"])

        lora_path = os.path.join(load_path, "lora")
        if os.path.exists(lora_path) and self.use_lora:
            from peft import PeftModel
            self.llm = PeftModel.from_pretrained(self.llm, lora_path)
