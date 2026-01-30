"""Main Sharingan analyzer class."""

from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from sharingan.core.config import SharinganConfig
from sharingan.core.result import AttentionResult
from sharingan.models.registry import get_adapter
from sharingan.attention.extractor import extract_attention


class Sharingan:
    """Main class for attention visualization and analysis.

    Args:
        model_name: HuggingFace model name or path
        config: Optional SharinganConfig instance

    Example:
        >>> analyzer = Sharingan("Qwen/Qwen3-0.6B")
        >>> result = analyzer.analyze("The capital of France is")
        >>> result.plot()
    """

    def __init__(
        self,
        model_name: str,
        config: SharinganConfig | None = None,
    ):
        self.model_name = model_name
        self.config = config or SharinganConfig()
        self.device = self.config.resolve_device()
        self.dtype = self.config.resolve_dtype()

        self.model = None
        self.tokenizer = None
        self.adapter = None
        self._loaded = False

    def load(self) -> "Sharingan":
        """Load model and tokenizer.

        Returns:
            Self for chaining
        """
        if self._loaded:
            return self

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            cache_dir=self.config.cache_dir,
            trust_remote_code=True,
        )

        # Load model with eager attention (required for attention weights)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            cache_dir=self.config.cache_dir,
            torch_dtype=self.dtype,
            device_map=self.device if self.device != "cpu" else None,
            attn_implementation="eager",  # Required for attention weights
            trust_remote_code=True,
        )

        if self.device == "cpu":
            self.model = self.model.to(self.device)

        self.model.eval()

        # Get model adapter for handling architecture-specific details
        self.adapter = get_adapter(self.model, self.model_name)

        self._loaded = True
        return self

    def analyze(
        self,
        prompt: str,
        *,
        generate: bool = False,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        do_sample: bool = False,
    ) -> AttentionResult:
        """Analyze attention patterns for a prompt.

        Args:
            prompt: Input text to analyze
            generate: Whether to generate new tokens
            max_new_tokens: Maximum tokens to generate if generate=True
            temperature: Sampling temperature for generation
            do_sample: Whether to use sampling for generation

        Returns:
            AttentionResult with attention data and analysis methods
        """
        self.load()

        # Tokenize input
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            return_attention_mask=True,
        ).to(self.device)

        prompt_length = inputs.input_ids.shape[1]
        generated_length = 0

        if generate:
            # Generate with attention output
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=do_sample,
                    output_attentions=True,
                    return_dict_in_generate=True,
                )

            # Get final sequence for attention extraction
            final_ids = outputs.sequences
            generated_length = final_ids.shape[1] - prompt_length

            # Re-run forward pass on full sequence to get attention
            inputs = {"input_ids": final_ids, "attention_mask": torch.ones_like(final_ids)}

        # Extract attention weights
        attention = extract_attention(
            self.model,
            inputs,
            adapter=self.adapter,
            offload_to_cpu=self.config.offload_to_cpu,
        )

        # Get tokens for labeling
        token_ids = inputs["input_ids"][0].tolist()
        tokens = [self.tokenizer.decode([tid]) for tid in token_ids]

        return AttentionResult(
            attention=attention,
            tokens=tokens,
            prompt_length=prompt_length,
            generated_length=generated_length,
            model_name=self.model_name,
            config={
                "device": self.device,
                "dtype": str(self.dtype),
            },
        )

    def analyze_batch(
        self,
        prompts: list[str],
        **kwargs,
    ) -> list[AttentionResult]:
        """Analyze multiple prompts.

        Args:
            prompts: List of input texts
            **kwargs: Arguments passed to analyze()

        Returns:
            List of AttentionResult objects
        """
        return [self.analyze(prompt, **kwargs) for prompt in prompts]

    def __repr__(self) -> str:
        status = "loaded" if self._loaded else "not loaded"
        return f"Sharingan(model='{self.model_name}', {status})"
