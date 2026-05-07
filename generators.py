"""Answer generation backends for local Transformers and OpenAI."""

from __future__ import annotations

import os
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from settings import GenerationSettings

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover
    OpenAI = None

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
except ImportError:  # pragma: no cover
    AutoModelForCausalLM = None
    AutoTokenizer = None
    pipeline = None


class BaseGenerator(ABC):
    """Abstract answer generation interface."""

    @abstractmethod
    def generate_answer(self, prompt: str) -> str:
        """
        Generates an answer for a RAG prompt.

        @param prompt: Prompt string.
        @return: Generated answer text.
        """
        raise NotImplementedError


class OpenAIGenerator(BaseGenerator):
    """OpenAI API answer generator."""

    def __init__(self, settings: GenerationSettings) -> None:
        """
        Initializes the OpenAI generator.

        @param settings: Generation settings.
        @return: None.
        @raises ImportError: If the openai package is not installed.
        """

        if OpenAI is None:
            raise ImportError("openai package is required for the OpenAI backend.")

        load_dotenv(override=True)

        self.model_name = settings.openai_model
        self.temperature = settings.temperature

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY was not found. Check your .env file.")


        self.client = OpenAI(api_key=api_key)

    def generate_answer(self, prompt: str) -> str:
        """
        Generates an answer using the OpenAI API.

        @param prompt: Prompt string.
        @return: Generated answer text.
        """

        response = self.client.responses.create(
            model=self.model_name,
            input=prompt,
            temperature=self.temperature,
        )
        return response.output_text.strip()


class LocalTransformersGenerator(BaseGenerator):
    """Local Transformers generator for gpt-oss models."""

    def __init__(self, settings: GenerationSettings) -> None:
        """
        Initializes the local Transformers generator.

        @param settings: Generation settings.
        @return: None.
        @raises ImportError: If transformers is not installed.
        """

        if AutoModelForCausalLM is None or AutoTokenizer is None or pipeline is None:
            raise ImportError("transformers is required for the local generator.")

        self.model_id = settings.local_model_id
        self.system_prompt = settings.local_system_prompt
        self.max_new_tokens = settings.max_new_tokens
        self.temperature = settings.temperature
        self.device_map = settings.local_device_map
        self.torch_dtype = self._resolve_torch_dtype(settings.local_torch_dtype)

        self.offload_folder = Path("artifacts") / "offload"
        self.offload_folder.mkdir(parents=True, exist_ok=True)

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            trust_remote_code=True,
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=self.torch_dtype,
            device_map=self.device_map,
            offload_folder=str(self.offload_folder),
            trust_remote_code=True,
        )

        self.pipe = pipeline(
            task="text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
        )

    @staticmethod
    def _resolve_torch_dtype(dtype_name: str) -> Any:
        """
        Converts the configured dtype string into a torch dtype when possible.

        @param dtype_name: Configured dtype name such as "auto", "float16", or "bfloat16".
        @return: Torch dtype object or the string "auto".
        """

        if dtype_name == "auto":
            return "auto"

        if torch is not None and hasattr(torch, dtype_name):
            return getattr(torch, dtype_name)

        return "auto"

    @staticmethod
    def _extract_text_from_generated_output(generated: Any) -> str:
        """
        Extracts text from the Transformers pipeline output.

        @param generated: Generated output payload.
        @return: Extracted text.
        """

        if isinstance(generated, list) and len(generated) > 0:
            last_item = generated[-1]
            if isinstance(last_item, dict):
                content = last_item.get("content", "")
                if isinstance(content, list):
                    parts: list[str] = []
                    for block in content:
                        if isinstance(block, dict):
                            text_value = block.get("text")
                            if text_value:
                                parts.append(str(text_value))
                    if parts:
                        return "\n".join(parts).strip()
                return str(content).strip()
            return str(last_item).strip()

        if isinstance(generated, dict):
            return str(generated.get("content", "")).strip()

        return str(generated).strip()

    @staticmethod
    def _clean_harmony_output(text: str) -> str:
        """
        Cleans harmony-style local model output.

        This removes visible internal sections like:
        - analysis
        - assistant
        - final
        - assistantfinal

        @param text: Raw generated text.
        @return: Cleaned final answer text.
        """

        cleaned = text.strip()

        lower_cleaned = cleaned.lower()
        marker = "assistantfinal"
        idx = lower_cleaned.find(marker)
        if idx != -1:
            cleaned = cleaned[idx + len(marker):].strip()

        cleaned = re.sub(
            r"^analysis.*?(?=assistantfinal|finalanswer:|answer:)",
            "",
            cleaned,
            flags=re.IGNORECASE | re.DOTALL,
        ).strip()

        cleaned = re.sub(r"\bassistantfinal\b", "", cleaned, flags=re.IGNORECASE).strip()
        cleaned = re.sub(r"\bfinalanswer:\b", "Answer:", cleaned, flags=re.IGNORECASE).strip()
        cleaned = re.sub(r"^(assistant|final)\s*", "", cleaned, flags=re.IGNORECASE).strip()

        return cleaned

    def generate_answer(self, prompt: str) -> str:
        """
        Generates an answer using the local Transformers backend.

        @param prompt: Prompt string.
        @return: Generated answer text.
        """

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt},
        ]

        generation_kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.temperature > 0,
            "return_full_text": False,
        }

        if self.temperature > 0:
            generation_kwargs["temperature"] = self.temperature

        outputs = self.pipe(
            messages,
            **generation_kwargs,
        )

        generated = outputs[0].get("generated_text", "")
        raw_text = self._extract_text_from_generated_output(generated)
        return self._clean_harmony_output(raw_text)


def build_generator(settings: GenerationSettings, backend: str | None = None) -> BaseGenerator:
    """
    Factory function for generator backends.

    @param settings: Generation settings.
    @param backend: Explicit backend override.
    @return: Generator instance.
    @raises ValueError: If the backend is unsupported.
    """

    selected = (backend or settings.default_backend).lower()

    if selected == "openai":
        return OpenAIGenerator(settings)

    if selected == "local":
        return LocalTransformersGenerator(settings)

    raise ValueError(f"Unsupported generator backend: {selected}")