import torch
from transformers import LlamaForCausalLM, LlamaTokenizer

from .lib_dantaggen.app import get_result


MODEL_PATHS = ["KBlueLeaf/DanTagGen-alpha", "KBlueLeaf/DanTagGen-beta"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class DanTagGen:
    """DanTagGen node."""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": (MODEL_PATHS,),
                "artist": ("STRING",),
                "characters": ("STRING",),
                "copyrights": ("STRING",),
                "special_tags": ("STRING",),
                "tag": ("STRING",),
                "blacklist": ("STRING",),
                "rating": (["safe", "sensitive", "nsfw", "nsfw, explicit"],),
                "target": (["very_short", "short", "long", "very_long"],),
                "width": (
                    "INT",
                    {"default": 1024, "min": 256, "max": 4096, "step": 32},
                ),
                "height": (
                    "INT",
                    {"default": 1024, "min": 256, "max": 4096, "step": 32},
                ),
                "escape_bracket": ("BOOLEAN", {"default": False}),
                "temperature": ("FLOAT", {"default": 1.35, "step": 0.05}),
            },
        }

    # Returns (final output, LLM output)
    RETURN_TYPES = ("STRING", "STRING")
    FUNCTION = "generate"
    CATEGORY = "prompt"

    def generate(
        self,
        model: str,
        rating: str,
        artist: str,
        characters: str,
        copyrights: str,
        target: str,
        special_tags: list[str],
        general: str,
        width: float,
        height: float,
        blacklist: str,
        escape_bracket: bool,
        temperature: float = 1.35,
    ):
        models = {
            model_path: [
                LlamaForCausalLM.from_pretrained(
                    model_path, attn_implementation="flash_attention_2"
                )
                .requires_grad_(False)
                .eval()
                .half()
                .to(DEVICE),
                LlamaTokenizer.from_pretrained(model_path),
            ]
            for model_path in MODEL_PATHS
        }
        text_model, tokenizer = models[model]
        yield from get_result(
            text_model,
            tokenizer,
            rating,
            artist,
            characters,
            copyrights,
            target,
            special_tags,
            general,
            width / height,
            blacklist,
            escape_bracket,
            temperature,
        )


NODE_CLASS_MAPPINGS = {
    "PromptDanTagGen": DanTagGen,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PromptDanTagGen": "Danbooru Tag Generator",
}
