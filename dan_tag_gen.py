import os
import torch
from transformers import LlamaForCausalLM, LlamaTokenizer

from .lib_dantaggen.app import get_result
from .lib_dantaggen.kgen.metainfo import TARGET
from .lib_dantaggen.kgen.logging import logger

MODEL_PATHS = [
    "KBlueLeaf/DanTagGen-alpha",
    "KBlueLeaf/DanTagGen-beta",
    "KBlueLeaf/DanTagGen-delta",
    "KBlueLeaf/DanTagGen-delta-rev2",
]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

ext_dir = os.path.dirname(os.path.realpath(__file__))

# text_model = None
# tokenizer = None
# modelPath = None
def loadModel(model_path):
    global text_model
    # global text_model, tokenizer, modelPath
    # if modelPath == model_path and text_model is not None:
    #     logger.warning(f"返回旧模型: {model_path}, {modelPath}")
    #     return text_model, tokenizer

    # logger.warning(f"重新加载模型: {model_path}, {modelPath}")

    #Find gguf model
    try:
        from llama_cpp import Llama, LLAMA_SPLIT_MODE_NONE

        text_model = Llama(
            f"{ext_dir}\models\{model_path}",
            n_ctx=384,
            split_mode=LLAMA_SPLIT_MODE_NONE,
            n_gpu_layers=100,
            verbose=False,
        )
        tokenizer = None
    except Exception as e:
        # 捕获其他所有异常
        logger.warning(f"Llama-cpp-python not found, using transformers to load model: {e}")
        from transformers import LlamaForCausalLM, LlamaTokenizer

        text_model = (
            LlamaForCausalLM.from_pretrained(model_path).eval().half()
        )
        tokenizer = LlamaTokenizer.from_pretrained(model_path)
        if torch.cuda.is_available():
            text_model = text_model.cuda()
        else:
            text_model = text_model.cpu()
    return text_model, tokenizer

class DanTagGen:
    """DanTagGen node."""
    model = None
    text_model = None
    tokenizer = None

    @classmethod
    def INPUT_TYPES(s):
        MODEL_PATHSs = MODEL_PATHS[:]
        for f in os.listdir(ext_dir + "/models"):
            if f.endswith(".gguf"):
                MODEL_PATHSs.append(f)

        return {
            "required": {
                "model": (MODEL_PATHSs,),
                "artist": ("STRING", {"default": ""}),
                "characters": ("STRING", {"default": ""}),
                "copyrights": ("STRING", {"default": ""}),
                "special_tags": ("STRING", {"default": ""}),
                "general": ("STRING", {"default": "", "multiline": True}),
                "blacklist": ("STRING", {"default": ""}),
                "rating": (["safe", "sensitive", "nsfw", "nsfw, explicit"],),
                "target": (list(TARGET.keys()),),
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

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("output", "llm_output")
    FUNCTION = "generate"
    CATEGORY = "_for_testing"

    def generate(
        self,
        model: str,
        rating: str,
        artist: str,
        characters: str,
        copyrights: str,
        target: str,
        special_tags: str,
        general: str,
        width: float,
        height: float,
        blacklist: str,
        escape_bracket: bool,
        temperature: float
    ):
        if self.model != model:
            self.model = model
            self.text_model, self.tokenizer = loadModel(model)
        # models = {
        #     model_path: [
        #         LlamaForCausalLM.from_pretrained(model_path)
        #         .requires_grad_(False)
        #         .eval()
        #         .half()
        #         .to(DEVICE),
        #         LlamaTokenizer.from_pretrained(model_path),
        #     ]
        #     for model_path in MODEL_PATHS
        # }
        # text_model, tokenizer = models[model]
        result = list(
            get_result(
                self.text_model,
                self.tokenizer,
                rating,
                artist,
                characters,
                copyrights,
                target,
                [s.strip() for s in special_tags.split(",") if s],
                general,
                width / height,
                blacklist,
                escape_bracket,
                temperature,
            )
        )[-1]
        output, llm_output, _ = result
        return {"result": (output, llm_output)}


NODE_CLASS_MAPPINGS = {
    "PromptDanTagGen": DanTagGen,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PromptDanTagGen": "Danbooru Tag Generator",
}
