from .lib_dantaggen.app import get_result

class DanTagGen:
    """DanTagGen node."""
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    FUNCTION = "generate"
    CATEGORY = "prompt"

    def generate(self):
        pass

NODE_CLASS_MAPPINGS = {

}

NODE_DISPLAY_NAME_MAPPINGS = {

}
