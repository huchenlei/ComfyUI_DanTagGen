import subprocess

subprocess.run(
    "pip install flash-attn --no-build-isolation",
    env={"FLASH_ATTENTION_SKIP_CUDA_BUILD": "TRUE"},
    shell=True,
)

from time import time_ns

import gradio as gr
import torch
from transformers import LlamaForCausalLM, LlamaTokenizer

from .kgen.generate import tag_gen
from .kgen.metainfo import SPECIAL, TARGET


MODEL_PATHS = ["KBlueLeaf/DanTagGen-alpha", "KBlueLeaf/DanTagGen-beta"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")


@torch.no_grad()
def get_result(
    text_model: LlamaForCausalLM,
    tokenizer: LlamaTokenizer,
    rating: str = "",
    artist: str = "",
    characters: str = "",
    copyrights: str = "",
    target: str = "long",
    special_tags: list[str] = ["1girl"],
    general: str = "",
    aspect_ratio: float = 0.0,
    blacklist: str = "",
    escape_bracket: bool = False,
    temperature: float = 1.35,
):
    start = time_ns()
    print("=" * 50, "\n")
    # Use LLM to predict possible summary
    # This prompt allow model itself to make request longer based on what it learned
    # Which will be better for preference sim and pref-sum contrastive scorer
    prompt = f"""
rating: {rating or '<|empty|>'}
artist: {artist.strip() or '<|empty|>'}
characters: {characters.strip() or '<|empty|>'}
copyrights: {copyrights.strip() or '<|empty|>'}
aspect ratio: {f"{aspect_ratio:.1f}" or '<|empty|>'}
target: {'<|' + target + '|>' if target else '<|long|>'}
general: {", ".join(special_tags)}, {general.strip().strip(",")}<|input_end|>
""".strip()

    artist = artist.strip().strip(",").replace("_", " ")
    characters = characters.strip().strip(",").replace("_", " ")
    copyrights = copyrights.strip().strip(",").replace("_", " ")
    special_tags = [tag.strip().replace("_", " ") for tag in special_tags]
    general = general.strip().strip(",")
    black_list = set(
        [tag.strip().replace("_", " ") for tag in blacklist.strip().split(",")]
    )

    prompt_tags = special_tags + general.strip().strip(",").split(",")
    len_target = TARGET[target]
    llm_gen = ""

    for llm_gen, extra_tokens in tag_gen(
        text_model,
        tokenizer,
        prompt,
        prompt_tags,
        len_target,
        black_list,
        temperature=temperature,
        top_p=0.95,
        top_k=100,
        max_new_tokens=256,
        max_retry=5,
    ):
        yield "", llm_gen, f"Total cost time: {(time_ns()-start)/1e9:.2f}s"
    print()
    print("-" * 50)

    general = f"{general.strip().strip(',')}, {','.join(extra_tokens)}"
    tags = general.strip().split(",")
    tags = [tag.strip() for tag in tags if tag.strip()]
    special = special_tags + [tag for tag in tags if tag in SPECIAL]
    tags = [tag for tag in tags if tag not in special]

    final_prompt = ", ".join(special)
    if characters:
        final_prompt += f", \n\n{characters}"
    if copyrights:
        final_prompt += ", "
        if not characters:
            final_prompt += "\n\n"
        final_prompt += copyrights
    if artist:
        final_prompt += f", \n\n{artist}"
    final_prompt += f""", \n\n{', '.join(tags)},
masterpiece, newest, absurdres, {rating}"""

    print(final_prompt)
    print("=" * 50)

    if escape_bracket:
        final_prompt = (
            final_prompt.replace("[", "\\[")
            .replace("]", "\\]")
            .replace("(", "\\(")
            .replace(")", "\\)")
        )

    yield final_prompt, llm_gen, f"Total cost time: {(time_ns()-start)/1e9:.2f}s  |  Total general tags: {len(special+tags)}"


if __name__ == "__main__":
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

    def wrapper(
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

    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        gr.Markdown("""# DanTagGen beta DEMO""")
        with gr.Accordion("Introduction and Instructions"):
            gr.Markdown(
                """
#### What is this:
DanTagGen(Danbooru Tag Generator) is a LLM model designed for generating Danboou Tags with provided informations.<br>
It aims to provide user a more convinient way to make prompts for Text2Image model which is trained on Danbooru datasets.
#### How to use it:
1. Fill the informations on the left section.
2. Put the general tags you want to use into the "Input your general tags" textarea. ("prompt before refined")
3. If you want to ban some tags. Put them into the "black list" text area.
4. Choose the target length: **Long or Short is recommended**
    * Very Short: around 10 tags
    * Short: around 20 tags
    * Long: around 40 tags
    * very long: around 60 tags
5. Adjust some parameters
    * Width and height is for calculating the aspect ratio. It is recommended to directly put the height and width you want to use
6. Submit!!
7. You will get formated result on the upper-right section, LLM raw result on the bottom-right section.
#### Notice
The formated result use same format as what Kohaku-XL Delta used. <br>
The performance of using the output from this demo for other model is not guaranteed.
"""
            )
        with gr.Row():
            with gr.Column(scale=4):
                with gr.Row():
                    with gr.Column(scale=2):
                        rating = gr.Radio(
                            ["safe", "sensitive", "nsfw", "nsfw, explicit"],
                            value="safe",
                            label="Rating",
                        )
                        special_tags = gr.Dropdown(
                            SPECIAL,
                            value=["1girl"],
                            label="Special tags",
                            multiselect=True,
                        )
                        characters = gr.Textbox(label="Characters")
                        copyrights = gr.Textbox(label="Copyrights(Series)")
                        artist = gr.Textbox(label="Artist")
                        target = gr.Radio(
                            ["very_short", "short", "long", "very_long"],
                            value="long",
                            label="Target length",
                        )
                    with gr.Column(scale=2):
                        general = gr.TextArea(label="Input your general tags", lines=6)
                        black_list = gr.TextArea(
                            label="tag Black list (seperated by comma)", lines=5
                        )
                        with gr.Row():
                            width = gr.Slider(
                                value=1024,
                                minimum=256,
                                maximum=4096,
                                step=32,
                                label="Width",
                            )
                            height = gr.Slider(
                                value=1024,
                                minimum=256,
                                maximum=4096,
                                step=32,
                                label="Height",
                            )
                        with gr.Row():
                            temperature = gr.Slider(
                                value=1.35,
                                minimum=0.1,
                                maximum=2,
                                step=0.05,
                                label="Temperature",
                            )
                            escape_bracket = gr.Checkbox(
                                value=False,
                                label="Escape bracket",
                            )
                        model = gr.Dropdown(
                            list(models.keys()),
                            value=list(models.keys())[-1],
                            label="Model",
                        )
                submit = gr.Button("Submit")
            with gr.Column(scale=3):
                formated_result = gr.TextArea(
                    label="Final output", lines=14, show_copy_button=True
                )
                llm_result = gr.TextArea(label="LLM output", lines=10)
                cost_time = gr.Markdown()
        submit.click(
            wrapper,
            inputs=[
                model,
                rating,
                artist,
                characters,
                copyrights,
                target,
                special_tags,
                general,
                width,
                height,
                black_list,
                escape_bracket,
                temperature,
            ],
            outputs=[
                formated_result,
                llm_result,
                cost_time,
            ],
            show_progress=True,
        )

    demo.launch()
