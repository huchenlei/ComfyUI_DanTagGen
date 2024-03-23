# ComfyUI_DanTagGen

ComfyUI node of [Kohaku's DanTagGen Demo](https://huggingface.co/KBlueLeaf/DanTagGen?not-for-all-audiences=true). It can generate the detail tags/core tags about the character you put in the prompts. It can also add some extra elements into your prompt.

![image](https://github.com/huchenlei/ComfyUI_DanTagGen/assets/20929282/1c6828ab-47de-4871-b317-4f27863343f8)

## What is DanTagGen

DanTagGen(Danbooru Tag Generator) is a LLM model designed for generating Danboou Tags with provided informations.
It aims to provide user a more convinient way to make prompts for Text2Image model which is trained on Danbooru datasets.

More information about model arch and training data can be found in the HuggingFace Model card:

[KBlueLeaf/DanTagGen-beta · Hugging Face](https://huggingface.co/KBlueLeaf/DanTagGen-beta)


## How to use it
Load the [example workflow](https://github.com/huchenlei/ComfyUI_DanTagGen/blob/main/examples/dtg.json) and connect the output to `CLIP Text Encode (Prompt)`'s text input. You can right click `CLIP Text Encode (Prompt)` to convert in-node text input to external text input.

![image](https://github.com/huchenlei/ComfyUI_DanTagGen/assets/20929282/466bfcb2-4a8c-48a3-8c5a-53ad170e0f6c)

### Options

* tag length:
  * very short: around 10 tags
  * short: around 20 tags
  * long: around 40 tags
  * very long: around 60 tags
  * ***short or long is recommended***
* Ban tags: The black list of tags you don't want to see in final prompt. Regex supported.
* Temperature: Higher = more dynamic result, Lower = better coherence between tags.

## Faster inference

If you think the transformers implementation is slow and want to get better speed. You can install `llama-cpp-python` by yourself and then download the gguf model from HuggingFace and them put them into the `models` folder.

(Automatic installation/download script for llama-cpp-python and gguf model are WIP)

More information about `llama-cpp-python`:

* [abetlen/llama-cpp-python: Python bindings for llama.cpp (github.com)](https://github.com/abetlen/llama-cpp-python)
* [jllllll/llama-cpp-python-cuBLAS-wheels: Wheels for llama-cpp-python compiled with cuBLAS support (github.com)](https://github.com/jllllll/llama-cpp-python-cuBLAS-wheels)

## Web UI Alternative
The official A1111/Forge extension for DTG can be found here: https://github.com/KohakuBlueleaf/a1111-sd-webui-dtg.
