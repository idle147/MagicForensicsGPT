import math
import torch
from transformers import AutoModel, AutoTokenizer
from processor.image_processor import ImageProcessor
from transformers import TextIteratorStreamer
from threading import Thread


def split_model(model_name):
    device_map = {}
    world_size = torch.cuda.device_count()
    num_layers = {
        "InternVL2-1B": 24,
        "InternVL2-2B": 24,
        "InternVL2-4B": 32,
        "InternVL2-8B": 32,
        "InternVL2-26B": 48,
        "InternVL2-40B": 60,
        "InternVL2-Llama3-76B": 80,
    }[model_name]
    # Since the first GPU will be used for ViT, treat it as half a GPU.
    num_layers_per_gpu = math.ceil(num_layers / (world_size - 0.5))
    num_layers_per_gpu = [num_layers_per_gpu] * world_size
    num_layers_per_gpu[0] = math.ceil(num_layers_per_gpu[0] * 0.5)
    layer_cnt = 0
    for i, num_layer in enumerate(num_layers_per_gpu):
        for j in range(num_layer):
            device_map[f"language_model.model.layers.{layer_cnt}"] = i
            layer_cnt += 1
    device_map["vision_model"] = 0
    device_map["mlp1"] = 0
    device_map["language_model.model.tok_embeddings"] = 0
    device_map["language_model.model.embed_tokens"] = 0
    device_map["language_model.output"] = 0
    device_map["language_model.model.norm"] = 0
    device_map["language_model.lm_head"] = 0
    device_map[f"language_model.model.layers.{num_layers - 1}"] = 0

    return device_map


def main():
    image_processor = ImageProcessor()

    # set the max number of tiles in `max_num`
    pixel_values = image_processor.load_image("./examples/image1.jpg").to(torch.bfloat16).cuda()
    generation_config = dict(max_new_tokens=1024, do_sample=True)

    # pure-text conversation (纯文本对话)
    question = "Hello, who are you?"
    response, history = model.chat(tokenizer, None, question, generation_config, history=None, return_history=True)
    print(f"User: {question}\nAssistant: {response}")

    question = "Can you tell me a story?"
    response, history = model.chat(tokenizer, None, question, generation_config, history=history, return_history=True)
    print(f"User: {question}\nAssistant: {response}")

    # single-image single-round conversation (单图单轮对话)
    question = "<image>\nPlease describe the image shortly."
    response = model.chat(tokenizer, pixel_values, question, generation_config)
    print(f"User: {question}\nAssistant: {response}")

    # single-image multi-round conversation (单图多轮对话)
    question = "<image>\nPlease describe the image in detail."
    response, history = model.chat(tokenizer, pixel_values, question, generation_config, history=None, return_history=True)
    print(f"User: {question}\nAssistant: {response}")

    question = "Please write a poem according to the image."
    response, history = model.chat(tokenizer, pixel_values, question, generation_config, history=history, return_history=True)
    print(f"User: {question}\nAssistant: {response}")

    # multi-image multi-round conversation, combined images (多图多轮对话，拼接图像)
    pixel_values1 = image_processor.load_image("./examples/image1.jpg", max_num=12).to(torch.bfloat16).cuda()
    pixel_values2 = image_processor.load_image("./examples/image2.jpg", max_num=12).to(torch.bfloat16).cuda()
    pixel_values = torch.cat((pixel_values1, pixel_values2), dim=0)

    question = "<image>\nDescribe the two images in detail."
    response, history = model.chat(tokenizer, pixel_values, question, generation_config, history=None, return_history=True)
    print(f"User: {question}\nAssistant: {response}")

    question = "What are the similarities and differences between these two images."
    response, history = model.chat(tokenizer, pixel_values, question, generation_config, history=history, return_history=True)
    print(f"User: {question}\nAssistant: {response}")

    # multi-image multi-round conversation, separate images (多图多轮对话，独立图像)
    pixel_values1 = image_processor.load_image("./examples/image1.jpg", max_num=12).to(torch.bfloat16).cuda()
    pixel_values2 = image_processor.load_image("./examples/image2.jpg", max_num=12).to(torch.bfloat16).cuda()
    pixel_values = torch.cat((pixel_values1, pixel_values2), dim=0)
    num_patches_list = [pixel_values1.size(0), pixel_values2.size(0)]

    question = "Image-1: <image>\nImage-2: <image>\nDescribe the two images in detail."
    response, history = model.chat(
        tokenizer, pixel_values, question, generation_config, num_patches_list=num_patches_list, history=None, return_history=True
    )
    print(f"User: {question}\nAssistant: {response}")

    question = "What are the similarities and differences between these two images."
    response, history = model.chat(
        tokenizer, pixel_values, question, generation_config, num_patches_list=num_patches_list, history=history, return_history=True
    )
    print(f"User: {question}\nAssistant: {response}")

    # batch inference, single image per sample (单图批处理)
    pixel_values1 = image_processor.load_image("./examples/image1.jpg", max_num=12).to(torch.bfloat16).cuda()
    pixel_values2 = image_processor.load_image("./examples/image2.jpg", max_num=12).to(torch.bfloat16).cuda()
    num_patches_list = [pixel_values1.size(0), pixel_values2.size(0)]
    pixel_values = torch.cat((pixel_values1, pixel_values2), dim=0)

    questions = ["<image>\nDescribe the image in detail."] * len(num_patches_list)
    responses = model.batch_chat(
        tokenizer, pixel_values, num_patches_list=num_patches_list, questions=questions, generation_config=generation_config
    )
    for question, response in zip(questions, responses):
        print(f"User: {question}\nAssistant: {response}")


def stream_inference():
    image_processor = ImageProcessor()
    # Initialize the streamer
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=10)
    # Define the generation configuration
    generation_config = dict(max_new_tokens=1024, do_sample=False, streamer=streamer)

    question = "Hello, who are you?"
    # Start the model chat in a separate thread
    thread = Thread(
        target=model.chat,
        kwargs=dict(
            tokenizer=tokenizer,
            pixel_values=image_processor.load_image("./examples/image1.jpg", max_num=12).to(torch.bfloat16).cuda(),
            question=question,
            history=None,
            return_history=False,
            generation_config=generation_config,
        ),
    )
    thread.start()

    # Initialize an empty string to store the generated text
    generated_text = ""
    # Loop through the streamer to get the new text as it is generated
    for new_text in streamer:
        if new_text == model.conv_template.sep:
            break
        generated_text += new_text
        print(new_text, end="", flush=True)  # Print each new chunk of generated text on the same line


if __name__ == "__main__":
    # If you set `load_in_8bit=True`, you will need two 80GB GPUs.
    # If you set `load_in_8bit=False`, you will need at least three 80GB GPUs.
    path = "OpenGVLab/InternVL2-Llama3-76B"
    device_map = split_model("InternVL2-Llama3-76B")
    model = AutoModel.from_pretrained(
        path,
        torch_dtype=torch.bfloat16,
        load_in_8bit=True,
        low_cpu_mem_usage=True,
        use_flash_attn=True,
        trust_remote_code=True,
        device_map=device_map,
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)

    main()
    # stream_inference()
