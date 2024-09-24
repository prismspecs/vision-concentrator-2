import torch
from diffusers import StableDiffusionXLPipeline
import os
import numpy as np
from tqdm import tqdm

def interpolate_embeddings(embed1, embed2, num_steps):
    """Interpolates between two embeddings."""
    embeddings = []
    for alpha in np.linspace(0, 1, num_steps):
        interpolated = (1 - alpha) * embed1 + alpha * embed2
        embeddings.append(interpolated)
    return embeddings

def generate_images(visions, frames_per_transition=30):
    model_file = os.getenv("MODEL_FILE", "/app/models/sd_xl_turbo_1.0_fp16.safetensors")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load the SDXL pipeline
    pipe = StableDiffusionXLPipeline.from_single_file(
        model_file,
        torch_dtype=torch.float16,
    )
    pipe.to(device)

    # Enable memory-efficient attention
    try:
        pipe.enable_xformers_memory_efficient_attention()
    except ModuleNotFoundError:
        print("xformers not installed; proceeding without memory-efficient attention.")

    # Create output directory
    os.makedirs('output', exist_ok=True)
    image_paths = []
    frame_idx = 0

    # Initialize tokenizers and text encoders
    tokenizer_1 = pipe.tokenizer
    text_encoder_1 = pipe.text_encoder

    tokenizer_2 = pipe.tokenizer_2
    text_encoder_2 = pipe.text_encoder_2

    # Prepare negative prompt embeddings
    negative_prompt = ""  # Customize if needed

    # Encode negative prompts for text encoder 1
    negative_input_1 = tokenizer_1(
        negative_prompt,
        padding="max_length",
        max_length=tokenizer_1.model_max_length,
        truncation=True,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        neg_embeds_1 = text_encoder_1(
            negative_input_1.input_ids,
            attention_mask=negative_input_1.attention_mask,
        )[0]  # Shape: [batch_size, seq_len1, hidden_size1]

    # Encode negative prompts for text encoder 2
    negative_input_2 = tokenizer_2(
        negative_prompt,
        padding="max_length",
        max_length=tokenizer_2.model_max_length,
        truncation=True,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        neg_embeds_2 = text_encoder_2(
            negative_input_2.input_ids,
            attention_mask=negative_input_2.attention_mask,
        )[0]  # Shape: [batch_size, seq_len2, hidden_size2]

    # Get pooled embeddings by mean pooling
    neg_pooled_embeds_2 = neg_embeds_2.mean(dim=1)  # Shape: [batch_size, hidden_size2]

    for i in range(len(visions) - 1):
        start_prompt = visions[i]
        end_prompt = visions[i + 1]

        # Tokenize and encode prompts for text encoder 1
        start_input_1 = tokenizer_1(
            start_prompt,
            padding="max_length",
            max_length=tokenizer_1.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).to(device)
        end_input_1 = tokenizer_1(
            end_prompt,
            padding="max_length",
            max_length=tokenizer_1.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            start_embeds_1 = text_encoder_1(
                start_input_1.input_ids,
                attention_mask=start_input_1.attention_mask,
            )[0]
            end_embeds_1 = text_encoder_1(
                end_input_1.input_ids,
                attention_mask=end_input_1.attention_mask,
            )[0]

        # Tokenize and encode prompts for text encoder 2
        start_input_2 = tokenizer_2(
            start_prompt,
            padding="max_length",
            max_length=tokenizer_2.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).to(device)
        end_input_2 = tokenizer_2(
            end_prompt,
            padding="max_length",
            max_length=tokenizer_2.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            start_embeds_2 = text_encoder_2(
                start_input_2.input_ids,
                attention_mask=start_input_2.attention_mask,
            )[0]
            end_embeds_2 = text_encoder_2(
                end_input_2.input_ids,
                attention_mask=end_input_2.attention_mask,
            )[0]

        # Get pooled embeddings by mean pooling
        start_pooled_embeds_2 = start_embeds_2.mean(dim=1)  # Shape: [batch_size, hidden_size2]
        end_pooled_embeds_2 = end_embeds_2.mean(dim=1)

        # Print embedding shapes for debugging
        print(f"Start prompt embeds shape: {start_embeds_1.shape}")
        print(f"End prompt embeds shape: {end_embeds_1.shape}")
        print(f"Start pooled embeds shape: {start_pooled_embeds_2.shape}")
        print(f"End pooled embeds shape: {end_pooled_embeds_2.shape}")

        # Interpolate between embeddings
        interpolated_prompt_embeds = interpolate_embeddings(
            start_embeds_1, end_embeds_1, frames_per_transition
        )
        interpolated_add_text_embeds = interpolate_embeddings(
            start_pooled_embeds_2, end_pooled_embeds_2, frames_per_transition
        )

        # Generate images for each interpolated embedding
        for emb1, emb2 in tqdm(
            zip(interpolated_prompt_embeds, interpolated_add_text_embeds),
            total=frames_per_transition,
            desc=f"Transition {i+1}/{len(visions)-1}",
        ):
            with torch.autocast(device):
                image = pipe(
                    prompt_embeds=emb1,
                    add_text_embeds=emb2,
                    negative_prompt_embeds=neg_embeds_1,
                    negative_add_text_embeds=neg_pooled_embeds_2,
                    num_inference_steps=25,
                    guidance_scale=7.5,
                ).images[0]

            image_path = f"output/frame_{frame_idx:05d}.png"
            image.save(image_path)
            image_paths.append(image_path)
            frame_idx += 1

    # Generate an image for the last vision
    last_input_1 = tokenizer_1(
        visions[-1],
        padding="max_length",
        max_length=tokenizer_1.model_max_length,
        truncation=True,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        last_embeds_1 = text_encoder_1(
            last_input_1.input_ids,
            attention_mask=last_input_1.attention_mask,
        )[0]

    last_input_2 = tokenizer_2(
        visions[-1],
        padding="max_length",
        max_length=tokenizer_2.model_max_length,
        truncation=True,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        last_embeds_2 = text_encoder_2(
            last_input_2.input_ids,
            attention_mask=last_input_2.attention_mask,
        )[0]

    last_pooled_embeds_2 = last_embeds_2.mean(dim=1)

    with torch.autocast(device):
        image = pipe(
            prompt_embeds=last_embeds_1,
            add_text_embeds=last_pooled_embeds_2,
            negative_prompt_embeds=neg_embeds_1,
            negative_add_text_embeds=neg_pooled_embeds_2,
            num_inference_steps=25,
            guidance_scale=7.5,
        ).images[0]

    image_path = f"output/frame_{frame_idx:05d}.png"
    image.save(image_path)
    image_paths.append(image_path)

    return image_paths
