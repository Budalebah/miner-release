import re
import os
import sys
import torch
import io
import gc
import logging
import time
from diffusers import AutoencoderKL, DPMSolverMultistepScheduler
from vendor.lpw_stable_diffusion_xl import StableDiffusionXLLongPromptWeightingPipeline
from vendor.lpw_stable_diffusion import StableDiffusionLongPromptWeightingPipeline

def get_local_model_ids(config):
    local_files = os.listdir(config.base_dir)
    local_model_ids = [model['name'] for model in config.model_configs.values() if model['name'] + ".safetensors" in local_files]
    return local_model_ids

def load_model(config, model_id):
    # Error handling for excluded sdxl models
    if config.exclude_sdxl and model_id.startswith("SDXL"):
        error_message = f"Loading of 'sdxl' models is disabled. Model '{model_id}' cannot be loaded as per configuration."
        print(error_message)  # Print the error message to the console
        raise ValueError(error_message)  # Optionally, raise an exception to halt the process
    
    start_time = time.time()
    model_config = config.model_configs.get(model_id, None)
    if model_config is None:
        raise Exception(f"Model configuration for {model_id} not found.")

    model_file_path = os.path.join(config.base_dir, f"{model_id}.safetensors")

    # Load the main model
    if model_config['type'] == "sd15":
        pipe = StableDiffusionLongPromptWeightingPipeline.from_single_file(model_file_path, torch_dtype=torch.float16).to('cuda:' + str(config.cuda_device_id))
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, use_karras_sigmas=True, algorithm_type="sde-dpmsolver++")
    else:
        pipe = StableDiffusionXLLongPromptWeightingPipeline.from_single_file(model_file_path, torch_dtype=torch.float16).to('cuda:' + str(config.cuda_device_id))
    pipe.safety_checker = None
    # TODO: Add support for other schedulers

    if 'vae' in model_config:
        vae_name = model_config['vae']
        vae_file_path = os.path.join(config.base_dir, f"{vae_name}.safetensors")
        vae = AutoencoderKL.from_single_file(vae_file_path, torch_dtype=torch.float16).to('cuda:' + str(config.cuda_device_id))
        pipe.vae = vae
    
    end_time = time.time()  # End measuring time
    # Calculate and log the loading latency
    loading_latency = end_time - start_time

    return pipe, loading_latency

def unload_model(config, model_id):
    if model_id in config.loaded_models:
        del config.loaded_models[model_id]
        torch.cuda.empty_cache()
        gc.collect()

def load_lora(pipe, config, lora_id):
    # Verify lora_id exists in the config's lora configurations
    if lora_id not in config.lora_configs:
        raise ValueError(f"LoRa ID '{lora_id}' not found in configuration.")
    
    # Construct the path to the LoRa weights file
    lora_name = f"{lora_id}.safetensors"
    lora_path = os.path.join(config.base_dir, lora_name)
    
    if not os.path.exists(lora_path):
        raise FileNotFoundError(f"LoRa weights file '{lora_path}' not found.")
    try:
        unique_id = config.lora_cache.get_unique_id(lora_id)
        if unique_id is None:
            unique_id = f"default_{len(config.lora_cache.unique_ids)}"
            config.lora_cache.unique_ids[lora_id] = unique_id
        pipe.load_lora_weights(lora_path, adapter_name=unique_id)
        pipe.set_adapters(config.lora_cache.get_unique_id(lora_id))
        return pipe
    except Exception as e:
        # It's a good practice to catch and handle or log specific exceptions
        print(f"Failed to load LoRa weights for '{lora_id}': {e}")
        raise

def unload_lora(config, current_model):
    if config.lora_cache.cache:
        lora_id = next(iter(config.lora_cache.cache))
        current_model.unload_lora_weights()
        logging.debug(f"Unloaded LoRa weights for '{lora_id}' to free up resources.")
        # Remove the unloaded LoRa model from the cache
        config.lora_cache.remove(lora_id)
        torch.cuda.empty_cache()
        gc.collect()

def load_lora_with_cache(config, current_model, lora_id):
    cached_model = config.lora_cache.get(lora_id)
    if cached_model is not None:
        cached_model.set_adapters(config.lora_cache.get_unique_id(lora_id))
        print(f"Using cached lora with identifier: {lora_id}")
        return cached_model

    current_model_with_lora = load_lora(current_model, config, lora_id)
    config.lora_cache.put(lora_id, current_model_with_lora)
    return current_model_with_lora

def load_default_model(config):
    model_ids = get_local_model_ids(config)
    if not model_ids:
        logging.error("No local models found. Exiting...")
        sys.exit(1)  # Exit if no models are available locally
    default_model_id = model_ids[8]

    if default_model_id not in config.loaded_models:
        logging.info(f"Loading default model {default_model_id}...")
        current_model, _ = load_model(config, default_model_id)
        config.loaded_models[default_model_id] = current_model
        print(f"Default model {default_model_id} loaded successfully.")
    
    if not config.exclude_sdxl and config.enable_lora:
        print("Lora adapter enabled")
    # Only first two loras loaded to adapt the lpw_xl pipeline
        for lora_id in list(config.lora_configs.keys())[:2]:
            try:
                logging.info(f"Loading LoRa model {lora_id}...")
                _ = load_lora_with_cache(config, current_model, lora_id)
                print(f"LoRa weight {lora_id} loaded successfully.")
            except Exception as e:
                logging.error(f"Failed to load LoRa model {lora_id}: {e}")

def reload_model(config, model_id_from_signal):
    if config.loaded_models:
        model_to_unload = next(iter(config.loaded_models))
        unload_model(config, model_to_unload)
        logging.debug(f"Unloaded model {model_to_unload} to make space for {model_id_from_signal}")

    logging.info(f"Loading model {model_id_from_signal}...")
    current_model, _ = load_model(config, model_id_from_signal)
    config.loaded_models[model_id_from_signal] = current_model
    logging.info(f"Received model {model_id_from_signal} loaded successfully.")

def execute_model(config, model_id, prompt, neg_prompt, height, width, num_iterations, guidance_scale, seed):
    try:
        current_model = config.loaded_models.get(model_id)
        loading_latency = None  # Indicates no loading occurred if the model was already loaded

        kwargs = {
            # For better/stable image quality, consider using larger height x weight values
            'height': min(height - height % 8, config.config['processing_limits']['max_height']),
            'width': min(width - width % 8, config.config['processing_limits']['max_width']),
            'num_inference_steps': min(num_iterations, config.config['processing_limits']['max_iterations']),
            'guidance_scale': guidance_scale,
            'negative_prompt': neg_prompt,
        }

        if seed is not None and seed >= 0:
            kwargs['generator'] = torch.Generator().manual_seed(seed)

        logging.debug(f"Executing model {model_id} with parameters: {kwargs}")
        
        if not config.exclude_sdxl and config.enable_lora:
            lora_pattern = re.compile(r"<lora:([^:>]+):(\d+)>")
            match = lora_pattern.search(prompt)
            if match:
                try:
                    current_model.enable_lora()
                    lora_id = match.group(1)
                    current_model_with_lora = load_lora_with_cache(config, current_model, lora_id)
                except Exception as e:
                    logging.error(f"Error loading LoRa weights for '{lora_id}': {e}")
                inference_start_time = time.time()
                images = current_model_with_lora(prompt, **kwargs).images
                inference_end_time = time.time()
            else:
                current_model.disable_lora()
                inference_start_time = time.time()                
                images = current_model(prompt, **kwargs).images
                inference_end_time = time.time()
        else:
            inference_start_time = time.time()
            images = current_model(prompt, **kwargs).images
            inference_end_time = time.time()

        # Calculate and log the inference latency
        inference_latency = inference_end_time - inference_start_time
        image_data = io.BytesIO()
        images[0].save(image_data, format='PNG')
        image_data.seek(0)
        return image_data, inference_latency, loading_latency
    
    except Exception as e:
        logging.error(f"Error executing model {model_id}: {e}")
        raise