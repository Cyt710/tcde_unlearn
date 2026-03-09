import torch
from diffusers import DiffusionPipeline
from transformers import CLIPTextModel, CLIPTokenizer

def load_custom_diffusion_pipeline(
    model_id: str,  
    custom_text_encoder_path: str,  
    device: str = "cuda", 
    torch_dtype: torch.dtype = torch.float16,  
    use_safety_checker: bool = False  
):
    
    print(f"Loading base model from {model_id}...")
    
    # load basic pipeline
    safety_checker = None if not use_safety_checker else None
    pipe = DiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        safety_checker=safety_checker
    ).to(device)
    
    if custom_text_encoder_path is not None:
        print(f"Loading custom CLIP text encoder from {custom_text_encoder_path}...")
        
        try:
            text_encoder = CLIPTextModel.from_pretrained(
                custom_text_encoder_path,
                torch_dtype=torch_dtype
            )
            
            tokenizer = CLIPTokenizer.from_pretrained(
                custom_text_encoder_path
            )
            
            text_encoder = text_encoder.to(device)
            
            pipe.text_encoder = text_encoder
            pipe.tokenizer = tokenizer
            
            print("Successfully replaced text encoder and tokenizer!")
            
        except Exception as e:
            raise Exception(f"Failed to load custom text encoder: {e}")
 
 

    return pipe

def generate_images(
    pipe: DiffusionPipeline,
    prompts: list,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    negative_prompt: str = None,
    seed: int = None
):
    
    if seed is not None:
        torch.manual_seed(seed)
    
    images = pipe(
        prompt=prompts,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        negative_prompt=negative_prompt,
        generator=torch.Generator(device=pipe.device).manual_seed(seed)
    ).images
    
    return images

#Usage example

if __name__ == "__main__":
   
    MODEL_ID = "your path to model"  
    CUSTOM_ENCODER_PATH = "your path to machine unlearned model"  
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    pipe = load_custom_diffusion_pipeline(
        model_id=MODEL_ID,
        custom_text_encoder_path=CUSTOM_ENCODER_PATH,
        device=DEVICE,
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32
    )
    
    prompts = [    
        
        "a picture of apple product on a table",
       
    ]
    
    images = generate_images(
        pipe=pipe,
        prompts=prompts,
        num_inference_steps=40,
        guidance_scale=7.5,
        seed=42
    )
    
    for i, img in enumerate(images):
        img.save(f"generated_image_apple0{i}.png")
        print(f"Saved generated_image_{i}.png")