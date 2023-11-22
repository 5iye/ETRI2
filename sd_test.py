from diffusers import StableDiffusionPipeline
from diffusers import DDIMScheduler
import torch 
from PIL import Image 
from tqdm.auto import tqdm


ddim_scheduler = DDIMScheduler(beta_start=0.00085,
                               beta_end=0.012,
                               beta_schedule="scaled_linear",
                               clip_sample=False,
                               set_alpha_to_one=False,
                               steps_offset=1)

model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(
    model_id, scheduler=ddim_scheduler)
pipe = pipe.to("cuda")


scheduler = pipe.scheduler
# print(scheduler.timesteps)
tokenizer = pipe.tokenizer
text_encoder = pipe.text_encoder
vae = pipe.vae #vraition auto encorder
unet = pipe.unet

prompt = ["a phto of a tiger"]
height = 512  # default height of Stable Diffusion
width = 512  # default width of Stable Diffusion
num_inference_steps = 25  # Number of denoising steps
guidance_scale = 7.5  # Scale for classifier-free guidance
generator = torch.manual_seed(0)  # Seed generator to create the initial latent noise
batch_size = len(prompt)

text_input = tokenizer(
    prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt"
)
with torch.no_grad():
    text_embeddings = text_encoder(text_input.input_ids.to("cuda"))[0]

    max_length = text_input.input_ids.shape[-1] # max length: 77
    uncond_input = tokenizer([""] * batch_size, 
                             padding="max_length", 
                             max_length=max_length, 
                             return_tensors="pt")
    uncond_embeddings = text_encoder(uncond_input.input_ids.to("cuda"))[0]

# text embedings.shape = 2, 77, 768]
text_embeddings = torch.cat([uncond_embeddings,
                             text_embeddings])

latents = torch.randn(
    (batch_size,
     unet.config.in_channels,
     height // 8, width // 8), # image (512,512) =>vae ==> Latent vector (64, 64)
     generator=generator,
)
latents = latents.to("cuda")

# print(latents.shape)

scheduler.set_timesteps(num_inference_steps)
for t in tqdm(scheduler.timesteps):
    # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
    latent_model_input = torch.cat([latents] * 2)

    latent_model_input = scheduler.scale_model_input(latent_model_input, 
                                                     timestep=t)

    # predict the noise residual
    with torch.no_grad():
        noise_pred = unet(latent_model_input, 
                          t, 
                          encoder_hidden_states=text_embeddings).sample

    # perform guidance
    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond) #null con

    # compute the previous noisy sample x_t -> x_t-1
    latents = scheduler.step(noise_pred, 
                             t, 
                             latents).prev_sample
    
latents = 1 / 0.18215 * latents  #0.18215 is nomalization sacaling factor (no meanning)
with torch.no_grad():
    image = vae.decode(latents).sample

image = (image / 2 + 0.5).clamp(0, 1).squeeze() # squeeze function : numpy => PIL conversion 
image = (image.permute(1, 2, 0) * 255).to(torch.uint8).cpu().numpy()
images = (image * 255).round().astype("uint8")
image = Image.fromarray(image)
image.save('test_image.jpg')
# for t in tqdm(scheduler.timesteps):
#     # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
#     latent_model_input = torch.cat([latents] * 2)

# prompt = "a photo of an astronaut riding a horse on mars"
# image = pipe(prompt).images[0]  
    
# image.save("astronaut_rides_horse.png")