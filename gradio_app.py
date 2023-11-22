import numpy as np
import gradio as gr
from diffusers import StableDiffusionPipeline
import torch
from PIL import Image
from sd_model import generate
from sd_model import select

model_id_1_5 = "runwayml/stable-diffusion-v1-5"
model_id_2_1 = "stabilityai/stable-diffusion-2-1"
model_id_RV = "SG161222/Realistic_Vision_V2.0"
model_id_any = "xyn-ai/anything-v4.0"



with gr.Blocks() as demo:
    gr.Markdown("Stable Diffusion App")
    sd_version = gr.Dropdown(
        value = model_id_1_5,
        choices = 
            [model_id_1_5,
            model_id_2_1,
            model_id_RV,
            model_id_any],
            label="SD version"
    )
    with gr.Row():
        text_input = gr.Textbox(label="Input prmopts")
        neg_input = gr.Textbox(label="Neg prmopts")
    with gr.Row():
        num_inference_steps = gr.Number(label="inference_step",value=50)
        guidance_scale = gr.Number(label="guidance_scale",value=7.5)
        seed = gr.Number(label="seed",value=42)
    
    
    gen_btn = gr.Button("Generate")
    feaure_map = gr.Dropdown(
        choices = 
            ["conv_in",
            "downblock0",
            "downblock1",
            "downblock2",
            "downblock3",
            "midblock",
            "upblock0",
            "upblock1",
            "upblock2",
            "upblock3"],
            label="feaure_map"
    )


    feautre_select = gr.Button("feautre_select")
    with gr.Row():
        image_output = gr.Image(label="Generated Image")
        feature_output = gr.Image(label="Generated Feature")

    gen_btn.click(generate,
                  inputs=[sd_version,
                          text_input,
                          neg_input,
                          num_inference_steps,
                          guidance_scale,
                          seed],
                  outputs=[image_output])

    feautre_select.click(select,
                    inputs=feaure_map, outputs=[feature_output])


if __name__ == "__main__":
    demo.launch(server_name="129.254.23.116", show_api=False)   

