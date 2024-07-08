from LLMs import get_random_models,get_response_model,model_dict
from dotenv import load_dotenv
import pandas as pd
import gradio as gr
import json
import os

load_dotenv()

scores={}
df=None
if os.path.exists("results.json"):
    with open("results.json", 'r') as file:
        scores=json.load(file)
else:
    for model_name in model_dict.keys():
        scores[model_name]=0

def save_scores():
    global df
    with open("results.json", 'w') as file:
        json.dump(scores,file)
        df=pd.DataFrame(list(scores.items()), columns=["Model", "Score"])

current_models=get_random_models()
while(current_models[0]==current_models[1]):
    current_models=get_random_models()
model1_name=None
model2_name=None
for key,value in model_dict.items():
    if value==current_models[0]:
        model1_name=key
    if value==current_models[1]:
        model2_name=key

def generate_responses(prompt):
    modeli1,modeli2=current_models[0],current_models[1]
    response1=get_response_model(modeli1, prompt)
    response2=get_response_model(modeli2, prompt)
    return [(prompt, response1)],[(prompt, response2)]

def voting(v_m1,v_m2):   
    if v_m1:
        scores[model1_name]+=1
    if v_m2:
        scores[model2_name]+=1
    save_scores()
    return df,model1_name,model2_name

def new_round():
    global current_models,model1_name,model2_name
    current_models=get_random_models()
    while(current_models[0]==current_models[1]):
        current_models=get_random_models()
    for key,value in model_dict.items():
        if value==current_models[0]:
            model1_name=key
        if value==current_models[1]:
            model2_name=key
    return None,None,"",""


def main():
    with gr.Blocks(css=".gradio-container") as demo:
        with gr.Tab("⚔️Arena(battle)"):
            gr.Markdown("# LLM Arena")
            gr.Markdown("📜 Rules")
            Rules="""
            - Ask any question and vote the 2 models based on their performances.
            - Make fair voting and don't try to manipulate the models
            - Use 🎲 Regenerate Models 🎲 option to start the arena with new set of models.
            - Model names cannot be known till voting to ensure fairness.
            - Use 🧹 Clear option to play with the same models again.
            """
            gr.Markdown(Rules)

            with gr.Row():
                    response1=gr.Chatbot(label="Model A")
                    response2=gr.Chatbot(label="Model B")

            with gr.Row():
                prompt=gr.Textbox(label="Enter your prompt:", placeholder="Enter your prompt here", lines=1,scale=3)
                generate_btn=gr.Button("Submit")
                
            with gr.Accordion("🥷 Current Models", open=False):
                with gr.Row():
                    mod1=gr.Textbox(interactive=False)
                    mod2=gr.Textbox(interactive=False)
                    
            with gr.Accordion("👆 Vote models", open=False):
                with gr.Row():
                    btn_v1=gr.Button("Model A wins")
                    btn_v2=gr.Button("Model B wins")
                    btn_both=gr.Button("Draw")

            with gr.Row():
                    new_round_btn=gr.Button("🎲 Regenerate Models 🎲")
                    clear_btn=gr.Button("🧹 Clear")
        with gr.Tab("🏆 Leaderboard"):
            df=pd.DataFrame(list(scores.items()), columns=["Model", "Score"])
            df1=gr.DataFrame(value=df,interactive=False)

        generate_btn.click(generate_responses, inputs=[prompt], outputs=[response1,response2])
        btn_v1.click(voting, inputs=[gr.State(True), gr.State(False)], outputs=[df1,mod1,mod2])
        btn_v2.click(voting, inputs=[gr.State(False), gr.State(True)], outputs=[df1,mod1,mod2])
        btn_both.click(voting, inputs=[gr.State(True), gr.State(True)], outputs=[df1,mod1,mod2])
        new_round_btn.click(new_round, outputs=[response1,response2,mod1,mod2])
        clear_btn.click(lambda:([],[],""),outputs=[response1,response2,prompt])

    demo.launch(share=True)

if __name__ == "__main__":
    main()