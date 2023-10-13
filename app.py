import gradio as gr
import pickle
from gradio.components import Textbox, Number, Checkbox, Radio

# Load the trained model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Define the prediction function
def predict_price(ram, screen_size, harddisk, graphics, brand_Apple):
    graphics = 0 if graphics == "Integrated" else 1
    price = model.predict([[ram, screen_size, harddisk, graphics, brand_Apple]])[0]
    return f"${price:,.2f}"

# Create the Gradio interface
iface = gr.Interface(
    fn=predict_price,
    inputs=[
        Number(label="RAM in MB"),
        Number(label="Screen Size in Inches"),
        Number(label="Hard Disk Size in MB"),
        Radio(["Integrated", "Dedicated"], label="Graphics Card"),
        Checkbox(label="Apple")
    ],
    outputs=[Textbox(label="Price")],
    title="Laptop Price Predictor",
    description="Predict the price of a laptop given its specifications.",
    examples=[
        [4000, 15.6, 1024, "Integrated", False],
        [8000, 13.3, 512, "Integrated", True],
        [16000, 15.6, 1024, "Dedicated", False]
    ]
)

# Launch the interface
iface.launch(inbrowser=True)
