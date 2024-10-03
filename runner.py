import os
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize OpenAI client
openai_client = OpenAI()

def run_gpt4o_mini(context, question):
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant. Use the provided context to answer the question."},
                {"role": "user", "content": f"Context: {context}\n\nQuestion: {question}"}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error running GPT-4o-mini: {str(e)}")
        return None

def run_custom_model(model_name, context, question):
    # Placeholder for custom model logic
    # You'll need to implement this based on how your custom models work
    return f"Custom model {model_name} response: This is a placeholder answer."

def run_model(model_name, context, question):
    if model_name == "gpt-4o-mini":
        return run_gpt4o_mini(context, question)
    elif model_name == "gpt-4o":
        # Implement GPT-4o logic here if different from GPT-4o-mini
        return run_gpt4o_mini(context, question)  # For now, using the same function
    else:
        return run_custom_model(model_name, context, question)