import os
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize OpenAI client with API key from .env
openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

def run_gpt4o_mini(question):
    try:
        # Check if the question is a dictionary and extract the prompt
        if isinstance(question, dict) and 'prompt' in question:
            question_text = question['prompt']
        else:
            question_text = str(question)

        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant. Answer the question to the best of your ability."},
                {"role": "user", "content": question_text}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error running GPT-4o-mini: {str(e)}")
        return None

def run_gpt4o(question):
    try:
        # Check if the question is a dictionary and extract the prompt
        if isinstance(question, dict) and 'prompt' in question:
            question_text = question['prompt']
        else:
            question_text = str(question)

        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant. Answer the question to the best of your ability."},
                {"role": "user", "content": question_text}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error running GPT-4o-mini: {str(e)}")
        return None

def run_custom_model(model_name, question):
    # Placeholder for custom model logic
    # You'll need to implement this based on how your custom models work
    return f"Custom model {model_name} response: This is a placeholder answer for the question provided."

def run_model(model_name, question):
    if model_name == "gpt-4o-mini":
        return run_gpt4o_mini(question)
    elif model_name == "gpt-4o":
        return run_gpt4o(question)
    else:
        return run_custom_model(model_name, question)