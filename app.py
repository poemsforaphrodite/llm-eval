import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from datetime import datetime, timedelta
import json
from pymongo import MongoClient
from dotenv import load_dotenv
import os
import bcrypt
from openai import OpenAI
from streamlit_plotly_events import plotly_events
from pinecone import Pinecone, ServerlessSpec
import threading  # {{ edit_25: Import threading for background processing }}
import tiktoken
from tiktoken.core import Encoding
from runner import run_model

# Set page configuration to wide mode
st.set_page_config(layout="wide")

# Load environment variables
load_dotenv()

# MongoDB connection
mongodb_uri = os.getenv('MONGODB_URI')
mongo_client = MongoClient(mongodb_uri)  # {{ edit_11: Rename MongoDB client to 'mongo_client' }}
db = mongo_client['llm_evaluation_system']
users_collection = db['users']
results_collection = db['evaluation_results']

# Initialize OpenAI client
openai_client = OpenAI()  # {{ edit_12: Rename OpenAI client to 'openai_client' }}

# Initialize Pinecone
pinecone_client = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))  # {{ edit_13: Initialize Pinecone client using Pinecone class }}

# Initialize the tokenizer
tokenizer: Encoding = tiktoken.get_encoding("cl100k_base")  # This is suitable for GPT-4 and recent models

# Authentication functions
def hash_password(password):
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

def verify_password(password, hashed_password):
    return bcrypt.checkpw(password.encode('utf-8'), hashed_password)

def authenticate(username, password):
    user = users_collection.find_one({"username": username})
    if user and verify_password(password, user['password']):
        return True
    return False

def signup(username, password):
    if users_collection.find_one({"username": username}):
        return False
    hashed_password = hash_password(password)
    # {{ edit_1: Initialize models list for the new user }}
    users_collection.insert_one({
        "username": username,
        "password": hashed_password,
        "models": []  # List to store user's models
    })
    return True
def upload_model(file):
    return "Model uploaded successfully!"

# Function to perform evaluation (placeholder)
def evaluate_model(model_identifier, metrics, username):
    # {{ edit_4: Differentiate between Custom and Named models }}
    user = users_collection.find_one({"username": username})
    models = user.get("models", [])
    selected_model = next((m for m in models if (m['model_name'] == model_identifier) or (m['model_id'] == model_identifier)), None)
    
    if selected_model:
        if selected_model.get("model_type") == "named":
            # For Named Models, use RAG-based evaluation
            return evaluate_named_model(model_identifier, prompt, context_dataset)
        else:
            # For Custom Models, proceed with existing evaluation logic
            results = {metric: round(np.random.rand() * 100, 2) for metric in metrics}
            return results
    else:
        st.error("Selected model not found.")
        return None

# Function to generate response using GPT-4-mini
def generate_response(prompt, context):
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"Context: {context}\n\nPrompt: {prompt}"}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Error generating response: {str(e)}")
        return None

# Function to clear the results database
def clear_results_database():
    try:
        results_collection.delete_many({})
        return True
    except Exception as e:
        st.error(f"Error clearing results database: {str(e)}")
        return False

# Function to generate embeddings using the specified model
def generate_embedding(text):
    try:
        embedding_response = openai_client.embeddings.create(
            model="text-embedding-3-large",  # {{ edit_3: Use the specified embedding model }}
            input=text,
            encoding_format="float"
        )
        embedding = embedding_response["data"][0]["embedding"]
        return embedding
    except Exception as e:
        st.error(f"Error generating embedding: {str(e)}")
        return None

# Function to handle Named Model Evaluation using RAG
def evaluate_named_model(model_name, prompt, context_dataset):
    # {{ edit_4: Implement evaluation using RAG and Pinecone with the specified embedding model }}
    try:
        # Initialize Pinecone index
        index = pinecone_client.Index(os.getenv('PINECONE_INDEX_NAME'))
        
        # Generate embedding for the prompt
        prompt_embedding = generate_embedding(prompt)
        if not prompt_embedding:
            st.error("Failed to generate embedding for the prompt.")
            return None
        
        # Retrieve relevant context using RAG by querying Pinecone with the embedding
        query_response = index.query(
            top_k=5,
            namespace=model_name,
            include_metadata=True,
            vector=prompt_embedding  # {{ edit_5: Use embedding vector for querying }}
        )
        
        # Aggregate retrieved context
        retrieved_context = " ".join([item['metadata']['text'] for item in query_response['matches']])
        
        # Generate response using the retrieved context
        response = generate_response(prompt, retrieved_context)
        
        # Evaluate the response
        evaluation = teacher_evaluate(prompt, retrieved_context, response)
        
        # Save the results
        save_results(model_name, prompt, retrieved_context, response, evaluation)
        
        return evaluation
    
    except Exception as e:
        st.error(f"Error in evaluating named model: {str(e)}")
        return None

# Example: When indexing data to Pinecone, generate embeddings using the specified model
def index_context_data(model_name, texts):
    try:
        index = pinecone_client.Index(os.getenv('PINECONE_INDEX_NAME'))
        for text in texts:
            embedding = generate_embedding(text)
            if embedding:
                index.upsert([
                    {
                        "id": f"{model_name}_{hash(text)}",
                        "values": embedding,
                        "metadata": {"text": text}
                    }
                ])
    except Exception as e:
        st.error(f"Error indexing data to Pinecone: {str(e)}")
def upload_model(file, username, model_type):
    # {{ edit_5: Modify upload_model to handle model_type }}
    model_id = f"{username}_model_{int(datetime.now().timestamp())}"
    if model_type == "custom":
        # Save the model file as needed
        model_path = os.path.join("models", f"{model_id}.bin")
        with open(model_path, "wb") as f:
            f.write(file.getbuffer())
        
        # Update user's models list
        users_collection.update_one(
            {"username": username},
            {"$push": {"models": {
                "model_id": model_id,
                "file_path": model_path,
                "uploaded_at": datetime.now(),
                "model_type": "custom"
            }}}
        )
        return f"Custom Model {model_id} uploaded successfully!"
    elif model_type == "named":
        # For Named Models, assume the model is managed externally (e.g., via Pinecone)
        users_collection.update_one(
            {"username": username},
            {"$push": {"models": {
                "model_id": model_id,
                "model_name": None,
                "file_path": None,
                "model_link": None,
                "uploaded_at": datetime.now(),
                "model_type": "named"
            }}}
        )
        return f"Named Model {model_id} registered successfully!"
    else:
        return "Invalid model type specified."

# Function to save results to MongoDB
def save_results(username, model, prompt, context, response, evaluation):  # {{ edit_29: Add 'username' parameter }}
    result = {
        "username": username,  # Use the passed 'username' parameter
        "model_id": model['model_id'],  # {{ edit_19: Associate results with 'model_id' }}
        "model_name": model.get('model_name'),
        "model_type": model.get('model_type', 'custom'),  # {{ edit_20: Include 'model_type' in results }}
        "prompt": prompt,
        "context": context,
        "response": response,
        "evaluation": evaluation,
        "timestamp": datetime.now()
    }
    results_collection.insert_one(result)

# Function for teacher model evaluation
def teacher_evaluate(prompt, context, response):
    try:
        evaluation_prompt = f"""
        Evaluate the following response based on the given prompt and context. 
        Rate each factor on a scale of 0 to 1, where 1 is the best (or least problematic for negative factors like Hallucination and Bias).
        Please provide scores with two decimal places, and avoid extreme scores of exactly 0 or 1 unless absolutely necessary.

        Prompt: {prompt}
        Context: {context}
        Response: {response}

        Factors to evaluate:
        1. Accuracy: How factually correct is the response?
        2. Hallucination: To what extent does the response contain made-up information? (Higher score means less hallucination)
        3. Groundedness: How well is the response grounded in the given context and prompt?
        4. Relevance: How relevant is the response to the prompt?
        5. Recall: How much of the relevant information from the context is included in the response?
        6. Precision: How precise and focused is the response in addressing the prompt?
        7. Consistency: How consistent is the response with the given information and within itself?
        8. Bias Detection: To what extent is the response free from bias? (Higher score means less bias)

        Provide the evaluation as a JSON object. Each factor should be a key mapping to an object containing 'score' and 'explanation'. 
        Do not include any additional text, explanations, or markdown formatting.
        """

        evaluation_response = openai_client.chat.completions.create(
            model="gpt-4o-mini",  
            messages=[
                {"role": "system", "content": "You are an expert evaluator of language model responses."},
                {"role": "user", "content": evaluation_prompt}
            ]
        )

        content = evaluation_response.choices[0].message.content.strip()

        # Ensure the response starts and ends with curly braces
        if not (content.startswith("{") and content.endswith("}")):
            st.error("Teacher evaluation did not return a valid JSON object.")
            st.error(f"Response content: {content}")
            return None

        try:
            evaluation = json.loads(content)
            return evaluation
        except json.JSONDecodeError as e:
            st.error(f"Error decoding evaluation response: {str(e)}")
            st.error(f"Response content: {content}")
            return None

    except Exception as e:
        st.error(f"Error in teacher evaluation: {str(e)}")
        return None

# Function to generate dummy data for demonstration
def generate_dummy_data():
    dates = pd.date_range(end=datetime.now(), periods=30).tolist()
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'Consistency', 'Bias']
    data = {
        'Date': dates * len(metrics),
        'Metric': [metric for metric in metrics for _ in range(len(dates))],
        'Value': np.random.rand(len(dates) * len(metrics)) * 100
    }
    return pd.DataFrame(data)

# Function to count tokens
def count_tokens(text: str) -> int:
    return len(tokenizer.encode(text))

# Sidebar Navigation
st.sidebar.title("LLM Evaluation System")

# Session state
if 'user' not in st.session_state:
    st.session_state.user = None

# Authentication
if not st.session_state.user:
    auth_option = st.sidebar.radio("Choose an option", ["Login", "Signup"])
    
    username = st.sidebar.text_input("Username")
    password = st.sidebar.text_input("Password", type="password")
    
    if auth_option == "Login":
        if st.sidebar.button("Login"):
            if authenticate(username, password):
                st.session_state.user = username
                st.rerun()
            else:
                st.sidebar.error("Invalid username or password")
    else:
        if st.sidebar.button("Signup"):
            if signup(username, password):
                st.sidebar.success("Signup successful. Please login.")
            else:
                st.sidebar.error("Username already exists")
else:
    st.sidebar.success(f"Welcome, {st.session_state.user}!")
    if st.sidebar.button("Logout"):
        st.session_state.user = None
        st.rerun()

    # Add Clear Results Database button
    if st.sidebar.button("Clear Results Database"):
        if clear_results_database():  # {{ edit_fix: Calling the newly defined clear_results_database function }}
            st.sidebar.success("Results database cleared successfully!")
        else:
            st.sidebar.error("Failed to clear results database.")

# App content
if st.session_state.user:
    app_mode = st.sidebar.selectbox("Choose the section", ["Dashboard", "Model Upload", "Evaluation", "Prompt Testing", "Manage Models", "History"])  # {{ edit_add: Added "History" to the sidebar navigation }}

    if app_mode == "Dashboard":
        st.title("Dashboard")
        st.write("### Real-time Metrics and Performance Insights")
        
        # Fetch the user from the database
        user = users_collection.find_one({"username": st.session_state.user})
        if user is None:
            st.error("User not found in the database.")
            st.stop()
        user_models = user.get("models", [])
        
        if user_models:
            model_options = [model['model_name'] if model['model_name'] else model['model_id'] for model in user_models]
            selected_model = st.selectbox("Select Model to View Metrics", ["All Models"] + model_options)
        else:
            st.error("You have no uploaded models.")
            selected_model = "All Models"
        
        try:
            query = {"username": st.session_state.user}
            if selected_model != "All Models":
                query["model_name"] = selected_model
                if not selected_model:
                    query = {"username": st.session_state.user, "model_id": selected_model}
            results = list(results_collection.find(query))
            if results:
                df = pd.DataFrame(results)
                
                # Count tokens for prompt, context, and response
                df['prompt_tokens'] = df['prompt'].apply(count_tokens)
                df['context_tokens'] = df['context'].apply(count_tokens)
                df['response_tokens'] = df['response'].apply(count_tokens)
                
                # Calculate total tokens for each row
                df['total_tokens'] = df['prompt_tokens'] + df['context_tokens'] + df['response_tokens']
                
                metrics = ["Accuracy", "Hallucination", "Groundedness", "Relevance", "Recall", "Precision", "Consistency", "Bias Detection"]
                for metric in metrics:
                    df[metric] = df['evaluation'].apply(lambda x: x.get(metric, {}).get('score', 0) if x else 0) * 100

                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df['query_number'] = range(1, len(df) + 1)  # Add query numbers
                
                @st.cache_data
                def create_metrics_graph(df, metrics):
                    fig = px.line(
                        df, 
                        x='query_number',  # Use query numbers on x-axis
                        y=metrics, 
                        title='Metrics Over Queries',
                        labels={metric: f"{metric} (%)" for metric in metrics},
                        markers=True,
                        template='plotly_dark',
                    )
                    color_discrete_sequence = px.colors.qualitative.Dark24
                    for i, metric in enumerate(metrics):
                        fig.data[i].line.color = color_discrete_sequence[i % len(color_discrete_sequence)]
                        fig.data[i].marker.color = color_discrete_sequence[i % len(color_discrete_sequence)]
                    fig.update_layout(
                        xaxis_title="Query Number",
                        yaxis_title="Metric Score (%)",
                        legend_title="Metrics",
                        hovermode="x unified",
                        margin=dict(l=50, r=50, t=100, b=50),
                        height=700  # Increase the height of the graph
                    )
                    return fig
                
                fig = create_metrics_graph(df, metrics)

                st.plotly_chart(fig, use_container_width=True)

                # Latest Metrics
                st.subheader("Latest Metrics")
                latest_result = df.iloc[-1]  # Get the last row (most recent query)
                latest_metrics = {metric: latest_result[metric] for metric in metrics}

                cols = st.columns(4)
                for i, (metric, value) in enumerate(latest_metrics.items()):
                    with cols[i % 4]:
                        color = 'green' if value >= 75 else 'orange' if value >= 50 else 'red'
                        st.metric(label=metric, value=f"{value:.2f}%", delta=None)
                        st.progress(value / 100)

                # Detailed Data View
                st.subheader("Detailed Data View")

                # Calculate aggregate metrics
                total_spans = len(df)
                total_tokens = df['total_tokens'].sum()

                # Display aggregate metrics
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Total Spans", f"{total_spans:,}")
                with col2:
                    st.metric("Total Tokens", f"{total_tokens:,.2f}M" if total_tokens >= 1e6 else f"{total_tokens:,}")

                # Prepare the data for display
                display_data = []
                for _, row in df.iterrows():
                    display_row = {
                        "Prompt": row['prompt'][:50] + "...",  # Truncate long prompts
                        "Context": row['context'][:50] + "...",  # Truncate long contexts
                        "Response": row['response'][:50] + "...",  # Truncate long responses
                    }
                    # Add metrics to the display row
                    for metric in metrics:
                        display_row[metric] = row[metric]  # Store as float, not string
                    
                    display_data.append(display_row)

                # Convert to DataFrame for easy display
                display_df = pd.DataFrame(display_data)

                # Function to color cells based on score
                def color_cells(val):
                    if isinstance(val, float):
                        if val >= 80:
                            color = 'green'
                        elif val >= 60:
                            color = '#90EE90'  # Light green
                        else:
                            color = 'red'
                        return f'background-color: {color}; color: black'
                    return ''

                # Apply the styling only to metric columns
                styled_df = display_df.style.applymap(color_cells, subset=metrics)

                # Format metric columns as percentages
                for metric in metrics:
                    styled_df = styled_df.format({metric: "{:.2f}%"})

                # Display the table with custom styling
                st.dataframe(
                    styled_df.set_properties(**{
                        'color': 'white',
                        'border': '1px solid #ddd'
                    }).set_table_styles([
                        {'selector': 'th', 'props': [('background-color', '#4CAF50'), ('color', 'white')]},
                        {'selector': 'td', 'props': [('text-align', 'left')]},
                        # Keep background white for non-metric columns
                        {'selector': 'td:nth-child(-n+3)', 'props': [('background-color', 'white !important')]}
                    ]), 
                    use_container_width=True,
                    height=400  # Set a fixed height with scrolling
                )
                
                # Placeholders for future sections
                st.subheader("Worst Performing Slice Analysis")
                st.info("This section will show analysis of the worst-performing data slices.")
                
                st.subheader("UMAP Visualization")
                st.info("This section will contain UMAP visualizations for dimensionality reduction insights.")
            else:
                st.info("No evaluation results available for the selected model.")
        except Exception as e:
            st.error(f"Error fetching data from database: {e}")
            st.error("Detailed error information:")
            st.error(str(e))
            import traceback
            st.error(traceback.format_exc())

    elif app_mode == "Model Upload":
        st.title("Upload Your Model")
        model_type = st.radio("Select Model Type", ["Custom", "Named"])  # {{ edit_6: Select model type }}
        uploaded_file = st.file_uploader("Choose a model file", type=[".pt", ".h5", ".bin"]) if model_type == "custom" else None
        
        if st.button("Upload Model"):
            if model_type == "custom" and uploaded_file is not None:
                result = upload_model(uploaded_file, st.session_state.user, model_type="custom")
                st.success(result)
            elif model_type == "named":
                result = upload_model(None, st.session_state.user, model_type="named")
                st.success(result)
            else:
                st.error("Please upload a valid model file for Custom models.")

    elif app_mode == "Evaluation":
        st.title("Evaluate Your Model")
        st.write("### Select Model and Evaluation Metrics")
        
        # Fetch the user from the database
        user = users_collection.find_one({"username": st.session_state.user})
        if user is None:
            st.error("User not found in the database.")
            st.stop()
        user_models = user.get("models", [])
        
        if not user_models:
            st.error("You have no uploaded models. Please upload a model first.")
        else:
            # {{ edit_1: Display model_name instead of model_id }}
            model_identifier = st.selectbox(
                "Choose a Model to Evaluate",
                [model['model_name'] if model['model_name'] else model['model_id'] for model in user_models]
            )
            
            # {{ edit_2: Remove metrics selection and set fixed metrics }}
            fixed_metrics = ["Accuracy", "Hallucination", "Groundedness", "Relevance", "Recall", "Precision", "Consistency", "Bias Detection"]
            st.write("### Evaluation Metrics")
            st.write(", ".join(fixed_metrics))
            
            # Modify the evaluation function call to use fixed_metrics
            if st.button("Start Evaluation"):
                with st.spinner("Evaluation in progress..."):
                    # {{ edit_3: Use fixed_metrics instead of user-selected metrics }}
                    results = evaluate_model(model_identifier, fixed_metrics, st.session_state.user)
                    # Fetch the current model document
                    current_model = next((m for m in user_models if (m['model_name'] == model_identifier) or (m['model_id'] == model_identifier)), None)
                    if current_model:
                        save_results(st.session_state.user, current_model, prompt, context, response, results)  # {{ edit_21: Pass current_model to save_results }}
                        st.success("Evaluation Completed!")
                        st.json(results)
                    else:
                        st.error("Selected model not found.")

    elif app_mode == "Prompt Testing":
        st.title("Prompt Testing")
        
        model_selection_option = st.radio("Select Model Option:", ["Choose Existing Model", "Add New Model"])
        
        if model_selection_option == "Choose Existing Model":
            user = users_collection.find_one({"username": st.session_state.user})
            user_models = user.get("models", [])
            
            if not user_models:
                st.error("You have no uploaded models. Please upload a model first.")
            else:
                model_options = [
                    f"{model['model_name']} ({model.get('model_type', 'Unknown').capitalize()})" 
                    for model in user_models
                ]
                selected_model = st.selectbox("Select a Model for Testing", model_options)
                
                model_name = selected_model.split(" (")[0]
                model_type = selected_model.split(" (")[1].rstrip(")")
        else:
            # Code for adding a new model (unchanged)
            ...

        st.subheader("Input for Model Testing")
        context_dataset = st.text_area("Enter Context Dataset (txt):", height=200)
        questions_json = st.text_area("Enter Questions (JSON format):", height=200)
        
        if st.button("Run Test"):
            if not model_name:
                st.error("Please select or add a valid Model.")
            elif not context_dataset or not questions_json:
                st.error("Please provide both context dataset and questions JSON.")
            else:
                try:
                    questions = json.loads(questions_json)
                    selected_model = next(
                        (m for m in user_models if m['model_name'] == model_name),
                        None
                    )
                    if selected_model:
                        with st.spinner("Starting evaluations..."):
                            evaluation_thread = threading.Thread(
                                target=run_custom_evaluations, 
                                args=(context_dataset, questions, selected_model, st.session_state.user)
                            )
                            evaluation_thread.start()
                            st.success("Evaluations are running in the background. You can navigate away or close the site.")
                    else:
                        st.error("Selected model not found.")
                except json.JSONDecodeError:
                    st.error("Invalid JSON format for questions. Please check your input.")

    elif app_mode == "Manage Models":
        st.title("Manage Your Models")
        # Fetch the user from the database
        user = users_collection.find_one({"username": st.session_state.user})
        if user is None:
            st.error("User not found in the database.")
            st.stop()
        user_models = user.get("models", [])
        
        # Update existing models to ensure they have a model_type
        for model in user_models:
            if 'model_type' not in model:
                model['model_type'] = 'simple'  # Default to 'simple' for existing models
        users_collection.update_one(
            {"username": st.session_state.user},
            {"$set": {"models": user_models}}
        )
        
        st.subheader("Add a New Model")
        model_type = st.radio("Select Model Type:", ["Simple Model", "Custom Model"])
        
        if model_type == "Simple Model":
            new_model_name = st.text_input("Enter New Model Name:")
            if st.button("Add Simple Model"):
                if new_model_name:
                    model_id = f"{st.session_state.user}_model_{int(datetime.now().timestamp())}"
                    users_collection.update_one(
                        {"username": st.session_state.user},
                        {"$push": {"models": {
                            "model_id": model_id,
                            "model_name": new_model_name,
                            "model_type": "simple",
                            "file_path": None,
                            "model_link": None,
                            "uploaded_at": datetime.now()
                        }}}
                    )
                    st.success(f"Simple Model '{new_model_name}' added successfully as {model_id}!")
                else:
                    st.error("Please enter a valid model name.")
        
        else:  # Custom Model
            custom_model_options = ["gpt-4o", "gpt-4o-mini"]
            selected_custom_model = st.selectbox("Select Custom Model:", custom_model_options)
            
            if st.button("Add Custom Model"):
                model_id = f"{st.session_state.user}_model_{int(datetime.now().timestamp())}"
                users_collection.update_one(
                    {"username": st.session_state.user},
                    {"$push": {"models": {
                        "model_id": model_id,
                        "model_name": selected_custom_model,
                        "model_type": "custom",
                        "file_path": None,
                        "model_link": None,
                        "uploaded_at": datetime.now()
                    }}}
                )
                st.success(f"Custom Model '{selected_custom_model}' added successfully as {model_id}!")
        
        st.markdown("---")
        
        if user_models:
            st.subheader("Your Models")
            for model in user_models:
                st.markdown(f"**Model ID:** {model['model_id']}")
                st.write(f"**Model Type:** {model.get('model_type', 'simple').capitalize()}")
                if model.get("model_name"):
                    st.write(f"**Model Name:** {model['model_name']}")
                if model.get("file_path"):
                    st.write(f"**File Path:** {model['file_path']}")
                st.write(f"**Uploaded at:** {model['uploaded_at']}")
                
                # Add delete option
                if st.button(f"Delete {model['model_id']}"):
                    # Delete the model file if exists and it's a Custom model
                    if model['file_path'] and os.path.exists(model['file_path']):
                        os.remove(model['file_path'])
                    # Remove model from user's models list
                    users_collection.update_one(
                        {"username": st.session_state.user},
                        {"$pull": {"models": {"model_id": model['model_id']}}}
                    )
                    st.success(f"Model {model['model_id']} deleted successfully!")
        else:
            st.info("You have no uploaded models.")

    elif app_mode == "History":  # {{ edit_add: Enhanced History UI }}
        st.title("History")
        st.write("### Your Evaluation History")
        
        try:
            # Fetch all evaluation results for the current user from MongoDB
            user_results = list(results_collection.find({"username": st.session_state.user}).sort("timestamp", -1))
            
            if user_results:
                # Convert results to a pandas DataFrame
                df = pd.DataFrame(user_results)
                
                # Normalize the evaluation JSON into separate columns
                eval_df = df['evaluation'].apply(pd.Series)
                for metric in ["Accuracy", "Hallucination", "Groundedness", "Relevance", "Recall", "Precision", "Consistency", "Bias Detection"]:
                    if metric in eval_df.columns:
                        df[metric + " Score"] = eval_df[metric].apply(lambda x: x.get('score', 0) * 100 if isinstance(x, dict) else 0)
                        df[metric + " Explanation"] = eval_df[metric].apply(lambda x: x.get('explanation', '') if isinstance(x, dict) else '')
                    else:
                        df[metric + " Score"] = 0
                        df[metric + " Explanation"] = ""
                
                # Select relevant columns to display
                display_df = df[[
                    "timestamp", "model_name", "prompt", "context", "response", 
                    "Accuracy Score", "Hallucination Score", "Groundedness Score",
                    "Relevance Score", "Recall Score", "Precision Score",
                    "Consistency Score", "Bias Detection Score"
                ]]
                
                # Rename columns for better readability
                display_df = display_df.rename(columns={
                    "timestamp": "Timestamp",
                    "model_name": "Model Name",
                    "prompt": "Prompt",
                    "context": "Context",
                    "response": "Response",
                    "Accuracy Score": "Accuracy (%)",
                    "Hallucination Score": "Hallucination (%)",
                    "Groundedness Score": "Groundedness (%)",
                    "Relevance Score": "Relevance (%)",
                    "Recall Score": "Recall (%)",
                    "Precision Score": "Precision (%)",
                    "Consistency Score": "Consistency (%)",
                    "Bias Detection Score": "Bias Detection (%)"
                })
                
                # Convert timestamp to a readable format
                display_df['Timestamp'] = pd.to_datetime(display_df['Timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')
                
                st.subheader("Evaluation Results")
                
                # Display the DataFrame with enhanced styling
                st.dataframe(
                    display_df.style.set_properties(**{
                        'background-color': '#f0f8ff',
                        'color': '#333',
                        'border': '1px solid #ddd'
                    }).set_table_styles([
                        {'selector': 'th', 'props': [('background-color', '#f5f5f5'), ('text-align', 'center')]},
                        {'selector': 'td', 'props': [('text-align', 'center'), ('vertical-align', 'top')]}
                    ]).format({
                        "Accuracy (%)": "{:.2f}",
                        "Hallucination (%)": "{:.2f}",
                        "Groundedness (%)": "{:.2f}",
                        "Relevance (%)": "{:.2f}",
                        "Recall (%)": "{:.2f}",
                        "Precision (%)": "{:.2f}",
                        "Consistency (%)": "{:.2f}",
                        "Bias Detection (%)": "{:.2f}"
                    }), use_container_width=True
                )
                
            else:
                st.info("You have no evaluation history yet.")
        
        except Exception as e:
            st.error(f"Error fetching history data: {e}")

# Add a footer
st.sidebar.markdown("---")
st.sidebar.info("LLM Evaluation System - v0.2")

# Add this function to handle custom model evaluations
def run_custom_evaluations(context_dataset, questions, selected_model, username):
    try:
        model_name = selected_model['model_name']
        for question in questions:
            answer = run_model(model_name, context_dataset, question)
            evaluation = teacher_evaluate(question, context_dataset, answer)
            save_results(username, selected_model, question, context_dataset, answer, evaluation)
    except Exception as e:
        print(f"Error in custom evaluation: {str(e)}")