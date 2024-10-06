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
from bson.objectid import ObjectId
import traceback  # Add this import at the top of your file
import umap
import plotly.graph_objs as go
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import plotly.colors as plc
import uuid
import time  # Add this import at the top of your file

# Add this helper function at the beginning of your file
def extract_prompt_text(prompt):
    if isinstance(prompt, dict):
        return prompt.get('prompt', '')
    elif isinstance(prompt, str):
        return prompt
    else:
        return str(prompt)

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

# Remove or comment out this line if it exists
# openai_client = OpenAI()

# Instead, use the openai_client from runner.py
from runner import openai_client

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
    
# Add this function to update the context for a model
def update_model_context(username, model_id, context):
    users_collection.update_one(
        {"username": username, "models.model_id": model_id},
        {"$set": {"models.$.context": context}}
    )


# Function to clear the results database
def clear_results_database(username, model_identifier=None):
    try:
        if model_identifier:
            # Clear results for the specific model
            results_collection.delete_many({
                "username": username,
                "$or": [
                    {"model_name": model_identifier},
                    {"model_id": model_identifier}
                ]
            })
        else:
            # Clear all results for the user
            results_collection.delete_many({"username": username})
        return True
    except Exception as e:
        st.error(f"Error clearing results database: {str(e)}")
        return False

# Function to generate embeddings using the specified model
def generate_embedding(text):
    try:
        embedding_response = openai_client.embeddings.create(
            model="text-embedding-3-large",  # {{ edit_3: Use the specified embedding model }}
            input=text
        )
        embedding = embedding_response.data[0].embedding
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

# Function to chunk text
def chunk_text(text, max_tokens=500):
    tokens = tokenizer.encode(text)
    chunks = []
    current_chunk = []
    current_length = 0

    for token in tokens:
        if current_length + 1 > max_tokens:
            chunks.append(tokenizer.decode(current_chunk))
            current_chunk = []
            current_length = 0
        current_chunk.append(token)
        current_length += 1

    if current_chunk:
        chunks.append(tokenizer.decode(current_chunk))

    return chunks

# Function to upload context to Pinecone
def upload_context_to_pinecone(context, username, model_name):
    chunks = chunk_text(context)
    index = pinecone_client.Index(os.getenv('PINECONE_INDEX_NAME'))
    
    namespace = f"{username}_{model_name}"  # Create a unique namespace for each user-model combination
    
    for chunk in chunks:
        embedding = generate_embedding(chunk)
        if embedding:
            index.upsert([
                {
                    "id": str(uuid.uuid4()),
                    "values": embedding,
                    "metadata": {"text": chunk}
                }
            ], namespace=namespace)  # Use the namespace when upserting

# Function to retrieve relevant context from Pinecone
def retrieve_context_from_pinecone(prompt, username, model_name):
    index = pinecone_client.Index(os.getenv('PINECONE_INDEX_NAME'))
    prompt_embedding = generate_embedding(prompt)
    
    namespace = f"{username}_{model_name}"  # Use the same namespace format for retrieval
    
    if prompt_embedding:
        results = index.query(
            vector=prompt_embedding,
            top_k=5,
            namespace=namespace,  # Use the namespace when querying
            include_metadata=True
        )
        
        retrieved_context = " ".join([result.metadata['text'] for result in results.matches])
        return retrieved_context
    
    return ""

# Modify the run_custom_evaluations function
def run_custom_evaluations(data, selected_model, username):
    try:
        model_name = selected_model['model_name']
        model_id = selected_model['model_id']
        model_type = selected_model.get('model_type', 'Unknown').lower()
        
        if model_type == 'simple':
            # For simple models, data is already in the correct format
            test_cases = data
        else:
            # For custom models, data is split into context_dataset and questions
            context_dataset, questions = data
            
            # Upload context to Pinecone with user and model-specific namespace
            upload_context_to_pinecone(context_dataset, username, model_name)
            
            test_cases = [
                {
                    "prompt": extract_prompt_text(question),
                    "context": "",  # This will be filled with retrieved context
                    "response": ""  # This will be filled by the model
                }
                for question in questions
            ]
        
        for test_case in test_cases:
            prompt_text = test_case["prompt"]
            
            # For custom models, retrieve context from Pinecone using the user and model-specific namespace
            if model_type != 'simple':
                retrieved_context = retrieve_context_from_pinecone(prompt_text, username, model_name)
                test_case["context"] = retrieved_context
            
            context = test_case["context"]
            
            # Get the student model's response using runner.py
            try:
                answer = run_model(model_name, prompt_text)
                if answer is None or answer == "":
                    st.warning(f"No response received from the model for prompt: {prompt_text}")
                    answer = "No response received from the model."
            except Exception as model_error:
                st.error(f"Error running model for prompt: {prompt_text}")
                st.error(f"Error details: {str(model_error)}")
                answer = f"Error: {str(model_error)}"
            
            # Get the teacher's evaluation
            try:
                evaluation = teacher_evaluate(prompt_text, context, answer)
                if evaluation is None:
                    st.warning(f"No evaluation received for prompt: {prompt_text}")
                    evaluation = {"Error": "No evaluation received"}
            except Exception as eval_error:
                st.error(f"Error in teacher evaluation for prompt: {prompt_text}")
                st.error(f"Error details: {str(eval_error)}")
                evaluation = {"Error": str(eval_error)}
            
            # Save the results
            save_results(username, selected_model, prompt_text, context, answer, evaluation)
        
        st.success("Evaluation completed successfully!")
    except Exception as e:
        st.error(f"Error in custom evaluation: {str(e)}")
        st.error(f"Detailed error: {traceback.format_exc()}")

# Function for teacher model evaluation
def teacher_evaluate(prompt, context, response):
    try:
        evaluation_prompt = f"""
        Evaluate the following response based on the given prompt and context. 
        Rate each factor on a scale of 0 to 1, where 1 is the best (or least problematic for negative factors like Hallucination and Bias).
        Please provide scores with two decimal places, and avoid extreme scores of exactly 0 or 1 unless absolutely necessary.

        Context: {context}
        Prompt: {prompt}
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
            st.session_state['selected_model'] = selected_model  # Store the selected model in session state

            # Add delete dataset button
            if selected_model != "All Models":
                if st.button("Delete Dataset"):
                    if st.session_state['selected_model']:
                        if clear_results_database(st.session_state.user, st.session_state['selected_model']):
                            st.success(f"All evaluation results for {st.session_state['selected_model']} have been deleted.")
                            st.rerun()  # Rerun the app to refresh the dashboard
                        else:
                            st.error("Failed to delete the dataset. Please try again.")
                    else:
                        st.error("No model selected. Please select a model to delete its dataset.")
        else:
            st.error("You have no uploaded models.")
            selected_model = "All Models"
            st.session_state['selected_model'] = selected_model
        
        try:
            query = {"username": st.session_state.user}
            if selected_model != "All Models":
                query["model_name"] = selected_model
                if not selected_model:
                    query = {"username": st.session_state.user, "model_id": selected_model}
            results = list(results_collection.find(query))
            if results:
                df = pd.DataFrame(results)
                
                # Check if required columns exist
                required_columns = ['prompt', 'context', 'response', 'evaluation']
                missing_columns = [col for col in required_columns if col not in df.columns]
                if missing_columns:
                    st.error(f"Error: Missing columns in the data: {', '.join(missing_columns)}")
                    st.error("Please check the database schema and ensure all required fields are present.")
                    st.stop()

                # Extract prompt text if needed
                df['prompt'] = df['prompt'].apply(extract_prompt_text)

                # Safely count tokens for prompt, context, and response
                def safe_count_tokens(text):
                    if isinstance(text, str):
                        return count_tokens(text)
                    else:
                        return 0  # or some default value

                df['prompt_tokens'] = df['prompt'].apply(safe_count_tokens)
                df['context_tokens'] = df['context'].apply(safe_count_tokens)
                df['response_tokens'] = df['response'].apply(safe_count_tokens)
                
                # Calculate total tokens for each row
                df['total_tokens'] = df['prompt_tokens'] + df['context_tokens'] + df['response_tokens']
                
                # Safely extract evaluation metrics
                metrics = ["Accuracy", "Hallucination", "Groundedness", "Relevance", "Recall", "Precision", "Consistency", "Bias Detection"]
                for metric in metrics:
                    df[metric] = df['evaluation'].apply(lambda x: x.get(metric, {}).get('score', 0) if isinstance(x, dict) else 0) * 100

                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df['query_number'] = range(1, len(df) + 1)  # Add query numbers
                
                # Set the threshold for notifications
                notification_threshold = st.slider("Set Performance Threshold for Notifications (%)", min_value=0, max_value=100, value=50)

                # Define the metrics to check
                metrics_to_check = metrics  # Or allow the user to select specific metrics

                # Check for evaluations where any of the metrics are below the threshold
                low_performance_mask = df[metrics_to_check].lt(notification_threshold).any(axis=1)
                low_performing_evaluations = df[low_performance_mask]

                # Display Notifications
                if not low_performing_evaluations.empty:
                    st.warning(f"⚠️ You have {len(low_performing_evaluations)} evaluations with metrics below {notification_threshold}%.")
                    with st.expander("View Low-Performing Evaluations"):
                        # Display the low-performing evaluations in a table
                        display_columns = ['timestamp', 'model_name', 'prompt', 'response'] + metrics_to_check
                        low_perf_display_df = low_performing_evaluations[display_columns].copy()
                        low_perf_display_df['timestamp'] = low_perf_display_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
                        
                        # Apply styling to highlight low scores
                        def highlight_low_scores(val):
                            if isinstance(val, float):
                                if val < notification_threshold:
                                    return 'background-color: red; color: white'
                            return ''
                        
                        styled_low_perf_df = low_perf_display_df.style.applymap(highlight_low_scores, subset=metrics_to_check)
                        styled_low_perf_df = styled_low_perf_df.format({metric: "{:.2f}%" for metric in metrics_to_check})
                        
                        st.dataframe(
                            styled_low_perf_df.set_properties(**{
                                'text-align': 'left',
                                'border': '1px solid #ddd'
                            }).set_table_styles([
                                {'selector': 'th', 'props': [('background-color', '#333'), ('color', 'white')]},
                                {'selector': 'td', 'props': [('vertical-align', 'top')]}
                            ]), 
                            use_container_width=True
                        )
                else:
                    st.success("🎉 All your evaluations have metrics above the threshold!")

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
                latest_metrics = df[metrics].mean()  # Calculate the average of all metrics

                cols = st.columns(4)
                for i, (metric, value) in enumerate(latest_metrics.items()):
                    with cols[i % 4]:
                        color = 'green' if value >= 75 else 'orange' if value >= 50 else 'red'
                        st.metric(label=metric, value=f"{value:.2f}%", delta=None)
                        st.progress(value / 100)

                # Add an explanation for the metrics
                st.info("These metrics represent the average scores across all evaluations.")

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
                    prompt_text = extract_prompt_text(row.get('prompt', ''))
                    display_row = {
                        "Prompt": prompt_text[:50] + "..." if prompt_text else "N/A",
                        "Context": str(row.get('context', ''))[:50] + "..." if row.get('context') else "N/A",
                        "Response": str(row.get('response', ''))[:50] + "..." if row.get('response') else "N/A",
                    }
                    # Add metrics to the display row
                    for metric in metrics:
                        display_row[metric] = row.get(metric, 0)  # Use get() with a default value
                    
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
                
                # UMAP Visualization with Clustering
                st.subheader("UMAP Visualization with Clustering")

                if len(df) > 2:
                    # Allow user to select metrics to include
                    metrics = ['Accuracy', 'Hallucination', 'Groundedness', 'Relevance', 'Recall', 'Precision', 'Consistency', 'Bias Detection']
                    selected_metrics = st.multiselect("Select Metrics to Include in UMAP", metrics, default=metrics)

                    if len(selected_metrics) < 2:
                        st.warning("Please select at least two metrics for UMAP.")
                    else:
                        # Allow user to select number of dimensions
                        n_components = st.radio("Select UMAP Dimensions", [2, 3], index=1)

                        # Allow user to adjust UMAP parameters
                        n_neighbors = st.slider("n_neighbors", min_value=2, max_value=50, value=15)
                        min_dist = st.slider("min_dist", min_value=0.0, max_value=1.0, value=0.1, step=0.01)

                        # Prepare data for UMAP
                        X = df[selected_metrics].values

                        # Normalize the data
                        scaler = StandardScaler()
                        X_scaled = scaler.fit_transform(X)

                        # Perform UMAP dimensionality reduction
                        reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=n_components, random_state=42)
                        embedding = reducer.fit_transform(X_scaled)

                        # Allow user to select the number of clusters
                        num_clusters = st.slider("Select Number of Clusters", min_value=2, max_value=10, value=3)

                        # Perform KMeans clustering on the UMAP embeddings
                        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
                        cluster_labels = kmeans.fit_predict(embedding)

                        # Create a DataFrame with the UMAP results and cluster labels
                        umap_columns = [f'UMAP{i+1}' for i in range(n_components)]
                        umap_data = {col: embedding[:, idx] for idx, col in enumerate(umap_columns)}
                        umap_data['Cluster'] = cluster_labels
                        umap_data['Model'] = df['model_name']
                        umap_data['Prompt'] = df['prompt']
                        umap_data['Response'] = df['response']
                        umap_data['Timestamp'] = df['timestamp']
                        umap_df = pd.DataFrame(umap_data)

                        # Include selected metrics in umap_df for hover info
                        for metric in selected_metrics:
                            umap_df[metric] = df[metric]

                        # Prepare customdata for hovertemplate
                        customdata_columns = ['Model', 'Prompt', 'Cluster'] + selected_metrics
                        umap_df['customdata'] = umap_df[customdata_columns].values.tolist()

                        # Build hovertemplate
                        hovertemplate = '<b>Model:</b> %{customdata[0]}<br>' + \
                                        '<b>Prompt:</b> %{customdata[1]}<br>' + \
                                        '<b>Cluster:</b> %{customdata[2]}<br>'
                        for idx, metric in enumerate(selected_metrics):
                            hovertemplate += f'<b>{metric}:</b> %{{customdata[{idx+3}]:.2f}}<br>'
                        hovertemplate += '<extra></extra>'  # Hide trace info

                        # Define color palette for clusters
                        cluster_colors = plc.qualitative.Plotly
                        num_colors = len(cluster_colors)
                        if num_clusters > num_colors:
                            cluster_colors = plc.sample_colorscale('Rainbow', [n/(num_clusters-1) for n in range(num_clusters)])
                        else:
                            cluster_colors = cluster_colors[:num_clusters]

                        # Map cluster labels to colors
                        cluster_color_map = {label: color for label, color in zip(range(num_clusters), cluster_colors)}
                        umap_df['Color'] = umap_df['Cluster'].map(cluster_color_map)

                        # Create the UMAP plot
                        if n_components == 3:
                            # 3D plot
                            fig = go.Figure()

                            for cluster_label in sorted(umap_df['Cluster'].unique()):
                                cluster_data = umap_df[umap_df['Cluster'] == cluster_label]
                                fig.add_trace(go.Scatter3d(
                                    x=cluster_data['UMAP1'],
                                    y=cluster_data['UMAP2'],
                                    z=cluster_data['UMAP3'],
                                    mode='markers',
                                    name=f'Cluster {cluster_label}',
                                    marker=dict(
                                        size=5,
                                        color=cluster_data['Color'],  # Color according to cluster
                                        opacity=0.8,
                                        line=dict(width=0.5, color='white')
                                    ),
                                    customdata=cluster_data['customdata'],
                                    hovertemplate=hovertemplate
                                ))

                            fig.update_layout(
                                title='3D UMAP Visualization with Clustering',
                                scene=dict(
                                    xaxis_title='UMAP Dimension 1',
                                    yaxis_title='UMAP Dimension 2',
                                    zaxis_title='UMAP Dimension 3'
                                ),
                                hovermode='closest',
                                template='plotly_dark',
                                height=800,
                                legend_title='Clusters'
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            # 2D plot
                            fig = go.Figure()

                            for cluster_label in sorted(umap_df['Cluster'].unique()):
                                cluster_data = umap_df[umap_df['Cluster'] == cluster_label]
                                fig.add_trace(go.Scatter(
                                    x=cluster_data['UMAP1'],
                                    y=cluster_data['UMAP2'],
                                    mode='markers',
                                    name=f'Cluster {cluster_label}',
                                    marker=dict(
                                        size=8,
                                        color=cluster_data['Color'],  # Color according to cluster
                                        opacity=0.8,
                                        line=dict(width=0.5, color='white')
                                    ),
                                    customdata=cluster_data['customdata'],
                                    hovertemplate=hovertemplate
                                ))

                            fig.update_layout(
                                title='2D UMAP Visualization with Clustering',
                                xaxis_title='UMAP Dimension 1',
                                yaxis_title='UMAP Dimension 2',
                                hovermode='closest',
                                template='plotly_dark',
                                height=800,
                                legend_title='Clusters'
                            )
                            st.plotly_chart(fig, use_container_width=True)

                        # Selectable Data Points
                        st.subheader("Cluster Analysis")

                        # Show cluster counts
                        cluster_counts = umap_df['Cluster'].value_counts().sort_index().reset_index()
                        cluster_counts.columns = ['Cluster', 'Number of Points']
                        st.write("### Cluster Summary")
                        st.dataframe(cluster_counts)

                        # Allow user to select clusters to view details
                        selected_clusters = st.multiselect("Select Clusters to View Details", options=sorted(umap_df['Cluster'].unique()), default=sorted(umap_df['Cluster'].unique()))

                        if selected_clusters:
                            selected_data = umap_df[umap_df['Cluster'].isin(selected_clusters)]
                            st.write("### Details of Selected Clusters")
                            st.dataframe(selected_data[['Model', 'Prompt', 'Response', 'Cluster'] + selected_metrics])
                        else:
                            st.info("Select clusters to view their details.")

                        st.info("""
                        **UMAP Visualization with Clustering**

                        This visualization includes clustering of the evaluation data points in the UMAP space.

                        **Features:**

                        - **Clustering Algorithm**: KMeans clustering is applied on the UMAP embeddings.
                        - **Cluster Selection**: Choose the number of clusters to identify patterns in the data.
                        - **Color Coding**: Each cluster is represented by a distinct color in the plot.
                        - **Interactive Exploration**: Hover over points to see detailed information, including the cluster label.
                        - **Cluster Analysis**: View summary statistics and details of selected clusters.

                        **Instructions:**

                        - **Select Metrics**: Choose which evaluation metrics to include in the UMAP calculation.
                        - **Adjust UMAP Parameters**: Fine-tune `n_neighbors` and `min_dist` for clustering granularity.
                        - **Choose Number of Clusters**: Use the slider to set how many clusters to identify.
                        - **Interact with the Plot**: Hover and click on clusters to explore data points.

                        **Interpreting Clusters:**

                        - **Cluster Composition**: Clusters group evaluations with similar metric profiles.
                        - **Model Performance**: Analyze clusters to identify strengths and weaknesses of models.
                        - **Data Patterns**: Use clustering to uncover hidden patterns in your evaluation data.

                        **Tips:**

                        - Experiment with different numbers of clusters to find meaningful groupings.
                        - Adjust UMAP parameters to see how the clustering changes with different embeddings.
                        - Use the cluster details to investigate specific evaluations and prompts.

                        Enjoy exploring your evaluation data with clustering!
                        """)
                else:
                    st.info("Not enough data for UMAP visualization. Please run more evaluations.")

                # Worst Performing Slice Analysis
                st.subheader("Worst Performing Slice Analysis")

                # Allow the user to select metrics to analyze
                metrics = ['Accuracy', 'Hallucination', 'Groundedness', 'Relevance', 'Recall', 'Precision', 'Consistency', 'Bias Detection']
                selected_metrics = st.multiselect("Select Metrics to Analyze", metrics, default=metrics)

                if selected_metrics:
                    # Set a threshold for "poor performance"
                    threshold = st.slider("Performance Threshold (%)", min_value=0, max_value=100, value=50)

                    # Filter data where any of the selected metrics are below the threshold
                    mask = df[selected_metrics].lt(threshold).any(axis=1)
                    worst_performing_df = df[mask]

                    if not worst_performing_df.empty:
                        st.write(f"Found {len(worst_performing_df)} evaluations below the threshold of {threshold}% in the selected metrics.")

                        # Display the worst-performing prompts and their metrics
                        st.write("### Worst Performing Evaluations")
                        display_columns = ['prompt', 'response'] + selected_metrics + ['timestamp']
                        worst_performing_display_df = worst_performing_df[display_columns].copy()
                        worst_performing_display_df['timestamp'] = worst_performing_display_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
                        
                        # Apply styling to highlight low scores
                        def highlight_low_scores(val):
                            if isinstance(val, float):
                                if val < threshold:
                                    return 'background-color: red; color: white'
                            return ''
                        
                        styled_worst_df = worst_performing_display_df.style.applymap(highlight_low_scores, subset=selected_metrics)
                        styled_worst_df = styled_worst_df.format({metric: "{:.2f}%" for metric in selected_metrics})

                        st.dataframe(
                            styled_worst_df.set_properties(**{
                                'text-align': 'left',
                                'border': '1px solid #ddd'
                            }).set_table_styles([
                                {'selector': 'th', 'props': [('background-color', '#333'), ('color', 'white')]},
                                {'selector': 'td', 'props': [('vertical-align', 'top')]}
                            ]), 
                            use_container_width=True
                        )

                        # Analyze the worst-performing slices based on prompt characteristics
                        st.write("### Analysis by Prompt Length")

                        # Add a column for prompt length
                        worst_performing_df['Prompt Length'] = worst_performing_df['prompt'].apply(lambda x: len(x.split()))

                        # Define bins for prompt length ranges
                        bins = [0, 5, 10, 20, 50, 100, 1000]
                        labels = ['0-5', '6-10', '11-20', '21-50', '51-100', '100+']
                        worst_performing_df['Prompt Length Range'] = pd.cut(worst_performing_df['Prompt Length'], bins=bins, labels=labels, right=False)

                        # Group by 'Prompt Length Range' and calculate average metrics
                        group_metrics = worst_performing_df.groupby('Prompt Length Range')[selected_metrics].mean().reset_index()

                        # Display the average metrics per prompt length range
                        st.write("#### Average Metrics per Prompt Length Range")
                        group_metrics = group_metrics.sort_values('Prompt Length Range')
                        st.dataframe(group_metrics.style.format({metric: "{:.2f}%" for metric in selected_metrics}))

                        # Visualization of average metrics per prompt length range
                        st.write("#### Visualization of Metrics by Prompt Length Range")
                        melted_group_metrics = group_metrics.melt(id_vars='Prompt Length Range', value_vars=selected_metrics, var_name='Metric', value_name='Average Score')
                        fig = px.bar(
                            melted_group_metrics, 
                            x='Prompt Length Range', 
                            y='Average Score', 
                            color='Metric', 
                            barmode='group',
                            title='Average Metric Scores by Prompt Length Range',
                            labels={'Average Score': 'Average Score (%)'},
                            height=600
                        )
                        st.plotly_chart(fig, use_container_width=True)

                        # Further analysis: show counts of worst-performing evaluations per model
                        st.write("### Worst Performing Evaluations per Model")
                        model_counts = worst_performing_df['model_name'].value_counts().reset_index()
                        model_counts.columns = ['Model Name', 'Count of Worst Evaluations']
                        st.dataframe(model_counts)

                        # Allow user to download the worst-performing data
                        csv = worst_performing_df.to_csv(index=False)
                        st.download_button(
                            label="Download Worst Performing Data as CSV",
                            data=csv,
                            file_name='worst_performing_data.csv',
                            mime='text/csv',
                        )
                    else:
                        st.info("No evaluations found below the specified threshold.")
                else:
                    st.warning("Please select at least one metric to analyze.")

            else:
                st.info("No evaluation results available for the selected model.")
        except Exception as e:
            st.error(f"Error processing data from database: {str(e)}")
            st.error("Detailed error information:")
            st.error(traceback.format_exc())
            st.stop()

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
        
        # For simple models, we'll use a single JSON file
        if model_type.lower() == "simple":
            st.write("For simple models, please upload a single JSON file containing prompts, contexts, and responses.")
            json_file = st.file_uploader("Upload Test Data JSON", type=["json"])
            
            if json_file is not None:
                try:
                    test_data = json.load(json_file)
                    st.success("Test data JSON file uploaded successfully!")
                    
                    # Display a preview of the test data
                    st.write("Preview of test data:")
                    st.json(test_data[:3] if len(test_data) > 3 else test_data)
                    
                except json.JSONDecodeError:
                    st.error("Invalid JSON format. Please check your file.")
            else:
                test_data = None
        else:
            # For other model types, keep the existing separate inputs for context and questions
            context_file = st.file_uploader("Upload Context Dataset", type=["txt"])
            if context_file is not None:
                context_dataset = context_file.getvalue().decode("utf-8")
                st.success("Context file uploaded successfully!")
                # Upload context to Pinecone with user and model-specific namespace
                upload_context_to_pinecone(context_dataset, st.session_state.user, model_name)
            else:
                context_dataset = None

            questions_file = st.file_uploader("Upload Questions JSON", type=["json"])
            if questions_file is not None:
                questions_json = questions_file.getvalue().decode("utf-8")
                st.success("Questions file uploaded successfully!")
            else:
                questions_json = None
        
        if st.button("Run Test"):
            if not model_name:
                st.error("Please select or add a valid Model.")
            elif model_type.lower() == "simple" and test_data is None:
                st.error("Please upload a valid test data JSON file.")
            elif model_type.lower() != "simple" and (not context_dataset or not questions_json):
                st.error("Please provide both context dataset and questions JSON.")
            else:
                try:
                    selected_model = next(
                        (m for m in user_models if m['model_name'] == model_name),
                        None
                    )
                    if selected_model:
                        with st.spinner("Starting evaluations..."):
                            if model_type.lower() == "simple":
                                evaluation_thread = threading.Thread(
                                    target=run_custom_evaluations, 
                                    args=(test_data, selected_model, st.session_state.user)
                                )
                            else:
                                questions = json.loads(questions_json)
                                evaluation_thread = threading.Thread(
                                    target=run_custom_evaluations, 
                                    args=((context_dataset, questions), selected_model, st.session_state.user)
                                )
                            evaluation_thread.start()
                            st.success("Evaluations are running in the background. You can navigate away or close the site.")
                    else:
                        st.error("Selected model not found.")
                except json.JSONDecodeError:
                    st.error("Invalid JSON format. Please check your input.")

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
                    model_data = {
                        "model_id": model_id,
                        "model_name": new_model_name,
                        "model_type": "simple",
                        "file_path": None,
                        "model_link": None,
                        "uploaded_at": datetime.now(),
                        "context": None  # We'll update this when running evaluations
                    }
                    users_collection.update_one(
                        {"username": st.session_state.user},
                        {"$push": {"models": model_data}}
                    )
                    st.success(f"Model '{model_data['model_name']}' added successfully as {model_id}!")
                else:
                    st.error("Please enter a valid model name.")
        
        else:  # Custom Model
            custom_model_options = ["gpt-4o", "gpt-4o-mini"]
            selected_custom_model = st.selectbox("Select Custom Model:", custom_model_options)
            
            if st.button("Add Custom Model"):
                if selected_custom_model:
                    model_id = f"{st.session_state.user}_model_{int(datetime.now().timestamp())}"
                    model_data = {
                        "model_id": model_id,
                        "model_name": selected_custom_model,
                        "model_type": "custom",
                        "file_path": None,
                        "model_link": None,
                        "uploaded_at": datetime.now()
                    }
                    users_collection.update_one(
                        {"username": st.session_state.user},
                        {"$push": {"models": model_data}}
                    )
                    st.success(f"Custom Model '{selected_custom_model}' added successfully as {model_id}!")
                else:
                    st.error("Please select a valid custom model.")
        
        st.markdown("---")
        
        if user_models:
            st.subheader("Your Models")
            
            # Clear All Pinecone Data Button
            if st.button("Clear All Pinecone Data"):
                try:
                    index = pinecone_client.Index(os.getenv('PINECONE_INDEX_NAME'))
                    namespaces_cleared = []
                    for model in user_models:
                        model_name = model.get('model_name')
                        if model_name:
                            namespace = f"{st.session_state.user}_{model_name}"
                            index.delete(delete_all=True, namespace=namespace)
                            namespaces_cleared.append(model_name)
                    
                    if namespaces_cleared:
                        st.success(f"Pinecone data cleared for all models: {', '.join(namespaces_cleared)}")
                    else:
                        st.info("No namespaces found to clear.")
                except Exception as e:
                    st.error(f"Error clearing all Pinecone data: {str(e)}")
            
            st.markdown("---")
            
            for model in user_models:
                st.markdown(f"**Model ID:** {model['model_id']}")
                st.write(f"**Model Type:** {model.get('model_type', 'simple').capitalize()}")
                if model.get("model_name"):
                    st.write(f"**Model Name:** {model['model_name']}")
                if model.get("file_path"):
                    st.write(f"**File Path:** {model['file_path']}")
                st.write(f"**Uploaded at:** {model['uploaded_at']}")
                
                # Add delete option
                col1, col2 = st.columns(2)
                with col1:
                    if st.button(f"Delete {model['model_id']}", key=f"delete_{model['model_id']}"):
                        # Delete the model file if exists and it's a Custom model
                        if model['file_path'] and os.path.exists(model['file_path']):
                            os.remove(model['file_path'])
                        # Remove model from user's models list
                        users_collection.update_one(
                            {"username": st.session_state.user},
                            {"$pull": {"models": {"model_id": model['model_id']}}}
                        )
                        st.success(f"Model {model['model_id']} deleted successfully!")
                        time.sleep(2)  # Give user time to see the message
                        st.experimental_rerun()  # Refresh the page
                
                # Modify clear Pinecone database option
                with col2:
                    if st.button(f"Clear Pinecone for {model['model_id']}", key=f"clear_pinecone_{model['model_id']}"):
                        try:
                            index = pinecone_client.Index(os.getenv('PINECONE_INDEX_NAME'))
                            model_name = model.get('model_name')
                            if model_name:
                                namespace = f"{st.session_state.user}_{model_name}"
                                index.delete(delete_all=True, namespace=namespace)
                                st.success(f"Pinecone data cleared for {model['model_id']}!")
                            else:
                                st.error("Model name is missing. Cannot determine namespace.")
                        except Exception as e:
                            st.error(f"Error clearing Pinecone data for {model['model_id']}: {str(e)}")
                
                st.markdown("---")
            
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
                
                # Extract prompt text using the helper function
                df['prompt'] = df['prompt'].apply(extract_prompt_text)
                
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
                        {'selector': 'td', 'props': [('text-align', 'left'), ('vertical-align', 'top')]}
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