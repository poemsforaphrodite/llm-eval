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
from streamlit_plotly_events import plotly_events  # {{ edit_1: Re-added import for 'plotly_events' }}
from pinecone import Pinecone, ServerlessSpec  # {{ edit_10: Import Pinecone classes instead of pinecone }}

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
def save_results(model, prompt, context, response, evaluation):
    result = {
        "username": st.session_state.user,
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
            model="gpt-4o-mini",  # Corrected model name
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
        st.experimental_rerun()

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
        
        # {{ edit_7: Add model selection dropdown }}
        user = users_collection.find_one({"username": st.session_state.user})
        user_models = user.get("models", [])
        
        if user_models:
            model_options = [model['model_name'] if model['model_name'] else model['model_id'] for model in user_models]
            selected_model = st.selectbox("Select Model to View Metrics", ["All Models"] + model_options)
        else:
            st.error("You have no uploaded models.")
            selected_model = "All Models"
        
        # Fetch evaluation results from MongoDB
        try:
            query = {"username": st.session_state.user}
            if selected_model != "All Models":
                query["model_name"] = selected_model
                # If model_name is None, fall back to model_id
                if not selected_model:
                    # Assuming model_id is unique and known
                    query = {"username": st.session_state.user, "model_id": selected_model}
            results = list(results_collection.find(query))
            if results:
                # Convert to DataFrame
                df = pd.DataFrame(results)
                
                # Extract evaluation metrics into separate columns and convert to percentage
                metrics = ["Accuracy", "Hallucination", "Groundedness", "Relevance", "Recall", "Precision", "Consistency", "Bias Detection"]
                for metric in metrics:
                    df[metric] = df['evaluation'].apply(lambda x: x.get(metric, {}).get('score', 0) * 100)
                
                # Convert timestamp to datetime
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                
                # {{ edit_8: Cache the graph creation to ensure it's only generated once }}
                @st.cache_data
                def create_metrics_graph(df, metrics):
                    fig = px.line(
                        df, 
                        x='timestamp', 
                        y=metrics, 
                        title='Metrics Over Time',
                        labels={metric: f"{metric} (%)" for metric in metrics},
                        markers=True,
                        template='plotly_dark'
                    )
                    # Customize colors for each metric
                    color_discrete_sequence = px.colors.qualitative.Dark24
                    for i, metric in enumerate(metrics):
                        fig.data[i].line.color = color_discrete_sequence[i % len(color_discrete_sequence)]
                    return fig

                fig = create_metrics_graph(df, metrics)

                # Display the Plotly chart and capture click events
                clicked_points = plotly_events(fig, click_event=True, hover_event=False)
                
                #st.plotly_chart(fig, use_container_width=True)
                
                # Handle click events
                if clicked_points:
                    clicked_point = clicked_points[0]
                    clicked_timestamp = clicked_point['x']
                    
                    # Filter the DataFrame for the clicked timestamp
                    detailed_results = df[df['timestamp'] == pd.to_datetime(clicked_timestamp)]
                    
                    st.markdown(f"### Detailed Data for **{clicked_timestamp}**")
                    
                    # Display detailed data in a table
                    st.dataframe(
                        detailed_results[[
                            'model_name', 'prompt', 'context', 'response', 'Accuracy',
                            'Hallucination', 'Groundedness', 'Relevance', 'Recall',
                            'Precision', 'Consistency', 'Bias Detection', 'timestamp'
                        ]].style.format({
                            "Accuracy": "{:.2f}%",
                            "Hallucination": "{:.2f}%",
                            "Groundedness": "{:.2f}%",
                            "Relevance": "{:.2f}%",
                            "Recall": "{:.2f}%",
                            "Precision": "{:.2f}%",
                            "Consistency": "{:.2f}%",
                            "Bias Detection": "{:.2f}%"
                        }).background_gradient(cmap='Greens')
                    )
                
                # Latest metrics section continues...
                st.subheader("Latest Metrics")
                latest_result = df.sort_values(by='timestamp', ascending=False).iloc[0]
                latest_metrics = {
                    "Accuracy": latest_result["Accuracy"],
                    "Hallucination": latest_result["Hallucination"],
                    "Groundedness": latest_result["Groundedness"],
                    "Relevance": latest_result["Relevance"],
                    "Recall": latest_result["Recall"],
                    "Precision": latest_result["Precision"],
                    "Consistency": latest_result["Consistency"],
                    "Bias Detection": latest_result["Bias Detection"]
                }

                # Define the number of columns per row
                cols_per_row = 4
                metric_items = list(latest_metrics.items())
                
                for i in range(0, len(metric_items), cols_per_row):
                    cols = st.columns(cols_per_row)
                    for j, (metric, value) in enumerate(metric_items[i:i+cols_per_row]):
                        with cols[j]:
                            # Determine color based on value
                            if metric in ["Accuracy", "Hallucination", "Groundedness", "Relevance", "Recall", "Precision", "Consistency"]:
                                color = 'green' if value >= 75 else 'orange' if value >= 50 else 'red'
                            else:  # Bias Detection
                                color = 'green' if value >= 75 else 'red'
                            
                            # Display metric with colored progress bar
                            st.markdown(f"**{metric}**")
                            st.progress(value / 100)
                            st.markdown(f"<p style='color:{color};'>{value:.2f}%</p>", unsafe_allow_html=True)
                
                # Worst Performing Slice Analysis (placeholder)
                st.subheader("Worst Performing Slice Analysis")
                st.write("This section will show analysis of the worst-performing data slices.")
                
                # UMAP Visualization (placeholder)
                st.subheader("UMAP Visualization")
                st.write("This section will contain UMAP visualizations for dimensionality reduction insights.")
            else:
                st.info("No evaluation results available for the selected model.")
        except Exception as e:
            st.error(f"Error fetching data from database: {e}")

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
        
        user = users_collection.find_one({"username": st.session_state.user})
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
                        save_results(current_model, prompt, context, response, results)  # {{ edit_21: Pass current_model to save_results }}
                        st.success("Evaluation Completed!")
                        st.json(results)
                    else:
                        st.error("Selected model not found.")

    elif app_mode == "Prompt Testing":
        st.title("Prompt Testing")
        
        # {{ edit_6: Use model_name instead of model_id }}
        model_selection_option = st.radio("Select Model Option:", ["Choose Existing Model", "Add New Model"])
        
        if model_selection_option == "Choose Existing Model":
            user = users_collection.find_one({"username": st.session_state.user})
            user_models = user.get("models", [])
            
            if not user_models:
                st.error("You have no uploaded models. Please upload a model first.")
            else:
                # Display model_name instead of model_id
                model_name = st.selectbox("Select a Model for Testing", [model['model_name'] if model['model_name'] else model['model_id'] for model in user_models])
        else:
            # Option to enter model name or upload a link
            new_model_option = st.radio("Add Model By:", ["Enter Model Name", "Upload Model Link"])
            
            if new_model_option == "Enter Model Name":
                model_name_input = st.text_input("Enter New Model Name:")
                if st.button("Save Model Name"):
                    if model_name_input:
                        # {{ edit_3: Save the new model name to user's models }}
                        model_id = f"{st.session_state.user}_model_{int(datetime.now().timestamp())}"
                        users_collection.update_one(
                            {"username": st.session_state.user},
                            {"$push": {"models": {
                                "model_id": model_id,
                                "model_name": model_name_input,
                                "file_path": None,
                                "model_link": None,
                                "uploaded_at": datetime.now()
                            }}}
                        )
                        st.success(f"Model '{model_name_input}' saved successfully as {model_id}!")
                        model_name = model_name_input  # Use model_name instead of model_id
                    else:
                        st.error("Please enter a valid model name.")
            else:
                model_link = st.text_input("Enter Model Link:")
                if st.button("Save Model Link"):
                    if model_link:
                        # {{ edit_4: Save the model link to user's models }}
                        model_id = f"{st.session_state.user}_model_{int(datetime.now().timestamp())}"
                        users_collection.update_one(
                            {"username": st.session_state.user},
                            {"$push": {"models": {
                                "model_id": model_id,
                                "model_name": None,
                                "file_path": None,
                                "model_link": model_link,
                                "uploaded_at": datetime.now()
                            }}}
                        )
                        st.success(f"Model link saved successfully as {model_id}!")
                        model_name = model_id  # Use model_id if model_name is not available
                    else:
                        st.error("Please enter a valid model link.")
        
        # Two ways to provide prompts
        prompt_input_method = st.radio("Choose prompt input method:", ["Single JSON", "Batch Upload"])
        
        if prompt_input_method == "Single JSON":
            json_input = st.text_area("Enter your JSON input:")
            if json_input:
                try:
                    data = json.loads(json_input)
                    st.success("JSON parsed successfully!")
                    st.json(data)
                except json.JSONDecodeError:
                    st.error("Invalid JSON. Please check your input.")
        else:
            uploaded_file = st.file_uploader("Upload a JSON file with prompts, contexts, and responses", type="json")
            if uploaded_file is not None:
                try:
                    data = json.load(uploaded_file)
                    st.success("JSON file loaded successfully!")
                    st.json(data)
                except json.JSONDecodeError:
                    st.error("Invalid JSON file. Please check your file contents.")
        
        if st.button("Run Test"):
            if not model_name:
                st.error("Please select or add a valid Model.")
            elif not data:
                st.error("Please provide valid JSON input.")
            else:
                with st.spinner("Testing in progress..."):
                    # {{ edit_5: Append prompt testing data to the selected model using model_name }}
                    user = users_collection.find_one({"username": st.session_state.user})
                    models = user.get("models", [])
                    selected_model = next((m for m in models if (m['model_name'] == model_name) or (m['model_id'] == model_name)), None)
                    
                    if selected_model:
                        if selected_model.get("model_link"):
                            # Handle model link if necessary
                            pass  # Implement link handling if required
                        
                        if isinstance(data, list):
                            # Batch processing
                            for item in data:
                                st.subheader(f"Evaluation for prompt: {item['prompt'][:50]}...")
                                if 'response' not in item:
                                    item['response'] = generate_response(item['prompt'], item['context'])
                                evaluation = teacher_evaluate(item['prompt'], item['context'], item['response'])
                                save_results(selected_model, item['prompt'], item['context'], item['response'], evaluation)
                                
                                # {{ edit_22: Enhanced display of results with expandable sections }}
                                with st.expander("View Results"):
                                    st.markdown("**Model Response:**")
                                    st.write(item['response'])
                                    st.markdown("**Teacher Evaluation:**")
                                    # Create a more aesthetic table for evaluation results
                                    eval_df = pd.DataFrame([
                                        {"Metric": metric, "Score": f"{details['score']:.2f}", "Explanation": details['explanation']}
                                        for metric, details in evaluation.items()
                                    ])
                                    st.table(eval_df.style.set_properties(**{
                                        'background-color': '#f9f9f9',
                                        'color': '#333',
                                        'border': '1px solid #ddd'
                                    }).hide_index())
                                
                                st.markdown("---")
                        else:
                            # Single item processing
                            if 'response' not in data:
                                data['response'] = generate_response(data['prompt'], data['context'])
                            evaluation = teacher_evaluate(data['prompt'], data['context'], data['response'])
                            save_results(selected_model, data['prompt'], data['context'], data['response'], evaluation)
                            
                            # {{ edit_23: Enhanced display of single test results with expandable sections }}
                            with st.expander("View Results"):
                                st.subheader("Model Response:")
                                st.write(data['response'])
                                st.subheader("Teacher Evaluation:")
                                
                                # Create a more aesthetic table for evaluation results
                                eval_df = pd.DataFrame([
                                    {"Metric": metric, "Score": f"{details['score']:.2f}", "Explanation": details['explanation']}
                                    for metric, details in evaluation.items()
                                ])
                                st.table(eval_df.style.set_properties(**{
                                    'background-color': '#f9f9f9',
                                    'color': '#333',
                                    'border': '1px solid #ddd'
                                }).hide_index())
                        
                        st.success("Testing Completed and Results Saved!")
                    else:
                        st.error("Selected model not found.")

    elif app_mode == "Manage Models":
        st.title("Manage Your Models")
        user = users_collection.find_one({"username": st.session_state.user})
        user_models = user.get("models", [])
        
        # {{ edit_1: Add option to add a new model }}
        st.subheader("Add a New Model")
        add_model_option = st.radio("Add Model By:", ["Enter Model Name", "Upload Model Link"])
        
        if add_model_option == "Enter Model Name":
            new_model_name = st.text_input("Enter New Model Name:")
            if st.button("Add Model Name"):
                if new_model_name:
                    model_id = f"{st.session_state.user}_model_{int(datetime.now().timestamp())}"
                    users_collection.update_one(
                        {"username": st.session_state.user},
                        {"$push": {"models": {
                            "model_id": model_id,
                            "model_name": new_model_name,
                            "file_path": None,
                            "model_link": None,
                            "uploaded_at": datetime.now()
                        }}}
                    )
                    st.success(f"Model '{new_model_name}' added successfully as {model_id}!")
                else:
                    st.error("Please enter a valid model name.")
        else:
            new_model_link = st.text_input("Enter Model Link:")
            if st.button("Add Model Link"):
                if new_model_link:
                    model_id = f"{st.session_state.user}_model_{int(datetime.now().timestamp())}"
                    users_collection.update_one(
                        {"username": st.session_state.user},
                        {"$push": {"models": {
                            "model_id": model_id,
                            "model_name": None,
                            "file_path": None,
                            "model_link": new_model_link,
                            "uploaded_at": datetime.now()
                        }}}
                    )
                    st.success(f"Model link added successfully as {model_id}!")
                else:
                    st.error("Please enter a valid model link.")
        
        st.markdown("---")
        
        if user_models:
            st.subheader("Your Models")
            for model in user_models:
                st.markdown(f"**Model ID:** {model['model_id']}")
                st.write(f"**Model Type:** {model.get('model_type', 'custom').capitalize()}")  # {{ edit_14: Handle missing 'model_type' with default 'custom' }}
                if model.get("model_name"):
                    st.write(f"**Model Name:** {model['model_name']}")
                if model.get("model_link"):
                    st.write(f"**Model Link:** [Link]({model['model_link']})")
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

# Function to handle model upload (placeholder)
