import json

def convert_json(input_file, output_file):
    # Read the input JSON file
    with open(input_file, 'r') as f:
        data = json.load(f)

    # Open the output file for writing
    with open(output_file, 'w') as f:
        # Process each entry in the original JSON
        for entry in data:
            # Create the new message structure
            messages = [
                {
                    "role": "system",
                    "content": "You are a teacher model evaluating the responses of a student AI model. Provide accurate assessments based on the given metrics."
                },
                {
                    "role": "user",
                    "content": entry["prompt"]
                },
                {
                    "role": "assistant",
                    "content": f"Student response: {entry['response']}\n\nEvaluation:\nAccuracy: {entry['Accuracy']}\nHallucination: {entry['Hallucination']}\nGroundedness: {entry['Groundedness']}\nRelevance: {entry['Relevance']}\nRecall: {entry['Recall']}\nPrecision: {entry['Precision']}\nConsistency: {entry['Consistency']}\nBias Detection: {entry['Bias Detection']}"
                }
            ]

            # Create the new JSON object
            new_entry = {"messages": messages}

            # Write the new JSON object to the output file
            json.dump(new_entry, f)
            f.write('\n')  # Add a newline for readability

# Usage
input_file = 'teacher.json'  # Replace with your input file name
output_file = 'converted_teacher.json'  # Replace with your desired output file name

convert_json(input_file, output_file)
print(f"Conversion complete. Output written to {output_file}")