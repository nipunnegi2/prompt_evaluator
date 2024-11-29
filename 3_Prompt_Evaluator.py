
import streamlit as st
from evaluation import Evaluator
from scam_classifier import ClassifyScam
from langsmith import Client
import json

# Replace with your actual API keys
OPENAI_API_KEY = "dfre"
LANGSMITH_API_KEY = "fvgregfv"

# Initialize LangSmith client
client = Client(api_key=LANGSMITH_API_KEY)

st.set_page_config(
    page_title="✅ Prompt Evaluator",
    page_icon="✅",
)

st.write("# Prompt Evaluation Page")

# Initialize Evaluator and Classifier
evaluator = Evaluator(api_key=OPENAI_API_KEY)
classifier = ClassifyScam(api_key=OPENAI_API_KEY)

# Fetch LangSmith dataset
dataset_name = "Phoenix dataset"
datasets = client.list_datasets()
dataset = next((ds for ds in datasets if ds.name == dataset_name), None)

if not dataset:
    st.error(f"Dataset '{dataset_name}' not found.")
    st.stop()

# Fetch examples from the dataset
examples = client.list_examples(dataset_id=dataset.id)
if st.button("Run Evaluation"):
    with st.spinner("Evaluating..."):
        results = []
        for example in examples:
            # Debug dataset structure
            st.write("Example Inputs:", example.inputs)
            st.write("Example Outputs:", example.outputs)

            # Extract the image from inputs
            image_data = example.inputs.get("image")
            if not image_data:
                st.error("No 'image' key found in example.inputs. Ensure your dataset includes an 'image' key.")
                st.stop()

            # Run classifier and evaluator
            ai_response = classifier.invoke(image_data)  # Returns a ScamResponse object
            ai_response_dict = ai_response.dict()  # Convert ScamResponse to a dictionary
            eval_result = evaluator.eval_chain.invoke({
                'ai_generated_answer': ai_response_dict,
                'human_answer': example.outputs.get("reasoning", "No reasoning provided")
            })

            # Log results to a local file
            result_entry = {
                "dataset_id": str(dataset.id),  # Convert UUID to string
                "example_id": str(example.id),  # Convert UUID to string
                "input_image": image_data,
                "classification_result": ai_response_dict,  # Use the dictionary version
                "evaluation_result": eval_result.dict()  # Convert evaluation result to dictionary
            }
            with open("evaluation_results.json", "a") as f:
                json.dump(result_entry, f)
                f.write("\n")

            # Append result for display
            results.append(result_entry)

        # Display results on the Streamlit page
        st.json(results)
