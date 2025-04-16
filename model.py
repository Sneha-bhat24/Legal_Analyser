import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain.agents import initialize_agent, Tool, AgentExecutor
from langchain.agents.agent_types import AgentType
from langchain.llms import HuggingFacePipeline
from langchain.tools import BaseTool
from typing import Literal
import json
from datetime import datetime
import os
from PIL import Image
import io

# Page configuration
st.set_page_config(
    page_title="Legal Analysis System",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main {
        background-color: #f9f9f9;
    }
    .stTextInput > div > div > input {
        border-radius: 20px;
    }
    .user-message {
        background-color: #e6f7ff;
        padding: 15px;
        border-radius: 15px;
        margin: 10px 0;
        border-left: 5px solid #1E88E5;
    }
    .bot-message {
        background-color: #f0f0f0;
        padding: 15px;
        border-radius: 15px;
        margin: 10px 0;
        border-left: 5px solid #9E9E9E;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load the Granite model and tokenizer (cached to prevent reloading)"""
    model_name = "ibm-granite/granite-3.2-2b-instruct"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map=None,  # no device mapping — everything on CPU
    torch_dtype=torch.float32,  # CPU doesn't support bfloat16
    trust_remote_code=True
)
    
    return model, tokenizer, device

# Load model and tokenizer
model, tokenizer, device = load_model()

# Function to generate responses
def generate_response(prompt: str) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=1024,
        temperature=0.7,
        do_sample=True
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Custom tools definition
class LegalCaseSearchTool(BaseTool):
    name: Literal["legal_case_search"] = "legal_case_search"  # Added type annotation
    description: str = "Searches legal cases from a JSON database."  # Added type annotation

    def _run(self, query: str) -> str:
        try:
            with open('legal_case_data.json') as f:
                cases = json.load(f)
            results = [
                f"Case {c['pdf_name']}: {c['case_info']['Petitioner']} vs {c['case_info']['Respondent']}"
                for c in cases if query.lower() in c['case_info']['Judgment Summary'].lower()
            ][:3]
            return "\n".join(results) if results else "No matching cases found."
        except Exception as e:
            return f"Error searching cases: {str(e)}"

class ConstitutionReferenceTool(BaseTool):
    name: Literal["constitutional_reference"] = "constitutional_reference"
    description: str = "Finds references to Indian Constitution articles."
    
    def _run(self, article: int) -> str:
        try:
            # Check if file exists and create dummy data if it doesn't
            if not os.path.exists('constitution_of_india.json'):
                # Create dummy data for demo purposes
                dummy_data = [
                    {"article": 14, "description": "Equality before law"},
                    {"article": 21, "description": "Protection of life and personal liberty"}
                ]
                with open('constitution_of_india.json', 'w') as f:
                    json.dump(dummy_data, f)
            
            with open('constitution_of_india.json') as f:
                constitution = json.load(f)
            
            result = next(
                (f"Article {a['article']}: {a['description']}" for a in constitution if a['article'] == article),
                "Article not found."
            )
            
            return result
        
        except Exception as e:
            return f"Error accessing constitution: {str(e)}"

class ImageAnalysisTool(BaseTool):
    name: Literal["image_analysis"] = "image_analysis"
    description: str = "Analyzes legal documents or evidence in image format."
    
    def _run(self, image_data: str) -> str:
        try:
            # In a real implementation, you would process the image here
            # For now, we'll return a placeholder message
            return "Image analyzed. The document appears to be a legal contract with signatures on the final page."
        except Exception as e:
            return f"Error analyzing image: {str(e)}"

# Function to process uploaded images
def process_image(uploaded_file):
    if uploaded_file is not None:
        # Read the image file
        image_bytes = uploaded_file.getvalue()
        
        # Convert to PIL Image for processing
        image = Image.open(io.BytesIO(image_bytes))
        
        # Return image for display
        return image
    return None

# Format response for display
class LegalAnalysisFormatter:
    @staticmethod
    def format_response(response: str) -> str:
        return f"""
        🌟 **Legal Analysis Report** 🌟
        📅 {datetime.now().strftime('%A, %B %d, %Y %I:%M %p')}
        ==================================================
        {response}
        ==================================================
        🔚 **End of Report**
        """

def setup_agent():
    # Define tools
    legal_case_tool = LegalCaseSearchTool()
    constitution_tool = ConstitutionReferenceTool()
    image_tool = ImageAnalysisTool()

    tools = [
        Tool(name=legal_case_tool.name, func=legal_case_tool.run, description=legal_case_tool.description),
        Tool(name=constitution_tool.name, func=constitution_tool.run, description=constitution_tool.description),
        Tool(name=image_tool.name, func=image_tool.run, description=image_tool.description)
    ]

    # Granite text generation pipeline
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=1024,
        temperature=0.7,
        top_p=0.9,
    )

    llm = HuggingFacePipeline(pipeline=pipe)

    # Initialize agent (this itself is a runnable agent)
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True
    )

    return agent


def main():
    # Initialize session state for chat history if it doesn't exist
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    if 'agent' not in st.session_state:
        st.session_state.agent = setup_agent()
    
    # Title and header
    st.title("⚖️ Indian Legal Analysis System")
    
    # Chat container
    chat_container = st.container()
    
    # Input container (at the bottom)
    with st.container():
        col1, col2 = st.columns([6, 1])
        
        with col1:
            user_input = st.text_input("Ask anything...", key="user_query")
        
        with col2:
            uploaded_file = st.file_uploader("📎", type=["jpg", "jpeg", "png", "pdf"], label_visibility="collapsed")
    
    # Process user input
    if user_input or uploaded_file:
        # Add user message to chat history
        if user_input:
            st.session_state.chat_history.append({"role": "user", "content": user_input})
        
        # Process uploaded image if any
        if uploaded_file:
            image = process_image(uploaded_file)
            if image:
                # Add image to chat history
                st.session_state.chat_history.append({"role": "user", "content": f"[Image: {uploaded_file.name}]", "image": image})
                
                # Use the image analysis tool
                image_analysis = st.session_state.agent.tools[2].func("image_data")
                user_input = f"{user_input} [Regarding the uploaded image: {uploaded_file.name}]" if user_input else f"Analyze this legal document image: {uploaded_file.name}"
        
        # Get response from agent
        with st.spinner("Thinking..."):
            try:
                # Call the agent with the input
                raw_response = st.session_state.agent.run(user_input)
                
                # Format the response
                formatted_response = LegalAnalysisFormatter.format_response(raw_response)
                
                # Add bot response to chat history
                st.session_state.chat_history.append({"role": "assistant", "content": formatted_response})
                
            except Exception as e:
                error_message = f"❌ Error: {str(e)}"
                st.session_state.chat_history.append({"role": "assistant", "content": error_message})
    
    # Display chat history
    with chat_container:
        for message in st.session_state.chat_history:
            if message["role"] == "user":
                st.markdown(f'<div class="user-message">👤 You: {message["content"]}</div>', unsafe_allow_html=True)
                
                # Display image if present
                if "image" in message:
                    st.image(message["image"], caption=f"Uploaded: {message['content'].split(': ')[1].rstrip(']')}")
            
            else:  # assistant
                st.markdown(f'<div class="bot-message">🤖 Assistant: {message["content"]}</div>', unsafe_allow_html=True)
        
        # Add some space at the bottom
        st.markdown("<br>" * 3, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
