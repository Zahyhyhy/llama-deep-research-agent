import asyncio
import streamlit as st
import requests
from typing import Dict, Any
from firecrawl import FirecrawlApp

# Set page configuration
st.set_page_config(
    page_title="LLaMA Deep Research Agent",
    page_icon="📘",
    layout="wide"
)

# Initialize session state for API keys if not exists
if "llama_api_key" not in st.session_state:
    st.session_state.llama_api_key = ""
if "llama_endpoint" not in st.session_state:
    st.session_state.llama_endpoint = ""
if "firecrawl_api_key" not in st.session_state:
    st.session_state.firecrawl_api_key = ""

# Sidebar for API keys
with st.sidebar:
    st.title("API Configuration")
    llama_endpoint = st.text_input(
        "LLaMA API Endpoint", 
        value=st.session_state.llama_endpoint,
        placeholder="e.g., https://api-inference.huggingface.co/models/meta-llama/Llama-3-3B-Instruct"
    )
    llama_api_key = st.text_input(
        "LLaMA API Key (optional)", 
        value=st.session_state.llama_api_key,
        type="password"
    )
    firecrawl_api_key = st.text_input(
        "Firecrawl API Key", 
        value=st.session_state.firecrawl_api_key,
        type="password"
    )
    
    if llama_endpoint:
        st.session_state.llama_endpoint = llama_endpoint
    if llama_api_key:
        st.session_state.llama_api_key = llama_api_key
    if firecrawl_api_key:
        st.session_state.firecrawl_api_key = firecrawl_api_key

# Main content
st.title("📘 LLaMA Deep Research Agent")
st.markdown("This LLaMA Agent performs deep research using Firecrawl and LLaMA API for reasoning and report generation.")

# Research topic input
research_topic = st.text_input("Enter your research topic:", placeholder="e.g., Latest developments in AI")

# LLaMA Completion Function
def call_llama(prompt: str) -> str:
    headers = {
        "Content-Type": "application/json"
    }
    if st.session_state.llama_api_key:
        headers["Authorization"] = f"Bearer {st.session_state.llama_api_key}"
    
    payload = {
        "inputs": prompt,  # For Hugging Face Inference API, use "inputs"
        "parameters": {
            "max_new_tokens": 1024,
            "temperature": 0.7
        }
    }

    response = requests.post(
        st.session_state.llama_endpoint,
        headers=headers,
        json=payload
    )
    
    if response.status_code == 200:
        result = response.json()
        return result[0]["generated_text"] if isinstance(result, list) else result.get("generated_text", "").strip()
    else:
        raise Exception(f"LLaMA API error: {response.status_code} - {response.text}")

# Deep research tool using Firecrawl
async def deep_research(query: str, max_depth: int, time_limit: int, max_urls: int) -> Dict[str, Any]:
    """
    Perform comprehensive web research using Firecrawl's deep research endpoint.
    """
    try:
        firecrawl_app = FirecrawlApp(api_key=st.session_state.firecrawl_api_key)
        params = {
            "maxDepth": max_depth,
            "timeLimit": time_limit,
            "maxUrls": max_urls
        }

        def on_activity(activity):
            st.write(f"[{activity['type']}] {activity['message']}")

        with st.spinner("Performing deep research..."):
            results = firecrawl_app.deep_research(
                query=query,
                params=params,
                on_activity=on_activity
            )

        return {
            "success": True,
            "final_analysis": results['data']['finalAnalysis'],
            "sources_count": len(results['data']['sources']),
            "sources": results['data']['sources']
        }
    except Exception as e:
        st.error(f"Deep research error: {str(e)}")
        return {"error": str(e), "success": False}

# Main research process
async def run_research_process(topic: str):
    # Step 1: Deep research using Firecrawl
    with st.spinner("Conducting deep web research..."):
        research_result = await deep_research(topic, max_depth=3, time_limit=180, max_urls=10)
    
    if not research_result["success"]:
        return "Research failed. Please try again."

    final_analysis = research_result["final_analysis"]
    sources = research_result["sources"]

    initial_prompt = f"""
You are a research assistant. Based on the analysis and sources below, generate a clear, structured research report.

Topic: {topic}

Analysis:
{final_analysis}

Sources:
{sources}

Write a concise research report with citations and key insights.
"""
    initial_report = call_llama(initial_prompt)

    with st.expander("View Initial Research Report"):
        st.markdown(initial_report)

    # Step 2: Enhance the report
    with st.spinner("Enhancing the report with additional information..."):
        enhancement_prompt = f"""
Enhance the following research report by:
- Adding more detail to complex parts
- Including examples, case studies, and applications
- Describing visual elements like diagrams
- Making the report comprehensive but factual

Original Report:
{initial_report}
"""
        enhanced_report = call_llama(enhancement_prompt)

    return enhanced_report

# Trigger
if st.button("Start Research", disabled=not (research_topic and st.session_state.llama_endpoint and st.session_state.firecrawl_api_key)):
    if not research_topic:
        st.warning("Please enter a research topic.")
    elif not st.session_state.firecrawl_api_key:
        st.warning("Please enter the Firecrawl API key.")
    elif not st.session_state.llama_endpoint:
        st.warning("Please enter the LLaMA API endpoint.")
    else:
        try:
            report_placeholder = st.empty()
            enhanced_report = asyncio.run(run_research_process(research_topic))

            report_placeholder.markdown("## Enhanced Research Report")
            report_placeholder.markdown(enhanced_report)

            st.download_button(
                "Download Report",
                enhanced_report,
                file_name=f"{research_topic.replace(' ', '_')}_report.md",
                mime="text/markdown"
            )
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

# Footer
st.markdown("---")
st.markdown("Powered by LLaMA and Firecrawl")
