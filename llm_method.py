import requests
import ollama
import langchain_ollama
from langchain_core.prompts import ChatPromptTemplate

from langchain_ollama import OllamaLLM

def get_llm_response(query: str, context: str) -> str:
    template = """
    Use the following context to answer the question with a brief description.
    
    Context: {context}
    
    Question to answer: {query}
    """    
    prompt = ChatPromptTemplate.from_template(template)
    # print(f"Generated prompt for LLM:\n{prompt}\n")

    llm = OllamaLLM(model="llama3.2")
    chain = prompt | llm
    response = chain.invoke({"context": context, "query": query})
    
    return response
   

# Example usage:
# answer = get_llm_response("What is Gen AI?", "GEN AI, or Generative Artificial Intelligence, is a type of AI that creates new content like text, images, music, and code by learning patterns from vast datasets. These large AI models, often called foundation models, can perform various tasks and can be adapted for specific uses with minimal additional training. Examples of Generative AI include tools like ChatGPT and Google")
# print(answer)