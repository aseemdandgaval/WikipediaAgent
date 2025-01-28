import re
import os
import wikipediaapi
import gradio as gr
from groq import Groq
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

from utils.context import system_prompt
os.environ["GROQ_API_KEY"] = "gsk..."

# Agent Class
class Agent:
    def __init__(self, client, system):
        self.client = client
        self.system = system
        self.memory = []
        # If there is no memory, initialize it with the system message
        if self.memory is not None:
            self.memory = [{"role": "system", "content": self.system}]

    def __call__(self, message=""):
        if message:
            self.memory.append({"role": "user", "content": message})
        result = self.execute()
        self.memory.append({"role": "assistant", "content": result})
        return result

    def execute(self):
        completion = client.chat.completions.create(
            messages = self.memory,
            model="llama3-70b-8192",
        )   
        return completion.choices[0].message.content
    
# Gloabal variables
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
wiki = wikipediaapi.Wikipedia(language='en',  user_agent="aseem" )
embeddings = OpenAIEmbeddings()
faiss_store = None
        
# Utils/Tools for the agent
def calculate(operation):
    return eval(operation)

def wikipedia_search(query, advanced_query, advanced_search=False, top_k=5):
    global faiss_store
    page = wiki.page(query)

    # Check if the page exists
    if page.exists():
        if advanced_search:
            # Get the full content of the Wikipedia page
            content = page.text
            # Split the content into chunks
            chunks = chunk_text(content)
            # Store the chunks in FAISS
            faiss_store = store_in_faiss(chunks)
            # Retrieve the top-k relevant chunks
            top_k_documents = retrieve_top_k(advanced_query, top_k)
            # Return the retrieved documents
            return f"Context: {' '.join(top_k_documents)}\n"
        else:
            return f"Summary: {page.summary}\n"
    else:
        return f"The page '{query}' does not exist on Wikipedia."


def chunk_text(text, chunk_size=512, chunk_overlap=50):
    """
    Uses LangChain's RecursiveCharacterTextSplitter to chunk the text.
    """
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_text(text)
    return chunks

def store_in_faiss(chunks):
    """
    Stores the chunks in a FAISS vector store.
    """
    vector_store = FAISS.from_texts(chunks, embeddings)
    return vector_store

def retrieve_top_k(query, top_k=5):
    """
    Retrieves the top-k most relevant chunks from FAISS.
    """
    if faiss_store is None:
        return "No vector data available. Perform advanced search first."

    # Retrieve top-k documents
    docs_and_scores = faiss_store.similarity_search_with_score(query, top_k)
    top_k_chunks = [doc.page_content for doc, score in docs_and_scores]
    return top_k_chunks

# Automatic execution of the agent
def run_agent(max_iterations=10, query: str = "", display_reasoning=True):
    agent = Agent(client=client, system=system_prompt)
    tools = ["calculate", "wikipedia_search"]
    next_prompt = query
    iteration = 0
    steps = 1
    partial_results = ""
  
    while iteration < max_iterations:
        iteration += 1
        result = agent(next_prompt)

        if display_reasoning:
            partial_results += f" -------- (Step {steps}) -------- \n"
            steps += 1
            partial_results += result + "\n\n"
            yield partial_results

        if "Thought" in result and "Action" in result:
            action = re.findall(r"Action: ([a-z_]+): (.+)", result, re.IGNORECASE)
            chosen_tool = action[0][0]
            args = action[0][1]
            if chosen_tool in tools:
                if chosen_tool == "calculate":
                    tool_result = eval(f"{chosen_tool}({'args'})")
                    next_prompt = f"Observation: {tool_result}"
                else:
                    tool_result = eval(f"{chosen_tool}({args})")
                    next_prompt = f"Observation: {tool_result}"
            else:
                next_prompt = "Observation: Tool not found"
        
            if display_reasoning:
                partial_results += f" -------- (Step {steps}) -------- \n"
                steps += 1
                partial_results += next_prompt[:100] + " ..." + "\n\n"
                yield partial_results
            continue
        
        if "Answer" in result:
            if display_reasoning:
                yield partial_results
            else:
                partial_results += result.split("Answer:")[-1].strip()
                yield partial_results
            break

    if iteration >= max_iterations:
        partial_text += "\nThe Wikipedia AI Agent is likely hallucinating. Please try again :("
        yield partial_text

def generate_response_stream(message, show_reasoning):
    # If show_reasoning = True, we'll show all the partial steps
    # If show_reasoning = False, we only yield the final answer
    yield from run_agent(query=message, display_reasoning=show_reasoning)

def main():
    interface = gr.Interface(
        fn=generate_response_stream,
        inputs=[
            gr.Textbox(label="Ask your question here:"),
            gr.Checkbox(label="Show reasoning")
        ],
        outputs=gr.Textbox(label="Agent Output"),
        title="Wikipedia AI Agent",
        description= (
            "Ask a question to the Wikipedia AI Agent."
            "For eg: \n"
            "- \"What is the weight of a tiger?\" \n"
            "- \"Why are fiber optic cables so fragile?\" \n"
            "- \"How does an internal combustion engine work?\" \n"
        )
    )
    interface.launch()


if __name__ == "__main__":
    main()