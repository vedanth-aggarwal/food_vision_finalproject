from langchain_community.llms import Ollama
llm = Ollama(model="llama3")
print(llm.invoke("Why is the sky blue?"))