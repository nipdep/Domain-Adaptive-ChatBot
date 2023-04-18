import pickle
from query_data import get_chain

from transformers import LlamaTokenizer, LlamaForCausalLM, GenerationConfig, pipeline
from langchain.llms import HuggingFacePipeline
from langchain import PromptTemplate, LLMChain

import torch


if __name__ == "__main__":
    with open("vectorstore.pkl", "rb") as f:
        vectorstore = pickle.load(f)

    tokenizer = LlamaTokenizer.from_pretrained("chavinlo/alpaca-native")
    base_model = LlamaForCausalLM.from_pretrained(
        "chavinlo/alpaca-native",
        load_in_8bit=True,
        device_map='auto',
    )

    pipe = pipeline(
        "text-generation",
        model=base_model, 
        tokenizer=tokenizer, 
        max_length=256,
        temperature=0.6,
        top_p=0.95,
        repetition_penalty=1.2
    )

    local_llm = HuggingFacePipeline(pipeline=pipe)

    qa_chain = get_chain(vectorstore, local_llm)
    chat_history = []
    print("Chat with your docs!")
    while True:
        print("Human:")
        question = input()
        result = qa_chain({"question": question, "chat_history": chat_history})
        chat_history.append((question, result["answer"]))
        print("AI:")
        print(result["answer"])
