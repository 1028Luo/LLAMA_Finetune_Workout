from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor
# load fine-tuned model from hub
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
import bitsandbytes

# import any embedding model on HF hub (https://huggingface.co/spaces/mteb/leaderboard)
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
# Settings.embed_model = HuggingFaceEmbedding(model_name="thenlper/gte-large") # alternative model

Settings.llm = None
Settings.chunk_size = 512
Settings.chunk_overlap = 25


model_name = "meta-llama/Llama-3.1-8B-instruct"
model = AutoModelForCausalLM.from_pretrained(model_name,
                                             load_in_8bit=True,
                                            device_map="cuda",
                                            trust_remote_code=False,
                                            revision="main")


# load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

# # Manually move model to GPU (optional, ensures GPU-only placement)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(device)

# articles available here: {add GitHub repo}
documents = SimpleDirectoryReader("RAG_Data").load_data()

# some ad hoc document refinement
# print(len(documents))
# for doc in documents:
#     if "Member-only story" in doc.text:
#         documents.remove(doc)
#         continue

#     if "The Data Entrepreneurs" in doc.text:
#         documents.remove(doc)

#     if " min read" in doc.text:
#         documents.remove(doc)

print("length of document: ", len(documents))

# store docs into vector DB
index = VectorStoreIndex.from_documents(documents)

# set number of docs to retreive
top_k = 6

# configure retriever
retriever = VectorIndexRetriever(
    index=index,
    similarity_top_k=top_k,
)

# assemble query engine
query_engine = RetrieverQueryEngine(
    retriever=retriever,
    node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.5)],
)


def chat():
    while True:
        question = input("You: ")
        if question.lower() == 'exit':
            break
        # query documents

        response = query_engine.query(question)

        # reformat response
        context = "Context:\n"
        for i in range(top_k):
            context = context + response.source_nodes[i].text + "\n\n"

        # # prompt with no context
        # intstructions_string = f"""You are called MyGPT, an expert assistant in Robotics and AI, providing detailed and accurate responses to any query about the knowledge that you have learnt and  in your RAG data base.\
        # MyGPT will tailor the length of its responses to match the query, providing clear explanation on papers, projects and concepts in the field of robotics and AI \
        # and should keep the answer concise.

        # Please respond to the following question.
        # """
        # prompt_template = lambda comment: f'''[INST] {intstructions_string} \n{comment} \n[/INST]'''

        # prompt = prompt_template(question)
        # # print(prompt)

        # model.eval()

        # inputs = tokenizer(prompt, return_tensors="pt")
        # outputs = model.generate(input_ids=inputs["input_ids"].to("cuda"), max_new_tokens=280)

        # print(tokenizer.batch_decode(outputs)[0])

        
        # ##### with Context #####
        # prompt_template_w_context = lambda context, question: f"""[INST]You are called MyGPT, an expert assistant in Robotics and AI, providing detailed and accurate responses to any query about the knowledge that you have learnt and  in your RAG data base.\
        # MyGPT will tailor the length of its responses to match the query, providing clear explanation on papers, projects and concepts in the field of robotics and AI \
        # and should keep the answer concise.

        # {context}
        # Please respond to the following question. Use the context above if it is helpful.

        # {question}
        # [/INST]
        # """


        prompt = [
            {"role": "system", "content": "You are called MyGPT, an expert assistant in Robotics and AI, providing detailed and accurate responses to any query about the knowledge that you have learnt and  in your RAG data base.\
        MyGPT will tailor the length of its responses to match the query, providing clear explanation on papers, projects and concepts in the field of robotics and AI \
        and should keep the answer concise. You can use the provided context if needed."},
            {"role": "system", "content": context},
            {"role": "user", "content": question},
        ]

        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
        )

        generation_args = {
            "max_new_tokens": 500,
            "return_full_text": False,
            "temperature": 0.0,
            "do_sample": False,
            "pad_token_id": tokenizer.pad_token_id
        }

        output = pipe(prompt, **generation_args)
        print(output[0]['generated_text'])




chat()