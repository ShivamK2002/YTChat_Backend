from fastapi import FastAPI
from pydantic import BaseModel
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()
app = FastAPI()

class QueryRequest(BaseModel):
    video_id: str
    question: str




llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash",
              temperature=0
              )



@app.post("/ask")
def ask_video(req: QueryRequest):
    try:
        # Fetch transcript
        transcript_list = YouTubeTranscriptApi().fetch(req.video_id, languages=["en"])
        transcript = " ".join([snippet.text for snippet in transcript_list])

        # Split into chunks
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_text(transcript)

        # Embeddings
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = FAISS.from_texts(chunks, embeddings)

        # Retriever
        retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k":4})

        # Prompt template
        prompt = PromptTemplate(
            template="""
            You are a helpful assistant.
            Answer ONLY from the provided transcript context.
            If the context is insufficient, just say you don't know.

            {context}
            Question: {question}
            """,
            input_variables=['context', 'question']
        )

        # Parallel chain: context + question
        parallel_chain = RunnableParallel({
            "context": retriever | RunnableLambda(lambda docs: "\n\n".join([doc.page_content for doc in docs])),
            "question": RunnablePassthrough()
        })

        # Final chain
        final_chain = parallel_chain | prompt | llm | StrOutputParser()
        answer = final_chain.invoke(req.question)

        return {"answer": answer}

    except TranscriptsDisabled:
        return {"answer": "No captions available for this video."}
    except Exception as e:
        return {"answer": f"Error: {str(e)}"}
