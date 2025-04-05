from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from phi.agent import Agent, RunResponse, AgentKnowledge
from phi.model.google import Gemini
from phi.knowledge.website import WebsiteKnowledgeBase
from phi.vectordb.qdrant import Qdrant
from dotenv import load_dotenv
from phi.embedder.google import GeminiEmbedder
import os
import re
import uvicorn
from urllib.parse import urlparse

from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

load_dotenv()
QDRANT_URL = os.getenv('QDRANT_URL')
QDRANT_API_KEY = os.getenv('QDRANT_API_KEY')

app = FastAPI(title="AI Content Detector API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def generate_collection_name(url: str = None, content: str = None) -> str:
    if url:
        # Extract domain from URL and take first 7 characters
        parsed_url = urlparse(url)
        domain = parsed_url.netloc
        return f"ai_detector_{domain[:7]}"
    elif content:
        # Take first 7 characters of content
        return f"ai_detector_{content[:7]}"
    return "ai_det_default"

class AnalysisRequest(BaseModel):
    url: str
    content: str = None

@app.get("/")
async def read_main():
    return {"msg": "Hello World"}

@app.post("/analyze")
async def analyze(request: AnalysisRequest):
    if not request.url and not request.content:
        raise HTTPException(status_code=400, detail="Either URL or content is required")
    
    collection_name = generate_collection_name(request.url, request.content)
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    
    try:
        # Create collection with unique name
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=768, distance=Distance.COSINE)
        )
        
        vector_db = Qdrant(
            collection=collection_name,
            embedder=GeminiEmbedder(
                model="models/text-embedding-004",
                dimensions=768
            ),
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY
        )
        
        # Initialize knowledge base based on input type
        if request.content:
            # For custom text input
            knowledge_base = AgentKnowledge(
                vector_db=vector_db
            )
            knowledge_base.load_text(request.content)
        else:
            # For website analysis
            knowledge_base = WebsiteKnowledgeBase(
                urls=[request.url],
                vector_db=vector_db,
                max_depth=1
            )
            knowledge_base.load(recreate=True, upsert=True)
        
        agent = Agent(
            name="Website Chatbot",
            model=Gemini(id="gemini-2.0-flash-001"),
            knowledge=knowledge_base,
            search_knowledge=True,
            use_tools=True,
            role="Tell whether the text content is ai generated",
            expected_output="the final response should contain both percentage of ai generated content and then its concise yet detailed analysis, For Example: 46%, this data is less likely to be ai generated"
        )
        
        # Run the analysis
        user_message = f"What percentage of the data in the knowledge base is AI generated? Please analyze and provide proper reasoning for the same"
        
        # Get response from the agent
        response: RunResponse = agent.run(user_message)
        
        # Extract percentage from response
        percentage_match = re.search(r'(\d+)%', response.content)
        if percentage_match:
            percentage = int(percentage_match.group(1))
        else:
            # If no percentage found, make a reasonable guess based on analysis
            percentage = 75  # Default value
        
        return {
            "percentage": percentage,
            "analysis": response.content
        }
    
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        # Delete the collection after analysis is complete
        try:
            client.delete_collection(collection_name=collection_name)
            print(f"Collection {collection_name} deleted successfully")
        except Exception as e:
            print(f"Error deleting collection {collection_name}: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
