import os
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from crewai import Agent, Task, Crew, Process
from langchain_groq import ChatGroq

app = FastAPI()

# SECURITY: Allows your dashboard to talk to this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# LLM SETUP: Replace 'GROQ_API_KEY' in Render Environment Variables
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    groq_api_key=os.environ.get("GROQ_API_KEY")
)

class ProjectRequest(BaseModel):
    prompt: str

@app.post("/run-agency")
async def run_agency(request: ProjectRequest):
    developer = Agent(
        role='Expert Agent',
        goal='Complete the user task efficiently',
        backstory='Specialized AI assistant.',
        llm=llm
    )
    task = Task(description=request.prompt, expected_output="Detailed result", agent=developer)
    crew = Crew(agents=[developer], tasks=[task], process=Process.sequential)
    result = crew.kickoff()
    return {"output": str(result)}

@app.get("/")
async def root():
    return {"status": "Agency Brain is Online"}

# PORT BINDING: This is the critical fix for Render's "Port scan timeout"
if __name__ == "__main__":
    # Render assigns a port dynamically; we must listen on it
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
