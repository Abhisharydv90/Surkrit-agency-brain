import os
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from crewai import Agent, Task, Crew, Process
from langchain_groq import ChatGroq

app = FastAPI()

# SECURITY: This allows your StackBlitz dashboard to talk to this code
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# LLM SETUP: Using Groq (Llama 3.3) for high-speed, free agents
# Ensure GROQ_API_KEY is in your Render Environment Variables
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    groq_api_key=os.environ.get("GROQ_API_KEY")
)

class ProjectRequest(BaseModel):
    prompt: str

@app.post("/run-agency")
async def run_agency(request: ProjectRequest):
    # 1. Define Agents
    manager = Agent(
        role='Operations Manager',
        goal='Finalize and polish the task result',
        backstory='Expert CEO focused on quality.',
        llm=llm
    )
    developer = Agent(
        role='Specialist',
        goal='Execute the user request perfectly',
        backstory='Highly skilled digital employee.',
        llm=llm
    )

    # 2. Define Task
    task = Task(
        description=request.prompt,
        expected_output="A high-quality completed task.",
        agent=developer
    )

    # 3. Create Crew
    crew = Crew(
        agents=[manager, developer],
        tasks=[task],
        process=Process.sequential
    )

    result = crew.kickoff()
    return {"output": str(result)}

@app.get("/")
async def root():
    return {"status": "Agency Brain is Online"}

# PORT FIX: This tells Render exactly where to find your app
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
