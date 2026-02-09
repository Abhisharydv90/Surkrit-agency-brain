import os
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from crewai import Agent, Task, Crew, Process, LLM

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Using Groq for high-speed, error-free agents
groq_llm = LLM(model="groq/llama-3.3-70b-versatile", api_key=os.environ.get("GROQ_API_KEY"))

class ProjectRequest(BaseModel):
    prompt: str

@app.post("/run-agency")
async def run_agency(request: ProjectRequest):
    manager = Agent(role='Manager', goal='Oversee project', backstory='CEO', llm=groq_llm)
    developer = Agent(role='Dev', goal='Write solution', backstory='Expert', llm=groq_llm)
    
    t1 = Task(description=request.prompt, expected_output="Detailed result", agent=developer)
    crew = Crew(agents=[manager, developer], tasks=[t1], process=Process.sequential)
    
    result = crew.kickoff()
    return {"output": str(result)}

@app.get("/")
async def root():
    return {"status": "Agency Brain is Online"}
