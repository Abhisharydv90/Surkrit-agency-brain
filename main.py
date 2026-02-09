import os
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

# These are the "Connectors" that were causing errors earlier
from crewai import Agent, Task, Crew, Process
from langchain_groq import ChatGroq

app = FastAPI()

# 1. THE SECURITY HANDSHAKE (CORS)
# This allows your StackBlitz dashboard to talk to this Render backend.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. SETUP THE BRAIN (Groq Llama 3.3)
# Make sure GROQ_API_KEY is added to your Render Environment Variables!
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    groq_api_key=os.environ.get("GROQ_API_KEY")
)

# 3. DEFINE THE DATA FORMAT
class ProjectRequest(BaseModel):
    prompt: str

# 4. THE MAIN AGENT ENDPOINT
@app.post("/run-agency")
async def run_agency(request: ProjectRequest):
    try:
        # Define your Digital Employees (Agents)
        manager = Agent(
            role='Operations Manager',
            goal='Review the final result for quality and clarity',
            backstory='Expert CEO focused on perfect delivery.',
            llm=llm
        )

        developer = Agent(
            role='Senior Specialist',
            goal='Execute the user request with high technical accuracy',
            backstory='A top-tier expert who follows instructions perfectly.',
            llm=llm
        )

        # Create the Task based on what you type in the dashboard
        task = Task(
            description=request.prompt,
            expected_output="A professional and detailed response to the client's request.",
            agent=developer
        )

        # Form the Crew (Digital Agency)
        crew = Crew(
            agents=[manager, developer],
            tasks=[task],
            process=Process.sequential,
            verbose=True
        )

        # Execute the mission
        result = crew.kickoff()
        
        # Return the output as a string so the dashboard can read it
        return {"output": str(result)}

    except Exception as e:
        # If something breaks, tell us why in the logs
        print(f"ERROR: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# 5. HEALTH CHECK (To see if the server is awake)
@app.get("/")
async def root():
    return {"status": "Sukrit Agency Brain is ONLINE and READY"}

# 6. RENDER PORT BINDING
# This is the "Magic Fix" for the Port Scan Timeout error.
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
