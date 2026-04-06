from fastapi import FastAPI
from pydantic import BaseModel
from infrastructure.vllm_model import VLLMModel
from application.agent import CodingAgent

app = FastAPI(title="Mini Coding Agent vLLM")

llm = VLLMModel()
agent = CodingAgent(llm)

class CodeRequest(BaseModel):
    description: str

@app.post("/suggest_code")
async def suggest_code(req: CodeRequest):
    return await agent.generate_code(req.description)