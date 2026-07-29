from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

app = FastAPI()

# Enable CORS for front-end requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# ---------- ROOT: serve the chat page ----------
# FastAPI will read index.html from the SAME folder as main.py and send it
# to the browser when someone visits your Render URL.
@app.get("/", include_in_schema=False)
def home():
    return FileResponse("index.html")


class ChatRequest(BaseModel):
    message: str

@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    user_message = request.message
    
    # Place your FastAPI AI/LLM or Chatbot logic here
    bot_reply = f"Hello! You asked about: '{user_message}'. How can ACB Caribbean Bank assist you further?"
    
    return {"reply": bot_reply}
