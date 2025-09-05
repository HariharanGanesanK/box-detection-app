from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Create a FastAPI app instance
app = FastAPI()

# Configure CORS (Cross-Origin Resource Sharing)
# This is crucial to allow your React frontend to communicate with this backend.
origins = ["*"]  # In production, you should restrict this to your frontend's domain

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    """
    Root endpoint that returns a welcome message.
    """
    print("Backend connection successful!")
    return {"message": "The backend is connected"}
