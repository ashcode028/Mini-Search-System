from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routes.search_apis import router as search_router

# Initialize FastAPI app
app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(search_router, prefix="/api/v1")

if __name__ == "__main__":
    import uvicorn

    # Load data on startup
    # search_system = InMemorySearch()
    # search_system.load("data")
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
