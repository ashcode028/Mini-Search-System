import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse

from handlers.search_engine import InMemorySearch
from models.responses import BaseResponse
from routes.search_apis import router as search_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Setup: create and load the search engine
    search_engine = InMemorySearch()
    data_dir = "data/sample_metadata"
    if os.path.exists(data_dir):
        search_engine.load(data_dir)

    # Attach to app state
    app.state.search_engine = search_engine

    yield  # Run the app

    # Save data
    search_engine.save(data_dir)


# Initialize FastAPI app
app = FastAPI(lifespan=lifespan)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Landing page route
@app.get("/", response_class=JSONResponse)
async def live_response():
    response = BaseResponse(message="Sematic Search is up and running", status="200")
    return JSONResponse(content=response.model_dump(), status_code=200)


# Include routers
app.include_router(search_router, prefix="/v1")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
