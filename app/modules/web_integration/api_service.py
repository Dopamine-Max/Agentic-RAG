from fastapi import FastAPI
from contextlib import asynccontextmanager
from pydantic import BaseModel
from crawler import crawl_parallel
import asyncio
import uvicorn

app = FastAPI()

class SearchRequest(BaseModel):
    query: str

@app.post("/web-search")
async def web_search(request: SearchRequest):
    try:
        return await asyncio.wait_for(
            crawl_parallel(request.query), 
            timeout=20 
        )
    except asyncio.TimeoutError:
        return {"results": []}
    
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
