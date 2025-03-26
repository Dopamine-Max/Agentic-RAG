import pytest
from modules.processing.web_processor import process_web_search

@pytest.mark.asyncio
async def test_web_search_flow():
    results = await process_web_search("large language models")
    assert len(results) >= 1
    assert 'url' in results[0]