import asyncio
from googlesearch import search
from typing import List, Dict, Any, Optional
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
from crawl4ai.content_filter_strategy import BM25ContentFilter
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator


def get_urls(query: str) -> List[str]:
    """
    Get relevant URLs from Google search
    
     Args:
        query: Search string to find relevant web pages
        num_results: Number of search results to return (default: 20)
        advanced: Flag to enable advanced search features (default: True)

    Returns:
        List of filtered URLs (max 5) that match the criteria:
        - Start with http:// or https://
        
    """
    try:
        results = search(
            query,
            num_results=20,
            advanced=True,
        )
        return [
            result.url for result in results 
            if result.url.startswith(('http://', 'https://'))
            # and not result.url.endswith(('.pdf', '.doc', '.docx'))
        ][:5]
    except Exception as e:
        print(f"Search failed: {str(e)}")
        return []

async def process_url(url: str, query: str) -> Optional[Dict]:
    """
    Process individual URL with content filtering
    
    Args:
        url: Web address to crawl and process
        query: Original search query used for content filtering

    Returns:
        Dictionary containing processed content with keys:
        - url: Source URL
        - content: Filtered markdown content
        - title: Extracted page title (if available)
        - description: Extracted page description (if available)

        Returns None if processing fails
        
    """
    try:
        # Initialize content filter and markdown generator
        bm25_filter = BM25ContentFilter(
            user_query=query,
            bm25_threshold=1.25
        )
        
        md_generator = DefaultMarkdownGenerator(
            content_filter=bm25_filter
        )

        # Configure browser and crawler
        browser_config = BrowserConfig(
            headless=True,
            verbose=False,
            extra_args=["--disable-gpu", "--disable-dev-shm-usage", "--no-sandbox"],
            user_agent_mode="desktop"
        )
        
        crawl_config = CrawlerRunConfig(
            cache_mode=CacheMode.BYPASS,
            excluded_tags=['nav', 'header', 'footer', 'script', 'style', 
                          'form', 'aside', 'sidebar', 'menu', 'button'],
            exclude_external_images=True,
            markdown_generator=md_generator
        )

        async with AsyncWebCrawler(config=browser_config) as crawler:
            async with asyncio.timeout(15):
                result = await crawler.arun(
                    url=url,
                    config=crawl_config,
                    extracted_metadata=["title", "description"],
                    session_id="session1"
                )
                
                if result.success:
                    return {
                        "url": url,
                        "content": result.markdown.fit_markdown,
                        "title": result.metadata.get("title"),
                        "description": result.metadata.get("description")
                    }
                print(f"Failed: {url} - {result.error_message}")
                return None
                
    except Exception as e:
        print(f"Error processing {url}: {str(e)}")
        return None

async def crawl_parallel(query: str, max_concurrent: int = 5) -> List[Dict]:
    """
    Main web crawling entry point
    
    Args:
        query: Search term to find relevant URLs
        max_concurrent: Maximum simultaneous requests allowed (default: 5)

    Returns:
        List of successfully processed URL results (filtered by BM25 relevance),
        excluding any failed or empty responses
        
    """
    urls = get_urls(query)
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def limited_process(url: str):
        async with semaphore:
            return await process_url(url, query)
    
    results = await asyncio.gather(*[limited_process(url) for url in urls])
    return [result for result in results if result is not None]