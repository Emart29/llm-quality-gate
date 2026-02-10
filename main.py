"""Main entry point for LLM Quality Gate system."""

import asyncio
import logging
from llm.factory import LLMFactory
from llm.base import LLMRequest

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def demo_providers():
    """Demonstrate LLM provider functionality."""
    logger.info("Starting LLM Quality Gate demo...")
    
    # Load configuration
    try:
        config = LLMFactory.load_config()
        logger.info("Configuration loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        return
    
    # List available providers
    providers = LLMFactory.list_available_providers(config)
    logger.info("Available providers:")
    for name, available in providers.items():
        status = "✓" if available else "✗"
        logger.info(f"  {status} {name}")
    
    # Test default provider
    try:
        provider = LLMFactory.create_default_provider(config)
        logger.info(f"Created default provider: {provider.provider_name}")
        
        # Health check
        is_healthy = await provider.health_check()
        logger.info(f"Provider health check: {'✓' if is_healthy else '✗'}")
        
        if is_healthy:
            # Test generation
            request = LLMRequest(
                prompt="What is the capital of France?",
                system_prompt="You are a helpful geography assistant.",
                max_tokens=50
            )
            
            logger.info("Testing generation...")
            response = await provider.generate_with_retry(request)
            
            if response.error:
                logger.error(f"Generation failed: {response.error}")
            else:
                logger.info(f"Generated response: {response.content}")
                logger.info(f"Usage: {response.usage}")
        
    except Exception as e:
        logger.error(f"Provider test failed: {e}")
    
    # Test role-based providers
    logger.info("\n" + "="*50)
    logger.info("Testing role-based providers...")
    
    try:
        # Test generator role
        generator = LLMFactory.create_provider_for_role("generator", config)
        logger.info(f"Generator provider: {generator.provider_name} ({generator.model})")
        
        # Test judge role
        judge = LLMFactory.create_provider_for_role("judge", config)
        logger.info(f"Judge provider: {judge.provider_name} ({judge.model})")
        
        # Demonstrate generator vs judge usage
        test_prompt = "Explain quantum computing in simple terms."
        
        # Generator creates content
        gen_request = LLMRequest(
            prompt=test_prompt,
            system_prompt="You are a helpful technical writer.",
            max_tokens=100
        )
        
        logger.info("Testing generator...")
        gen_response = await generator.generate_with_retry(gen_request)
        
        if gen_response.error:
            logger.error(f"Generator failed: {gen_response.error}")
        else:
            logger.info(f"Generated content: {gen_response.content[:100]}...")
            
            # Judge evaluates the content
            judge_request = LLMRequest(
                prompt=f"Evaluate this explanation for accuracy and clarity:\n\n{gen_response.content}",
                system_prompt="You are an expert technical reviewer. Rate the explanation on a scale of 1-10 and provide brief feedback.",
                max_tokens=50
            )
            
            logger.info("Testing judge...")
            judge_response = await judge.generate_with_retry(judge_request)
            
            if judge_response.error:
                logger.error(f"Judge failed: {judge_response.error}")
            else:
                logger.info(f"Judge evaluation: {judge_response.content}")
        
    except Exception as e:
        logger.error(f"Role-based provider test failed: {e}")


if __name__ == "__main__":
    asyncio.run(demo_providers())