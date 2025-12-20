import os
import time
import logging
from datetime import datetime
from typing import Optional, Any, List
from dataclasses import dataclass
from threading import Lock

from pydantic import BaseModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class KeyStats:
    """Statistics for a single API key"""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost: float = 0.0
    total_latency_ms: float = 0.0
    last_used: Optional[datetime] = None
    consecutive_failures: int = 0
    is_healthy: bool = True


class LLMKey:
    """Single LLM API key with its client"""
    
    def __init__(self, provider: str, api_key: str, model: str, cost_input: float, cost_output: float):
        self.provider = provider
        self.api_key = api_key
        self.model = model
        self.cost_per_1m_input = cost_input
        self.cost_per_1m_output = cost_output
        self.stats = KeyStats()
        self.lock = Lock()
        
        # Initialize client
        try:
            if provider == "gemini":
                self.client = ChatGoogleGenerativeAI(
                    model=model,
                    temperature=0.1,
                    google_api_key=api_key
                )
            elif provider == "openai":
                self.client = ChatOpenAI(
                    model=model,
                    temperature=0.1,
                    openai_api_key=api_key
                )
            elif provider == "anthropic":
                self.client = ChatAnthropic(
                    model=model,
                    temperature=0.1,
                    anthropic_api_key=api_key
                )
            else:
                raise ValueError(f"Unknown provider: {provider}")
            
            logger.info(f"✅ Initialized {provider} key: ...{api_key[-8:]}")
        except Exception as e:
            logger.error(f"❌ Failed to initialize {provider} key: {e}")
            self.stats.is_healthy = False
    
    def health_check(self) -> bool:
        """Check if this key is working"""
        try:
            start = time.time()
            self.client.invoke("health")
            elapsed = (time.time() - start) * 1000
            
            with self.lock:
                self.stats.is_healthy = True
                self.stats.consecutive_failures = 0
            
            logger.info(f"✅ {self.provider} ...{self.api_key[-8:]}: healthy ({elapsed:.0f}ms)")
            return True
            
        except Exception as e:
            with self.lock:
                self.stats.consecutive_failures += 1
                if self.stats.consecutive_failures >= 3:
                    self.stats.is_healthy = False
            
            logger.warning(f"⚠️  {self.provider} ...{self.api_key[-8:]}: failed ({e})")
            return False
    
    def invoke(self, messages: Any, structured_output: Optional[type[BaseModel]] = None) -> Any:
        """Make LLM request"""
        if not self.stats.is_healthy:
            raise RuntimeError(f"{self.provider} key unhealthy")
        
        with self.lock:
            self.stats.total_requests += 1
        
        start = time.time()
        
        try:
            if structured_output:
                llm = self.client.with_structured_output(structured_output)
                response = llm.invoke(messages)
            else:
                response = self.client.invoke(messages)
            
            elapsed = (time.time() - start) * 1000
            
            # Extract tokens
            input_tokens = 0
            output_tokens = 0
            
            if hasattr(response, 'usage_metadata'):
                input_tokens = getattr(response.usage_metadata, 'input_tokens', 0)
                output_tokens = getattr(response.usage_metadata, 'output_tokens', 0)
            elif hasattr(response, 'response_metadata'):
                usage = response.response_metadata.get('usage', {})
                input_tokens = usage.get('prompt_tokens', 0)
                output_tokens = usage.get('completion_tokens', 0)
            
            cost = (
                (input_tokens / 1_000_000) * self.cost_per_1m_input +
                (output_tokens / 1_000_000) * self.cost_per_1m_output
            )
            
            with self.lock:
                self.stats.successful_requests += 1
                self.stats.consecutive_failures = 0
                self.stats.total_input_tokens += input_tokens
                self.stats.total_output_tokens += output_tokens
                self.stats.total_cost += cost
                self.stats.total_latency_ms += elapsed
                self.stats.last_used = datetime.now()
            
            logger.info(
                f"✅ {self.provider} ...{self.api_key[-8:]}: "
                f"{elapsed:.0f}ms, {input_tokens}→{output_tokens} tokens, ${cost:.6f}"
            )
            
            return response
            
        except Exception as e:
            with self.lock:
                self.stats.failed_requests += 1
                self.stats.consecutive_failures += 1
                if self.stats.consecutive_failures >= 3:
                    self.stats.is_healthy = False
            
            logger.error(f"❌ {self.provider} ...{self.api_key[-8:]}: {e}")
            raise


class LLMGateway:
    """Simple gateway with multi-key support and round-robin"""
    
    def __init__(self):
        self.keys: List[LLMKey] = []
        self.current_index = 0
        self.lock = Lock()
        
        # Load Gemini keys
        gemini_keys = os.getenv("GOOGLE_API_KEY", "")
        if gemini_keys:
            for key in gemini_keys.split(","):
                key = key.strip()
                if key:
                    self.keys.append(LLMKey(
                        "gemini", 
                        key, 
                        os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
                        cost_input=0.075,
                        cost_output=0.30
                    ))
        
        # Load OpenAI keys
        # openai_keys = os.getenv("OPENAI_API_KEYS", "")
        # if openai_keys:
        #     for key in openai_keys.split(","):
        #         key = key.strip()
        #         if key:
        #             self.keys.append(LLMKey(
        #                 "openai",
        #                 key,
        #                 os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        #                 cost_input=0.15,
        #                 cost_output=0.60
        #             ))
        
        # # Load Anthropic keys
        # anthropic_keys = os.getenv("ANTHROPIC_API_KEYS", "")
        # if anthropic_keys:
        #     for key in anthropic_keys.split(","):
        #         key = key.strip()
        #         if key:
        #             self.keys.append(LLMKey(
        #                 "anthropic",
        #                 key,
        #                 os.getenv("ANTHROPIC_MODEL", "claude-3-5-haiku-20241022"),
        #                 cost_input=1.0,
        #                 cost_output=5.0
        #             ))
        
        if not self.keys:
            raise ValueError("No API keys found! Set GEMINI_API_KEYS, OPENAI_API_KEYS, or ANTHROPIC_API_KEYS")
        
        logger.info(f"🚀 Gateway initialized with {len(self.keys)} keys")
        
        # Initial health check
        self.health_check()
    
    def health_check(self):
        """Check health of all keys"""
        logger.info("🏥 Running health checks...")
        for key in self.keys:
            key.health_check()
    
    def _get_next_healthy_key(self) -> Optional[LLMKey]:
        """Get next healthy key using round-robin"""
        with self.lock:
            attempts = 0
            while attempts < len(self.keys):
                key = self.keys[self.current_index]
                self.current_index = (self.current_index + 1) % len(self.keys)
                attempts += 1
                
                if key.stats.is_healthy:
                    return key
            
            return None
    
    def invoke(
        self, 
        messages: Any,
        structured_output: Optional[type[BaseModel]] = None,
        max_retries: int = 3
    ) -> Any:
        """
        Invoke LLM with automatic failover
        
        Args:
            messages: Input messages (string or prompt messages)
            structured_output: Optional Pydantic model for structured output
            max_retries: Maximum retry attempts
            
        Returns:
            LLM response
        """
        last_error = None
        
        for attempt in range(max_retries):
            key = self._get_next_healthy_key()
            
            if not key:
                logger.error("❌ No healthy keys available")
                if attempt < max_retries - 1:
                    logger.info("🔄 Running health check to recover...")
                    self.health_check()
                    time.sleep(1)
                    continue
                else:
                    raise RuntimeError(f"All keys unavailable. Last error: {last_error}")
            
            try:
                return key.invoke(messages, structured_output)
                
            except Exception as e:
                last_error = str(e)
                logger.warning(f"⚠️  Attempt {attempt + 1} failed, trying next key...")
                time.sleep(0.5)
        
        raise RuntimeError(f"All retries failed. Last error: {last_error}")
    
    def get_stats(self) -> dict:
        """Get aggregated statistics"""
        total_requests = sum(k.stats.total_requests for k in self.keys)
        successful_requests = sum(k.stats.successful_requests for k in self.keys)
        failed_requests = sum(k.stats.failed_requests for k in self.keys)
        total_input_tokens = sum(k.stats.total_input_tokens for k in self.keys)
        total_output_tokens = sum(k.stats.total_output_tokens for k in self.keys)
        total_cost = sum(k.stats.total_cost for k in self.keys)
        
        return {
            "total_keys": len(self.keys),
            "healthy_keys": sum(1 for k in self.keys if k.stats.is_healthy),
            "total_requests": total_requests,
            "successful_requests": successful_requests,
            "failed_requests": failed_requests,
            "success_rate": f"{(successful_requests / total_requests * 100) if total_requests > 0 else 100:.2f}%",
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
            "total_cost": f"${total_cost:.6f}",
            "keys": [
                {
                    "provider": k.provider,
                    "key": f"...{k.api_key[-8:]}",
                    "healthy": k.stats.is_healthy,
                    "requests": f"{k.stats.successful_requests}/{k.stats.total_requests}",
                    "tokens": f"{k.stats.total_input_tokens}→{k.stats.total_output_tokens}",
                    "cost": f"${k.stats.total_cost:.6f}",
                    "avg_latency": f"{(k.stats.total_latency_ms / k.stats.successful_requests) if k.stats.successful_requests > 0 else 0:.0f}ms"
                }
                for k in self.keys
            ]
        }
    
    def print_stats(self):
        """Print formatted statistics"""
        stats = self.get_stats()
        
        print("\n" + "="*80)
        print("📊 LLM GATEWAY STATISTICS")
        print("="*80)
        print(f"Keys: {stats['healthy_keys']}/{stats['total_keys']} healthy")
        print(f"Requests: {stats['successful_requests']}/{stats['total_requests']} successful ({stats['success_rate']})")
        print(f"Tokens: {stats['total_input_tokens']:,} input → {stats['total_output_tokens']:,} output")
        print(f"Total Cost: {stats['total_cost']}")
        print()
        
        for key_stat in stats['keys']:
            status = "✅" if key_stat['healthy'] else "❌"
            print(f"{status} {key_stat['provider']} {key_stat['key']}")
            print(f"   Requests: {key_stat['requests']} | Tokens: {key_stat['tokens']}")
            print(f"   Cost: {key_stat['cost']} | Latency: {key_stat['avg_latency']}")
        
        print("="*80 + "\n")
    
    def reset_stats(self):
        """Reset all statistics"""
        for key in self.keys:
            with key.lock:
                key.stats = KeyStats()
        logger.info("📊 Statistics reset")


# Global gateway instance
gateway = LLMGateway()