import os
import logging
from typing import Optional, Dict, Any
from langchain_openai import ChatOpenAI, AzureChatOpenAI
from langchain_core.language_models import BaseChatModel

logger = logging.getLogger("LAAJ_Connection")


class LLMConnectionManager:
    """Manages LLM connections for agent and tool usage with separate configurations."""
    
    def __init__(self):
        """Initialize the connection manager with environment configurations."""
        self.agent_config = self._load_config("AGENT_LLM")
        self.tool_config = self._load_config("TOOL_LLM")
        
        # Create LLM instances
        self.agent_llm = self._create_llm(self.agent_config, "Agent")
        self.tool_llm = self._create_llm(self.tool_config, "Tool")
        
        # Log configurations
        logger.info("LLM Connection Manager initialized")
        logger.info(f"Agent LLM: {self.agent_config.get('model_name', 'Not configured')}")
        logger.info(f"Tool LLM: {self.tool_config.get('model_name', 'Not configured')}")
    
    def _load_config(self, prefix: str) -> Dict[str, Any]:
        """Load configuration for a specific LLM type from environment variables.
        
        Args:
            prefix: Either "AGENT_LLM" or "TOOL_LLM"
            
        Returns:
            Dictionary with configuration values
        """
        # Check for fallback to LAAJ_* or OPENAI_* variables
        model_name = os.getenv(f"{prefix}_MODEL")
        api_key = os.getenv(f"{prefix}_API_KEY")
        base_url = os.getenv(f"{prefix}_BASE_URL")
        azure_deployment = os.getenv(f"{prefix}_AZURE_DEPLOYMENT")
        
        # Fallback logic
        if not model_name and not azure_deployment:
            model_name = os.getenv("LAAJ_MODEL", "gpt-4-turbo-preview")
        if not api_key:
            api_key = os.getenv("LAAJ_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not base_url:
            base_url = os.getenv("LAAJ_BASE_URL") or os.getenv("OPENAI_BASE_URL")
        if not azure_deployment:
            azure_deployment = os.getenv("LAAJ_AZURE_DEPLOYMENT")
        
        config = {
            "model_name": model_name or azure_deployment,  # Use azure_deployment if model_name is not set
            "api_key": api_key,
            "base_url": base_url,
            "temperature": float(os.getenv(f"{prefix}_TEMPERATURE", os.getenv("LAAJ_TEMPERATURE", "0"))),
            "max_tokens": int(os.getenv(f"{prefix}_MAX_TOKENS", os.getenv("LAAJ_MAX_TOKENS", "4096"))) if os.getenv(f"{prefix}_MAX_TOKENS") or os.getenv("LAAJ_MAX_TOKENS") else None,
            "api_type": os.getenv(f"{prefix}_API_TYPE", os.getenv("LAAJ_API_TYPE")),
            "api_version": os.getenv(f"{prefix}_API_VERSION", os.getenv("LAAJ_API_VERSION", "2023-05-15")),
            "azure_deployment": azure_deployment,
            "org_id": os.getenv(f"{prefix}_ORG_ID", os.getenv("LAAJ_ORG_ID"))
        }
        
        return config
    
    def _create_llm(self, config: Dict[str, Any], llm_type: str) -> Optional[BaseChatModel]:
        """Create an LLM instance based on configuration.
        
        Args:
            config: Configuration dictionary
            llm_type: Type of LLM ("Agent" or "Tool") for logging
            
        Returns:
            LLM instance or None if not configured
        """
        if not config.get("api_key"):
            logger.warning(f"{llm_type} LLM not configured - no API key found")
            return None
        
        model_name = config["model_name"]
        azure_deployment = config.get("azure_deployment")
        
        # Check if this is an O1 model (check both model_name and azure_deployment)
        model_to_check = azure_deployment or model_name
        is_o1_model = model_to_check and (model_to_check.lower().startswith('o1') or 'o1-' in model_to_check.lower() or 'o4-' in model_to_check.lower())
        
        # Check if we're using Azure OpenAI
        is_azure = (config.get("base_url") and 'azure' in config["base_url"].lower()) or config.get("api_type") == "azure"
        is_litellm_proxy = config.get("base_url") and any(indicator in config["base_url"].lower() for indicator in ['litellm', '4000', 'proxy'])
        
        logger.info(f"{llm_type} LLM - Detected environment: Azure={is_azure}, LiteLLM Proxy={is_litellm_proxy}")
        
        # Initialize the LLM with all configurations
        if is_azure and not is_litellm_proxy:
            # Direct Azure OpenAI usage
            # Use azure_deployment from config, fallback to model_name if not set
            deployment_name = azure_deployment or model_name
            api_version = config.get("api_version", "2023-05-15")
            
            llm_kwargs = {
                "azure_deployment": deployment_name,
                "api_version": api_version,
            }
            
            if not is_o1_model:
                llm_kwargs["temperature"] = config["temperature"]
            else:
                logger.info(f"{llm_type} LLM - O1 model detected ({model_name}), using default temperature of 1.0")
            
            llm_kwargs["openai_api_key"] = config["api_key"]
            
            if config.get("base_url"):
                llm_kwargs["azure_endpoint"] = config["base_url"]
                
            if config.get("max_tokens"):
                if is_o1_model:
                    llm_kwargs["model_kwargs"] = {"max_completion_tokens": config["max_tokens"]}
                    logger.info(f"{llm_type} LLM - O1 model detected, using max_completion_tokens={config['max_tokens']}")
                else:
                    llm_kwargs["max_tokens"] = config["max_tokens"]
                
            logger.info(f"{llm_type} LLM - Using AzureChatOpenAI with deployment: {deployment_name}")
            return AzureChatOpenAI(**llm_kwargs)
            
        else:
            # Standard OpenAI or LiteLLM proxy usage
            llm_kwargs = {
                "model": model_name,
            }
            
            # For litellm proxy with Azure backend, we might need to prefix with "azure/"
            if is_litellm_proxy and is_azure and not model_name.startswith("azure/"):
                if azure_deployment:
                    llm_kwargs["model"] = f"azure/{azure_deployment}"
                    logger.info(f"{llm_type} LLM - Using litellm proxy with Azure model: {llm_kwargs['model']}")
                else:
                    logger.warning(f"{llm_type} LLM - Azure endpoint detected but model doesn't have 'azure/' prefix: {model_name}")
            
            if not is_o1_model:
                llm_kwargs["temperature"] = config["temperature"]
            else:
                logger.info(f"{llm_type} LLM - O1 model detected ({model_name}), using default temperature of 1.0")
            
            llm_kwargs["openai_api_key"] = config["api_key"]
            
            if config.get("base_url"):
                llm_kwargs["base_url"] = config["base_url"]
                
            if config.get("max_tokens"):
                if is_o1_model:
                    llm_kwargs["model_kwargs"] = {"max_completion_tokens": config["max_tokens"]}
                    logger.info(f"{llm_type} LLM - O1 model detected, using max_completion_tokens={config['max_tokens']}")
                else:
                    llm_kwargs["max_tokens"] = config["max_tokens"]
                
            if config.get("org_id"):
                llm_kwargs["openai_organization"] = config["org_id"]
            
            return ChatOpenAI(**llm_kwargs)
    
    def get_agent_llm(self) -> Optional[BaseChatModel]:
        """Get the LLM instance for agent reasoning.
        
        Returns:
            Agent LLM instance or None if not configured
        """
        return self.agent_llm
    
    def get_tool_llm(self) -> Optional[BaseChatModel]:
        """Get the LLM instance for tool calling.
        
        Returns:
            Tool LLM instance or fallback to agent LLM if not configured
        """
        # If tool LLM is not configured, fallback to agent LLM
        if self.tool_llm is None:
            logger.info("Tool LLM not configured, falling back to Agent LLM")
            return self.agent_llm
        return self.tool_llm
    
    def get_configurations(self) -> Dict[str, Dict[str, Any]]:
        """Get all configurations for logging/debugging.
        
        Returns:
            Dictionary with agent and tool configurations
        """
        return {
            "agent": self.agent_config,
            "tool": self.tool_config
        } 