import json 
from typing import Dict, Any, Optional
from langchain_core.messages import HumanMessage, SystemMessage
from prompts.json_fix import JSON_FIX_PROMPT
import asyncio
import logging
laaj_logger = logging.getLogger("LAAJ.JSONFixer")

async def fix_json_with_llm(self, malformed_content: str, file_path: str) -> Optional[Dict[str, Any]]:
        """
        Use LLM to fix malformed JSON content.
        
        Args:
            malformed_content: The malformed JSON content as a string
            file_path: Path to the file being processed (for logging)
            
        Returns:
            Fixed JSON object if successful, None if failed
        """
        try:
            laaj_logger.info(f"Attempting to fix JSON with LLM for file: {file_path}")
            print(f"Fixing JSON with LLM for: {file_path}")
            
            # Prepare the prompt
            prompt = f"{JSON_FIX_PROMPT}\n\n{malformed_content}"
            
            messages = [
                SystemMessage(content="You are a JSON formatting expert. Fix the JSON and return only the valid JSON object."),
                HumanMessage(content=prompt)
            ]
            
            # Call the LLM with a shorter timeout for JSON fixing
            response = await asyncio.wait_for(
                self.agent_llm.ainvoke(messages, config={"configurable": {"request_timeout": 60}}),
                timeout=60
            )
            print("FIXED content:", response)
            
            # Extract the response content
            fixed_content = response.content.strip()
            
            # Remove any markdown code blocks if present
            if "```json" in fixed_content:
                fixed_content = fixed_content.split("```json")[1].split("```")[0].strip()
            elif "```" in fixed_content:
                fixed_content = fixed_content.split("```")[1].split("```")[0].strip()
            
            # Try to parse the fixed JSON
            fixed_json = json.loads(fixed_content)
            laaj_logger.info(f"Successfully fixed JSON for file: {file_path}")
            print(f"Successfully fixed JSON for: {file_path}")
            return fixed_json
            
        except asyncio.TimeoutError:
            laaj_logger.error(f"LLM JSON fixing timed out for file: {file_path}")
            print(f"LLM JSON fixing timed out for: {file_path}")
            return None
        except json.JSONDecodeError as e:
            laaj_logger.error(f"LLM fixed JSON still invalid for file {file_path}: {e}")
            print(f"LLM fixed JSON still invalid for: {file_path}")
            return None
        except Exception as e:
            laaj_logger.error(f"Error during LLM JSON fixing for file {file_path}: {e}")
            print(f"Error during LLM JSON fixing for: {file_path}")
            return None
        
def simple_json_repair(content: str) -> Optional[Dict[str, Any]]:
    """
    Simple JSON repair function that handles common formatting issues.
    This is used as a fallback when LLM is not available.
    """
    try:
        # Remove markdown code blocks
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
        
        # Remove surrounding quotes if present
        if content.startswith('"') and content.endswith('"'):
            content = content[1:-1]
        
        # Try to unescape common escape sequences
        content = content.replace('\\"', '"').replace('\\n', '\n').replace('\\t', '\t')
        
        # Try to parse
        result = json.loads(content)
        laaj_logger.info("Simple JSON repair successful")
        return result
    except Exception as e:
        laaj_logger.debug(f"Simple JSON repair failed: {e}")
        return None