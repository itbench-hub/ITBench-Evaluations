import argparse
import json
import os
from typing import Dict, List, Any, Optional, TypedDict
from asteval import Interpreter
import re
import asyncio
import openai

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, BaseMessage
from langchain_core.tools import tool

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

import logging
laaj_logger = logging.getLogger("LAAJ.Agent")

from connection import LLMConnectionManager

import prompts 

EVAL_CRITERIA = ["ROOT_CAUSE_ENTITY", "ROOT_CAUSE_ENTITY_K" , "ROOT_CAUSE_REASONING", "PROPAGATION_CHAIN", "FAULT_LOCALIZATION", "ROOT_CAUSE_REASONING_PARTIAL", "ROOT_CAUSE_PROXIMITY", "ROOT_CAUSE_PROXIMITY_FP"]

SEMANTIC_EVAL_CRITERIA = ["PROPAGATION_CHAIN", "FAULT_LOCALIZATION", "ROOT_CAUSE_REASONING_PARTIAL", "ROOT_CAUSE_PROXIMITY", "ROOT_CAUSE_PROXIMITY_FP"]


def get_eval_prompt_value(criterion: str):
    var_name = f"{criterion}_PROMPT"
    return getattr(prompts, var_name)

def get_eval_output_format_value(criterion: str):
    var_name = f"{criterion}_OUTPUT_FORMAT"
    return getattr(prompts, var_name)


# State definition
class EvaluationState(TypedDict):
    messages: List[BaseMessage]
    ground_truth: Dict[str, Any]
    generated_response: Dict[str, Any]
    incident_id: str
    trial_id: Optional[str]
    args: Optional[argparse.Namespace]


class LAAJAgent:
    def __init__(self, model_name: Optional[str] = None):
        """Initialize the LAAJ evaluation agent with environment configuration."""
        # Override AGENT_LLM_MODEL if model_name is provided
        if model_name:
            os.environ["AGENT_LLM_MODEL"] = model_name
            laaj_logger.info(f"Overriding AGENT_LLM_MODEL with: {model_name}")
        
        # Initialize connection manager
        self.conn_manager = LLMConnectionManager()
        
        # Get LLM instances
        self.agent_llm = self.conn_manager.get_agent_llm()
        
        if not self.agent_llm:
            raise ValueError("Agent LLM not configured. Please set AGENT_LLM_API_KEY or LAAJ_API_KEY or OPENAI_API_KEY")
        
        # In this new architecture, the LLM does not use tools directly.
        # It generates placeholders that we will execute.
        
        # Set up a safe evaluator for mathematical expressions
        self.aeval = Interpreter()
        
        # Get configurations for logging
        configs = self.conn_manager.get_configurations()
        agent_config = configs["agent"]
        
        # Log configuration
        laaj_logger.info(f"LAAJ Agent Configuration:")
        laaj_logger.info(f"  Agent LLM Model: {agent_config.get('model_name', 'Not configured')}")
        laaj_logger.info(f"  Agent LLM Temperature: {agent_config.get('temperature', 'Not configured')}")
        laaj_logger.info(f"  Agent LLM Base URL: {agent_config.get('base_url', 'Default OpenAI')}")
        laaj_logger.info(f"  Agent LLM Max Tokens: {agent_config.get('max_tokens', 'Default')}")
        laaj_logger.info(f"  Agent LLM Type: {type(self.agent_llm).__name__ if self.agent_llm else 'None'}")
        
        print(f"LAAJ Agent Configuration:")
        print(f"  Agent LLM Model: {agent_config.get('model_name', 'Not configured')}")
        print(f"  Agent LLM Temperature: {agent_config.get('temperature', 'Not configured')}")
        print(f"  Agent LLM Base URL: {agent_config.get('base_url', 'Default OpenAI')}")
        print(f"  Agent LLM Max Tokens: {agent_config.get('max_tokens', 'Default')}")
        print(f"  Agent LLM Type: {type(self.agent_llm).__name__ if self.agent_llm else 'None'}")
        print()
        
        # Build the graph
        self.graph = self._build_graph()

    def _build_incident_guidance(self, incident_id: Optional[str]) -> str:
        """Return formatted incident-specific guidance, if any."""
        if not incident_id:
            return ""
        bullet_lines_fully_correct = ""
        bullet_lines_paritally_correct = ""

        guidance_fully_correct = prompts.INCIDENT_SPECIFIC_FULLY_CORRECT_REASONING.get(str(incident_id))
        if guidance_fully_correct:
            instruction = prompts.FULLY_CORRECT_REASONING_FEW_SHOT
            if isinstance(guidance_fully_correct, (list, tuple)):
                bullet_lines_fully_correct = instruction + "\n".join(f"- {item}" for item in guidance_fully_correct if item)
            else:
                bullet_lines_fully_correct = instruction + f"- {guidance_fully_correct}"


        guidance_partially_correct = prompts.INCIDENT_SPECIFIC_PARTIALLY_CORRECT_REASONING.get(str(incident_id))
        if guidance_partially_correct:
            instruction = prompts.PARTIALLY_CORRECT_REASONING_FEW_SHOT
            if isinstance(guidance_partially_correct, (list, tuple)):
                bullet_lines_paritally_correct = instruction + "\n".join(f"- {item}" for item in guidance_partially_correct if item)
            else:
                bullet_lines_paritally_correct = instruction + f"- {guidance_partially_correct}"

        return f"{bullet_lines_fully_correct}\n{bullet_lines_paritally_correct}"

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow as a deterministic pipeline."""
        workflow = StateGraph(EvaluationState)
        
        # Add nodes
        workflow.add_node("llm_reasoning_node", self._llm_reasoning_node)
        workflow.add_node("deterministic_calculation_node", self._deterministic_calculation_node)
        
        # Set entry point
        workflow.set_entry_point("llm_reasoning_node")
        
        # Add edges
        workflow.add_edge("llm_reasoning_node", "deterministic_calculation_node")
        workflow.add_edge("deterministic_calculation_node", END)
        
        return workflow.compile()

    async def _llm_reasoning_node(self, state: EvaluationState) -> Dict[str, List[BaseMessage]]:
        """
        First node: Calls the LLM to get the complete reasoning and JSON structure,
        with mathematical calculations left as placeholders. Includes retry logic for rate limiting.
        """
        laaj_logger.info(f"Invoking LLM for reasoning for incident {state.get('incident_id')}, trial {state.get('trial_id')}")
        

        if state["args"].eval_criteria:
            SELECTED_EVAL_CRITERIA = []
            for criterion in EVAL_CRITERIA:
                if criterion in state["args"].eval_criteria:
                    SELECTED_EVAL_CRITERIA.append(criterion)
        else:
            SELECTED_EVAL_CRITERIA = EVAL_CRITERIA
        
        if "ROOT_CAUSE_REASONING" in SELECTED_EVAL_CRITERIA:
            incident_guidance = self._build_incident_guidance(state.get("incident_id"))
        else:
            incident_guidance = ""
        eval_prompt = prompts.EVALUATE_PROMPT_TEMPLATE.format(
            ground_truth=json.dumps(state["ground_truth"], indent=2),
            generated_response=json.dumps(state["generated_response"], indent=2),
            incident_specific_guidance=incident_guidance,
        )

        eval_prompts = {}
        eval_output_formats = {}
        criterion_index = 1
        for criterion in EVAL_CRITERIA:
            if criterion in SELECTED_EVAL_CRITERIA:
                
                if criterion == "ROOT_CAUSE_REASONING":
                    if "ROOT_CAUSE_ENTITY" not in SELECTED_EVAL_CRITERIA:
                        entity_correctness_steps=prompts.ENTITY_CORRECTNESS_STEPS
                    else:
                        entity_correctness_steps=""
                    eval_prompts[criterion] = get_eval_prompt_value(criterion).format(id=criterion_index, entity_correctness_steps=entity_correctness_steps)
                elif criterion == "ROOT_CAUSE_REASONING_PARTIAL":
                    if "ROOT_CAUSE_REASONING" not in SELECTED_EVAL_CRITERIA:
                        if "ROOT_CAUSE_ENTITY" not in SELECTED_EVAL_CRITERIA:
                            entity_and_reasoning_steps=prompts.ENTITY_CORRECTNESS_STEPS+"\n"+prompts.REASONING_CORRECTNESS_STEPS
                        else:
                            entity_and_reasoning_steps=prompts.REASONING_CORRECTNESS_STEPS
                    else:
                        entity_and_reasoning_steps=""
                    eval_prompts[criterion] = get_eval_prompt_value(criterion).format(id=criterion_index, entity_and_reasoning_steps=entity_and_reasoning_steps)
                elif criterion == "ROOT_CAUSE_ENTITY_K":
                    eval_prompts[criterion] = get_eval_prompt_value(criterion).format(id=criterion_index, k=state["args"].k)
                else:
                    eval_prompts[criterion] = get_eval_prompt_value(criterion).format(id=criterion_index)
                eval_output_formats[criterion] = get_eval_output_format_value(criterion)
                criterion_index += 1
            else:
                eval_prompts[criterion] = ""
                eval_output_formats[criterion] = ""
            if criterion in SEMANTIC_EVAL_CRITERIA:
                semantic_grouping = prompts.SEMANTIC_GROUPING_PROMPT
            else:
                semantic_grouping = prompts.NO_SEMANTIC_GROUPING_PROMPT

        system_prompt = prompts.LAAJ_SYSTEM_PROMPT.format(semantic_grouping=semantic_grouping,root_cause_entity=eval_prompts["ROOT_CAUSE_ENTITY"], root_cause_entity_k=eval_prompts["ROOT_CAUSE_ENTITY_K"], root_cause_reasoning=eval_prompts["ROOT_CAUSE_REASONING"], propagation_chain=eval_prompts["PROPAGATION_CHAIN"], fault_localization=eval_prompts["FAULT_LOCALIZATION"], root_cause_reasoning_partial=eval_prompts["ROOT_CAUSE_REASONING_PARTIAL"],root_cause_proximity=eval_prompts["ROOT_CAUSE_PROXIMITY"],root_cause_proximity_fp=eval_prompts["ROOT_CAUSE_PROXIMITY_FP"],root_cause_entity_output_format=eval_output_formats["ROOT_CAUSE_ENTITY"],root_cause_entity_k_output_format=eval_output_formats["ROOT_CAUSE_ENTITY_K"],root_cause_reasoning_output_format=eval_output_formats["ROOT_CAUSE_REASONING"],propagation_chain_output_format=eval_output_formats["PROPAGATION_CHAIN"],fault_localization_output_format=eval_output_formats["FAULT_LOCALIZATION"],root_cause_reasoning_partial_output_format=eval_output_formats["ROOT_CAUSE_REASONING_PARTIAL"],root_cause_proximity_output_format=eval_output_formats["ROOT_CAUSE_PROXIMITY"],root_cause_proximity_fp_output_format=eval_output_formats["ROOT_CAUSE_PROXIMITY_FP"])

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=eval_prompt),
        ]
        
        # Prepare metadata for observability
        metadata_config = {
            "metadata": {
                "incident_id": state.get("incident_id"),
                "trial_id": state.get("trial_id"),
            }
        }
        metadata_config["metadata"] = {k: v for k, v in metadata_config["metadata"].items() if v is not None}
        
        # Add a timeout to the configuration
        metadata_config["configurable"] = {"request_timeout": 300} # 5-minute timeout

        max_retries = 5
        retry_delay_seconds = 70
        api_timeout_seconds = 300  # 5-minute timeout for API calls

        for attempt in range(max_retries):
            try:
                # The LLM has no tools to call, it just generates the text
                # Wrap the API call in asyncio timeout
                response = await asyncio.wait_for(
                    self.agent_llm.ainvoke(messages, config=metadata_config),
                    timeout=api_timeout_seconds
                )

                content_str = response.content
                if "```json" in content_str:
                    content_str = content_str.split("```json")[1].split("```")[0].strip()
            
                # Step 1: Parse the JSON string from the LLM
                data = json.loads(content_str)

                laaj_logger.info(f"LLM call successful for incident {state.get('incident_id')}, trial {state.get('trial_id')}")
                return {"messages": [response]}

            except asyncio.TimeoutError:
                laaj_logger.error(
                    f"API call timed out after {api_timeout_seconds}s for incident {state.get('incident_id')}, "
                    f"trial {state.get('trial_id')} (attempt {attempt + 1}/{max_retries})"
                )
                if attempt + 1 >= max_retries:
                    laaj_logger.error(f"Max retries reached for incident {state.get('incident_id')} due to timeouts.")
                    error_content = f'{{"error": "LLM call failed due to persistent timeouts"}}'
                    return {"messages": [AIMessage(content=error_content)]}
                
                # Wait a bit before retrying after timeout
                await asyncio.sleep(30)  # Shorter wait after timeout

            except openai.RateLimitError as e:
                laaj_logger.warning(
                    f"Rate limit hit for incident {state.get('incident_id')}, trial {state.get('trial_id')} "
                    f"(attempt {attempt + 1}/{max_retries}). Waiting {retry_delay_seconds}s."
                )
                if attempt + 1 >= max_retries:
                    laaj_logger.error(f"Max retries reached for incident {state.get('incident_id')}. Failing this evaluation.")
                    error_content = f'{{"error": "LLM call failed due to persistent rate limiting: {e}"}}'
                    return {"messages": [AIMessage(content=error_content)]}
                
                await asyncio.sleep(retry_delay_seconds)
            
            except json.JSONDecodeError as e:
                laaj_logger.warning(
                    f"LaaJ generated invalid JSON {state.get('incident_id')}, trial {state.get('trial_id')} "
                    f"(attempt {attempt + 1}/{max_retries}). Waiting {retry_delay_seconds}s."
                )
                if attempt + 1 >= max_retries:
                    laaj_logger.error(f"Max retries reached for incident {state.get('incident_id')}. Failing this evaluation.")
                    error_content = f'{{"error": "LLM call failed due to invalid json response: {e}"}}'
                    return {"messages": [AIMessage(content=error_content)]}
                
                await asyncio.sleep(retry_delay_seconds)

            except Exception as e:
                laaj_logger.error(f"Unhandled error during LLM call for incident {state.get('incident_id')}: {e}", exc_info=True)
                error_content = f'{{"error": "LLM call failed with an unexpected error: {e}"}}'
                return {"messages": [AIMessage(content=error_content)]}
            


        # This should theoretically not be reached if the loop handles all cases
        final_error_content = f'{{"error": "LLM call failed after all retries for incident {state.get("incident_id")}"}}'
        return {"messages": [AIMessage(content=final_error_content)]}

    def _deterministic_calculation_node(self, state: EvaluationState) -> Dict[str, List[BaseMessage]]:
        """
        Second node: Deterministically finds all `calculator_tool` placeholders,
        evaluates the expressions, and replaces them with the results. Also computes
        final scores based on the outcomes of those calculations.
        """
        laaj_logger.info("Starting deterministic calculation node.")
        last_message = state["messages"][-1]
        content_str = last_message.content
        
        try:
            # Step 0: Clean the string to remove markdown code fences
            if "```json" in content_str:
                content_str = content_str.split("```json")[1].split("```")[0].strip()
            
            # Step 1: Parse the JSON string from the LLM
            data = json.loads(content_str)
            
            # Check if the data is a list, as expected. If not, it's likely an error from the LLM node.
            if not isinstance(data, list):
                if isinstance(data, dict) and "error" in data:
                    laaj_logger.error(f"LLM reasoning node returned an error: {data['error']}")
                else:
                    laaj_logger.error(f"Unexpected data format from LLM. Expected a list, got {type(data).__name__}.")
                # Since the content is already an error or unexpected, pass it through.
                return {"messages": [last_message]}
            
            # Step 2: Recursively find and evaluate all placeholders
            # This regex is simpler as it works on dictionary values, not a raw JSON string.
            pattern = re.compile(r'^calculator_tool\(expression=["\']([^"\']+)["\']\)$')

            def find_and_replace(obj):
                if isinstance(obj, dict):
                    for key, value in obj.items():
                        obj[key] = find_and_replace(value)
                elif isinstance(obj, list):
                    for i, item in enumerate(obj):
                        obj[i] = find_and_replace(item)
                elif isinstance(obj, str):
                    match = pattern.match(obj)
                    if match:
                        expression = match.group(1)
                        try:
                            # Use a safe evaluator to compute the result
                            result = self.aeval.eval(expression)
                            laaj_logger.info(f"Evaluated '{expression}' to {result}")
                            return result
                        except Exception as e:
                            laaj_logger.error(f"Failed to evaluate expression '{expression}': {e}")
                            return f"Error evaluating: {expression}"
                return obj

            evaluated_data = find_and_replace(data)

            final_content_str = json.dumps(evaluated_data, indent=2)
            final_message = AIMessage(content=final_content_str, id=last_message.id)
            laaj_logger.info("All calculations performed and final JSON generated.")
            
        except json.JSONDecodeError as e:
            laaj_logger.error(f"Failed to parse JSON from LLM: {e}")
            final_message = AIMessage(content=f'{{"error": "Failed to parse JSON from LLM", "original_content": "{content_str}"}}', id=last_message.id)
        except Exception as e:
            laaj_logger.error(f"Error in deterministic calculation node: {e}", exc_info=True)
            final_message = AIMessage(content=f'{{"error": "Error in calculation node: {e}"}}', id=last_message.id)

        return {"messages": [final_message]}
    
    async def evaluate_single(self, args: argparse.Namespace, ground_truth: Dict[str, Any], 
                       generated_response: Dict[str, Any], 
                       incident_id: str,
                       trial_id: Optional[str] = None) -> Dict[str, Any]:
        """Evaluate a single generated response against ground truth."""
        laaj_logger.info(f"Starting evaluation for incident {incident_id}")
        
        initial_state = {
            "messages": [],
            "ground_truth": ground_truth,
            "generated_response": generated_response,
            "incident_id": incident_id,
            "trial_id": trial_id,
            "args": args
        }
        
        # Run the evaluation
        try:
            if trial_id:
                laaj_logger.info(f"Invoking evaluation graph for incident {incident_id}, trial {trial_id}...")
            else:
                laaj_logger.info(f"Invoking evaluation graph for incident {incident_id}...")
            final_state = await self.graph.ainvoke(initial_state)
            
            # Extract the final evaluation from the last message
            last_message = final_state["messages"][-1]
            
            laaj_logger.debug(f"Final state has {len(final_state['messages'])} messages")
            laaj_logger.debug(f"Final message type: {type(last_message).__name__}")
            
            # Parse the JSON response
            try:
                # Find JSON content in the response
                content = last_message.content
                
                # Log first part of response for debugging
                laaj_logger.debug(f"Response starts with: {content[:200]}...")
                
                # Try to find JSON array or object
                json_start = content.find('[')
                json_obj_start = content.find('{')
                
                # Use whichever comes first
                if json_start == -1 or (json_obj_start != -1 and json_obj_start < json_start):
                    json_start = json_obj_start
                    json_end = content.rfind('}') + 1
                else:
                    json_end = content.rfind(']') + 1
                
                if json_start != -1 and json_end > json_start:
                    json_str = content[json_start:json_end]
                    laaj_logger.debug(f"Extracted JSON length: {len(json_str)}")
                    
                    # Clean up any markdown code blocks
                    if "```json" in json_str:
                        json_str = json_str.replace("```json", "").replace("```", "").strip()
                    
                    evaluation_result = json.loads(json_str)
                    laaj_logger.info(f"Successfully parsed evaluation result for incident {incident_id}")
                    
                    # Handle both array and object responses
                    if isinstance(evaluation_result, list):
                        return evaluation_result[0]
                    else:
                        return evaluation_result
                else:
                    laaj_logger.error(f"No valid JSON found in response for incident {incident_id}")
                    laaj_logger.debug(f"Response content: {content[:1000]}...")
                    return {"error": "No valid JSON found in response"}
                    
            except json.JSONDecodeError as e:
                laaj_logger.error(f"Failed to parse JSON for incident {incident_id}: {str(e)}")
                laaj_logger.debug(f"JSON string that failed to parse: {json_str[:500]}...")
                # Try to extract any useful information from the response
                if "scores" in content:
                    laaj_logger.info("Response contains scores but failed JSON parsing")
                return {"error": f"Failed to parse evaluation result: {str(e)}"}
            except Exception as e:
                laaj_logger.error(f"Unexpected error parsing result for incident {incident_id}: {str(e)}", exc_info=True)
                return {"error": f"Unexpected error: {str(e)}"}
                
        except Exception as e:
            laaj_logger.error(f"Error during evaluation of incident {incident_id}: {str(e)}", exc_info=True)
            return {"error": f"Evaluation error: {str(e)}"}
