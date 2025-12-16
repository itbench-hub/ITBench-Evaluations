import json 
from typing import Dict, List, Any, Tuple
from pathlib import Path
from json_fixer import *
import logging
laaj_logger = logging.getLogger("LAAJ.Loader")

def load_ground_truth(gt_file: str) -> Dict[str, Any]:
    """Load ground truth data from JSON file."""
    with open(gt_file, 'r') as f:
        data = json.load(f)
    
    # Convert to dictionary keyed by incident ID
    gt_dict = {}
    for incident in data:
        gt_dict[incident['id']] = incident
    
    return gt_dict


async def find_agent_outputs(output_dir: str, incident_id: str) -> Tuple[List[Dict[str, Any]], int]:
    """Find all agent output files for a given incident and count bad runs."""
    outputs = []
    bad_runs = 0
    incident_dir = Path(output_dir) / incident_id
    
    laaj_logger.info(f"Looking for agent outputs in: {incident_dir}")
    
    if not incident_dir.exists():
        laaj_logger.warning(f"No directory found for incident {incident_id} at {incident_dir}")
        print(f"Warning: No directory found for incident {incident_id}")
        return outputs, bad_runs
    
    # Find all trial directories
    trial_dirs = sorted([d for d in incident_dir.iterdir() if d.is_dir() and d.name.isdigit()])
    laaj_logger.info(f"Found {len(trial_dirs)} trial directories for incident {incident_id}")
    
    for trial_dir in trial_dirs:
        # Check for both possible filenames
        output_file_v1 = trial_dir / "outputs" / "agent_output.json"
        output_file_v2 = trial_dir / "outputs" / "agent_response.json"
        
        output_file = None
        if output_file_v1.exists():
            output_file = output_file_v1
        elif output_file_v2.exists():
            output_file = output_file_v2

        if output_file:
            try:
                with open(output_file, 'r') as f:
                    file_content = f.read()
                    
                # First attempt: try to parse as regular JSON
                try:
                    output_data = json.loads(file_content)
                    
                    # Handle case where the JSON file contains a markdown-wrapped JSON string
                    if isinstance(output_data, str):
                        laaj_logger.debug(f"Output data is a string, attempting to extract JSON from markdown")
                        # Check if it's markdown-wrapped JSON
                        if "```json" in output_data:
                            # Extract JSON from markdown code block
                            json_start = output_data.find("```json") + 7
                            json_end = output_data.rfind("```")
                            if json_start > 6 and json_end > json_start:
                                json_str = output_data[json_start:json_end].strip()
                                output_data = json.loads(json_str)
                                laaj_logger.debug(f"Successfully extracted and parsed JSON from markdown wrapper")
                            else:
                                raise ValueError("Could not extract JSON from markdown wrapper")
                        else:
                            # Try to parse the string as JSON directly
                            output_data = json.loads(output_data)
                            laaj_logger.debug(f"Successfully parsed string as JSON")
                    
                    outputs.append({
                        "trial": int(trial_dir.name),
                        "output": output_data,
                        "trial_dir": trial_dir
                    })
                    laaj_logger.debug(f"Successfully loaded trial {trial_dir.name} from {output_file.name}")
                    
                except (json.JSONDecodeError, ValueError) as json_error:
                    laaj_logger.warning(f"JSON parsing failed for {output_file}: {json_error}")
                    
                    # Try to fix the JSON using LLM first
                    fixed_data = None
                    
                    laaj_logger.info(f"Attempting to fix JSON with LLM for {output_file}")
                    fixed_data = await fix_json_with_llm(file_content, str(output_file))
                    
                    # If LLM failed or is not available, try simple repair
                    if fixed_data is None:
                        laaj_logger.info(f"Attempting simple JSON repair for {output_file}")
                        fixed_data = simple_json_repair(file_content)
                    
                    if fixed_data is not None:
                        outputs.append({
                            "trial": int(trial_dir.name),
                            "output": fixed_data,
                            "trial_dir": trial_dir
                        })
                       
                        laaj_logger.info(f"Successfully fixed and loaded trial {trial_dir.name} from {output_file.name}")
                       
                    else:
                        laaj_logger.error(f"All JSON repair attempts failed for {output_file}. Marking as bad run.")
                        print(f"Error: All JSON repair attempts failed for {output_file}")
                        bad_runs += 1
                        
            except Exception as e:
                laaj_logger.error(f"Error reading {output_file}: {e}. Marking as bad run.")
                print(f"Error reading {output_file}: {e}")
                bad_runs += 1
        else:
            laaj_logger.warning(f"No agent_output.json or agent_response.json found in {trial_dir}. Marking as bad run.")
            bad_runs += 1
    
    laaj_logger.info(f"Successfully loaded {len(outputs)} valid trial outputs for incident {incident_id}, found {bad_runs} bad runs.")
    return outputs, bad_runs