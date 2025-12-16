import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Any
import sys
from datetime import datetime
import logging
import asyncio

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
logs_dir_path = os.getenv("AGENT_ANALYTICS_LOGS_DIR","logs/")
os.makedirs(logs_dir_path, exist_ok=True)

# Set up custom logging for LAAJ agent
laaj_logger = logging.getLogger("LAAJ")
laaj_logger.setLevel(logging.DEBUG)

# Create a file handler for LAAJ-specific logs
laaj_log_file = os.path.join(logs_dir_path, f"laaj_agent_logs_{timestamp}.log")
file_handler = logging.FileHandler(laaj_log_file)
file_handler.setLevel(logging.DEBUG)

# Create a console handler for important logs
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# Create formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# Add handlers to logger
laaj_logger.addHandler(file_handler)
laaj_logger.addHandler(console_handler)

# add observability
# Initialize agent analytics SDK BEFORE other imports to ensure proper instrumentation
from agent_analytics.instrumentation import agent_analytics_sdk


log_filename = f"agent_analytics_sdk_logs_{timestamp}"

# Initialize logging with agent_analytics_sdk - this must happen before other imports
agent_analytics_sdk.initialize_logging(
    logs_dir_path=logs_dir_path,
    log_filename=log_filename
)
from loader import load_ground_truth, find_agent_outputs
from aggregator import calculate_statistics
from laaj_agent import LAAJAgent
from laaj_agent import EVAL_CRITERIA


def main():
    """Main entry point for the LAAJ agent."""
    parser = argparse.ArgumentParser(description="LLM-as-a-Judge Agent for RCA Evaluation")
    parser.add_argument("--gt", required=True, help="Path to ground truth JSON file")
    parser.add_argument("--incident", help="Incident ID to evaluate. If not provided, all incidents in the agent-outputs directory are evaluated.")
    parser.add_argument("--model", default="gpt-4-turbo-preview", help="Model to use for evaluation")
    parser.add_argument("--agent-outputs", required=True, help="Directory containing generated agent outputs")
    parser.add_argument("--k", default="3", help="k value for ROOT_CAUSE_ENTITY_K metric")
    parser.add_argument("--eval_criteria", nargs="+", choices=EVAL_CRITERIA,  help="Criteria for evaluation")
    
    args = parser.parse_args()
    
    # Load ground truth
    ground_truths = load_ground_truth(args.gt)
    
    # Determine which incidents to evaluate
    incident_ids_to_evaluate = []
    if args.incident:
        if args.incident not in ground_truths:
            print(f"Error: Incident {args.incident} not found in ground truth file")
            sys.exit(1)
        
        agent_output_dir = Path(args.agent_outputs) / args.incident
        if not agent_output_dir.exists() or not agent_output_dir.is_dir():
            print(f"Error: Output directory for incident {args.incident} not found in {args.agent_outputs}")
            sys.exit(1)
            
        incident_ids_to_evaluate.append(args.incident)
    else:
        # Evaluate all incidents found in the output directory
        output_dir = Path(args.agent_outputs)
        for incident_dir in output_dir.iterdir():
            if incident_dir.is_dir():
                incident_id = incident_dir.name
                if incident_id in ground_truths:
                    incident_ids_to_evaluate.append(incident_id)
                else:
                    laaj_logger.warning(f"Skipping incident '{incident_id}' found in agent outputs but not in ground truth.")

    if not incident_ids_to_evaluate:
        print("No valid incidents found to evaluate.")
        sys.exit(0)
        
    print(f"Found {len(incident_ids_to_evaluate)} incidents to evaluate: {', '.join(incident_ids_to_evaluate)}")
    
    # Check for API key before initializing
    if not (os.getenv("AGENT_LLM_API_KEY") or os.getenv("LAAJ_API_KEY") or os.getenv("OPENAI_API_KEY")):
        print("Error: No API key found. Please set one of the following environment variables:")
        print("  - AGENT_LLM_API_KEY (for agent LLM)")
        print("  - LAAJ_API_KEY (fallback)")
        print("  - OPENAI_API_KEY (fallback)")
        print("\nYou can also configure separate LLMs for agent and tools:")
        print("  # Agent LLM configuration")
        print("  export AGENT_LLM_API_KEY='your-api-key'")
        print("  export AGENT_LLM_MODEL='gpt-4-turbo-preview'")
        print("  export AGENT_LLM_BASE_URL='http://your-litellm-proxy:4000'")
        print("\n  # Tool LLM configuration (optional, falls back to agent LLM)")
        print("  export TOOL_LLM_API_KEY='your-api-key'")
        print("  export TOOL_LLM_MODEL='gpt-3.5-turbo'")
        print("  export TOOL_LLM_BASE_URL='http://your-litellm-proxy:4000'")
        sys.exit(1)
    
    # Initialize the agent
    agent = LAAJAgent(model_name=args.model if args.model != "gpt-4-turbo-preview" else None)
    
    # Run the evaluations asynchronously
    asyncio.run(run_evaluations(agent, args, incident_ids_to_evaluate, ground_truths))


async def run_evaluations(agent: LAAJAgent, args: argparse.Namespace, incident_ids: List[str], ground_truths: Dict[str, Any]):
    """Gather and run all evaluations concurrently."""
    
    CONCURRENCY_LIMIT = 10  # Limit concurrent requests to avoid rate limiting
    sem = asyncio.Semaphore(CONCURRENCY_LIMIT)

    async def throttled_evaluate_single(gt_for_eval, generated_response, incident_id, trial_num):
        async with sem:
            return await agent.evaluate_single(args,
                gt_for_eval, generated_response, incident_id, trial_id=str(trial_num)
            )

    evaluation_coroutines = []
    contexts = []
    bad_runs_per_incident = {}

    for incident_id in incident_ids:
        laaj_logger.info(f"\n{'='*20} Preparing Incident: {incident_id} {'='*20}")
        print(f"\nPreparing Incident: {incident_id}")

        gt_incident = ground_truths[incident_id]
        trial_outputs, bad_runs = await find_agent_outputs(args.agent_outputs, incident_id)
        bad_runs_per_incident[incident_id] = bad_runs
        
        if not trial_outputs:
            laaj_logger.warning(f"No valid outputs found for incident {incident_id}, skipping evaluation for it.")
            print(f"Warning: No valid outputs found for incident {incident_id}, skipping.")
            continue
        
        for trial_data in trial_outputs:
            trial_num = trial_data["trial"]
            generated_response = trial_data["output"]

            if generated_response is None:
                laaj_logger.warning(f"Empty output found for incident {incident_id}, skipping evaluation for it.")
                print(f"Warning: Empty output found for incident {incident_id}, skipping.")
                bad_runs_per_incident[incident_id] += 1
                continue
            
            laaj_logger.info(f"Queueing evaluation for Trial {trial_num}")
            
            gt_for_eval = {
                "groups": gt_incident.get("groups", []),
                "propagations": gt_incident.get("propagations", []),
                "aliases": gt_incident.get("aliases",[])
            }
            
            coroutine = throttled_evaluate_single(gt_for_eval, generated_response, incident_id, trial_num)
            evaluation_coroutines.append(coroutine)
            contexts.append({"incident_id": incident_id, "trial": trial_num})

    print(f"\nStarting evaluation of {len(evaluation_coroutines)} trials concurrently with a limit of {CONCURRENCY_LIMIT}...")
    results_flat = await asyncio.gather(*evaluation_coroutines)
    print("All evaluations complete.")

    # Associate results with their context and group by incident
    results_by_incident = {}
    for i, result in enumerate(results_flat):
        context = contexts[i]
        incident_id = context["incident_id"]
        trial_num = context["trial"]
        
        result["trial"] = trial_num
        
        if incident_id not in results_by_incident:
            results_by_incident[incident_id] = []
        results_by_incident[incident_id].append(result)

        # Print summary for this trial
        if "scores" in result:
            print(f"Incident {incident_id}, Trial {trial_num} Scores:")
            for metric, data in result["scores"].items():
                if isinstance(data, dict) and "score" in data:
                    print(f"  - {metric}: {data['score']}")
        else:
            print(f"Incident {incident_id}, Trial {trial_num}: Error in evaluation - {result.get('error')}")

    all_incidents_results = []
    for incident_id in incident_ids:
        evaluations = results_by_incident.get(incident_id, [])
        bad_runs = bad_runs_per_incident.get(incident_id, 0)
        
        # Only include incidents that had some activity (good or bad runs)
        if not evaluations and bad_runs == 0:
            continue

        all_incidents_results.append({
            "incident_id": incident_id,
            "evaluations": sorted(evaluations, key=lambda x: x['trial']),
            "total_bad_runs": bad_runs,
        })
    
    # Calculate and add statistics
    statistics = calculate_statistics(all_incidents_results)

    # Output all results
    output_data = {
        "ground_truth_file": args.gt,
        "agent_output_directory": args.agent_outputs,
        "statistics": statistics,
        "results": all_incidents_results
    }
    
    # Pretty print the results
    print("\n" + "="*80)
    print("COMPLETE EVALUATION RESULTS:")
    print("="*80)
    print(json.dumps(output_data, indent=2))
    
    # Save to file
    if args.incident:
        output_file = f"laaj_evaluation_incident_{args.incident}.json"
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = f"laaj_evaluation_all_incidents_{timestamp}.json"

    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")

if __name__ == "__main__":
    main()