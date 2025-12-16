# ITBench Evaluations

A toolkit for running **LAAJ (LLM-as-a-Judge)** evaluations on root cause analysis (RCA) agent outputs.  
The runner loads a ground-truth JSON file, repairs malformed agent responses when needed, adjudicates multiple metrics through the LAAJ agent, aggregates pass@1 / macro averages, and emits detailed JSON + log artifacts for each batch of incidents.

---

## Repository layout

| Path | Description |
| --- | --- |
| `runner.py` | CLI entrypoint that orchestrates loading data, running the LAAJ agent asynchronously, writing logs, and aggregating results. |
| `laaj_agent.py` | LangGraph-based workflow that calls the configured LLM with the prompts in `prompts/` to grade each metric. |
| `loader.py` | Discovers incident/trial folders, loads `agent_output.json` or `agent_response.json`, and repairs malformed JSON via `json_fixer.py`. |
| `aggregator.py` | Computes per-incident and macro statistics (mean, stderr, pass@1) for the metrics listed below. |
| `connection.py` | Creates agent LLM clients (OpenAI, Azure OpenAI, or LiteLLM proxy) using environment variables. |
| `prompts/` | Contains the system prompts, per-metric templates, JSON output schemas, and optional incident-specific guidance. |
| `data/` | Example incident output hierarchy (`incident_id/trial/outputs/agent_output.json`). |
| `env_example.txt` | Sample configuration for pointing the agent at a LiteLLM proxy. |

---

## Quick start

1. **Install prerequisites**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

2. **Configure environment variables**
   Copy `env_example.txt` to `.env` or export the variables manually. At minimum you need an API key for the agent LLM.

   | Purpose | Keys (checked in this order) |
   | --- | --- |
   | Agent LLM (required) | `AGENT_LLM_API_KEY`
   | Agent model / endpoint | `AGENT_LLM_MODEL`, `AGENT_LLM_BASE_URL`, `AGENT_LLM_AZURE_DEPLOYMENT`, plus optional `AGENT_LLM_TEMPERATURE`, `AGENT_LLM_MAX_TOKENS` |
   | Logging | `AGENT_ANALYTICS_LOGS_DIR` (defaults to `logs/`) |

3. **Prepare data**
   - **Ground truth JSON** – an array of incidents, each containing at least `id`, `groups`, `propagations`, and optional `aliases`.
   - **Agent outputs** – directory structured as:
     ```
     agent-outputs/
       <incident_id>/
         <trial_number>/
           outputs/agent_output.json
           # or outputs/agent_response.json
     ```
     Each JSON file should match the schema expected by the prompts; malformed files are automatically repaired via `json_fixer.py` (LLM first, simple heuristics second). Unreadable and missing files are counted as bad runs.

---

## Running evaluations

```bash
python runner.py \
  --gt path/to/ground_truth.json \
  --agent-outputs path/to/agent-outputs \
  --k 3 \
  --eval_criteria ROOT_CAUSE_ENTITY ROOT_CAUSE_REASONING
```

- `--incident` limits the run to a single incident directory.
- `--model` overrides `AGENT_LLM_MODEL` for the session.
- `--k` configures the `ROOT_CAUSE_ENTITY_K` metric (default `3`).
- `--eval_criteria` lets you grade a subset of metrics; omit to run all available criteria.
- Evaluations are throttled to 10 concurrent requests to help avoid rate limits.

### Outputs

- Structured results are emitted to `laaj_evaluation_incident_<id>.json` or `laaj_evaluation_all_incidents_<timestamp>.json`.  
  The payload mirrors the console dump:
  ```json
  {
    "statistics": {
      "per_incident": { ... },
      "overall": {
        "root_cause_entity_precision": {"mean": 0.58, "stderr": 0.04, "pass@1": 0.42},
        "total_bad_runs": 3
      }
    },
    "results": [
      {"incident_id": "1", "evaluations": [...], "total_bad_runs": 1}
    ]
  }
  ```
- Logs: general run logs plus agent-facing reasoning logs go to `logs/` (or `AGENT_ANALYTICS_LOGS_DIR`). Each run creates:
  - `laaj_agent_logs_<timestamp>.log`
  - `agent_analytics_sdk_logs_<timestamp>.log`

---

## Metrics covered

`aggregator.py` computes macro-level mean, stderr, and pass@1 (for categorical metrics) for the following:

- `root_cause_entity` (precision/recall/F1 + pass@1): Whether the correct root cause entity was identified
- `root_cause_entity_k` (precision/recall/F1 + pass@1, configurable `k`): Whether the correct root cause entity was identified in the first k model predictions
- `root_cause_reasoning`: Whether the reasoning for the root cause was correct (0, 0.5 or 1).
- `propagation_chain`: Scores the full propagation chain
- `fault_localization_component_identification`: Checks if the model correctly identified the first semantic component to exhibit a significant failure symptom
- `root_cause_reasoning_partial`: Awards partial credit for reasoning if the model correctly analyzed a downstream symptom when it missed the root cause entity.
- `root_cause_proximity` (precision/recall/F1): Compute closeness between model root cause entities and the Ground-Truth (GT) root-cause entities based on distance (number of hops) between the model entity’s component and any GT root-cause component
- `root_cause_proximity_with_fp` (precision/recall/F1): Similar to root_cause_proximity_no_fp but distance is relative to the GT path length
