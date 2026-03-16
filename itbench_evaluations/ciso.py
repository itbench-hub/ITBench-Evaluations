"""CISO domain evaluation - OPA-based compliance checking.

OPA (Open Policy Agent) evaluation for Kubernetes compliance scenarios.
"""

import json
import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("itbench_evaluations.ciso")

# CISO evaluation criteria (deterministic OPA checks)
CISO_EVAL_CRITERIA = [
    "required_files_exist",
    "fetch_generates_data",
    "violation_detection",
    "compliance_validation",
]


class CISOEvaluator:
    """OPA-based evaluator for CISO compliance scenarios.

    This evaluator runs fetch.sh and policy.rego scripts created by the agent,
    then validates them using OPA against both violating and compliant resources.
    """

    REQUIRED_FILES = ["fetch.sh", "policy.rego"]
    TIMEOUT_SECONDS = 60

    def __init__(self):
        """Initialize CISO evaluator."""
        pass

    async def evaluate_single(
        self,
        scenario_dir: Path,
        agent_workdir: Path,
        scenario_id: str,
        trial_id: str = "0",
    ) -> Dict[str, Any]:
        """Evaluate a single CISO agent output.

        Args:
            scenario_dir: Path to scenario containing static-resources-compliant/
            agent_workdir: Path to agent's working directory with fetch.sh and policy.rego
            scenario_id: Scenario identifier
            trial_id: Trial number/identifier

        Returns:
            Dict with scores matching SRE/FinOps result format
        """
        scenario_dir = Path(scenario_dir)
        agent_workdir = Path(agent_workdir)

        # Verify compliant resources exist
        compliant_resources = scenario_dir / "static-resources-compliant"
        if not compliant_resources.exists():
            logger.error(f"Compliant resources not found: {compliant_resources}")
            return self._create_error_result(
                scenario_id, trial_id, "Compliant resources not found"
            )

        scores = {}
        details = {}
        temp_dir = None

        try:
            # Criterion 1: Check required files exist
            files_exist, files_details = self._check_required_files(agent_workdir)
            scores["required_files_exist"] = {
                "calculation": 1 if files_exist else 0,
                "justification": files_details,
                "details": {},
            }

            if not files_exist:
                return self._create_result(scenario_id, trial_id, scores, details)

            # Create temporary working directory
            temp_dir = tempfile.mkdtemp(prefix="ciso_eval_")
            work_path = Path(temp_dir) / "workdir"
            shutil.copytree(agent_workdir, work_path)

            details["temp_dir"] = temp_dir
            details["work_path"] = str(work_path)

            # Criterion 2: Execute fetch.sh and check for collected_data.json
            fetch_success, fetch_details = self._run_fetch_script(work_path)
            scores["fetch_generates_data"] = {
                "calculation": 1 if fetch_success else 0,
                "justification": fetch_details,
                "details": {},
            }

            if not fetch_success:
                return self._create_result(scenario_id, trial_id, scores, details)

            # Criterion 3: OPA violation detection (should return false)
            violation_result, violation_details = self._run_opa_eval(work_path)
            violation_pass = violation_result is False
            scores["violation_detection"] = {
                "calculation": 1 if violation_pass else 0,
                "justification": violation_details,
                "details": {"opa_result": violation_result},
            }

            details["violation_check_result"] = violation_result

            # Criterion 4: Replace resources and verify compliance
            replace_success = self._replace_with_compliant_resources(
                work_path, compliant_resources
            )
            details["compliant_resources_replaced"] = replace_success

            if not replace_success:
                scores["compliance_validation"] = {
                    "calculation": 0,
                    "justification": "Failed to replace with compliant resources",
                    "details": {},
                }
                return self._create_result(scenario_id, trial_id, scores, details)

            # Re-run fetch.sh with compliant resources
            (work_path / "collected_data.json").unlink(missing_ok=True)
            fetch_success_2, _ = self._run_fetch_script(work_path)

            if not fetch_success_2:
                scores["compliance_validation"] = {
                    "calculation": 0,
                    "justification": "fetch.sh failed with compliant resources",
                    "details": {},
                }
                return self._create_result(scenario_id, trial_id, scores, details)

            # Run OPA again (should return true for compliant resources)
            compliance_result, compliance_details = self._run_opa_eval(work_path)
            compliance_pass = compliance_result is True
            scores["compliance_validation"] = {
                "calculation": 1 if compliance_pass else 0,
                "justification": compliance_details,
                "details": {"opa_result": compliance_result},
            }

            details["compliance_check_result"] = compliance_result

        except Exception as e:
            logger.error(f"Evaluation failed for {scenario_id}/{trial_id}: {e}")
            return self._create_error_result(scenario_id, trial_id, str(e))

        finally:
            # Cleanup temporary directory
            if temp_dir and Path(temp_dir).exists():
                shutil.rmtree(temp_dir, ignore_errors=True)

        return self._create_result(scenario_id, trial_id, scores, details)

    def _create_result(
        self,
        scenario_id: str,
        trial_id: str,
        scores: Dict[str, Any],
        details: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Create result dict matching SRE/FinOps format."""
        return {
            "incident_id": scenario_id,  # Match SRE/FinOps naming
            "trial_id": trial_id,
            "scores": scores,
            "details": details,
        }

    def _create_error_result(
        self, scenario_id: str, trial_id: str, error: str
    ) -> Dict[str, Any]:
        """Create error result."""
        return {
            "incident_id": scenario_id,
            "trial_id": trial_id,
            "scores": {
                criterion: {"calculation": 0, "justification": error, "details": {}}
                for criterion in CISO_EVAL_CRITERIA
            },
            "details": {"error": error},
        }

    def _check_required_files(self, workdir: Path) -> Tuple[bool, str]:
        """Check if required files exist."""
        missing = [f for f in self.REQUIRED_FILES if not (workdir / f).exists()]

        if missing:
            return False, f"Missing files: {', '.join(missing)}"
        return True, "All required files exist"

    def _run_fetch_script(self, workdir: Path) -> Tuple[bool, str]:
        """Execute fetch.sh and check for collected_data.json."""
        fetch_script = workdir / "fetch.sh"

        # Make executable
        try:
            os.chmod(fetch_script, 0o755)
        except Exception as e:
            return False, f"Failed to make fetch.sh executable: {e}"

        # Run fetch.sh
        try:
            result = subprocess.run(
                ["./fetch.sh"],
                cwd=workdir,
                capture_output=True,
                text=True,
                timeout=self.TIMEOUT_SECONDS,
            )

            if result.returncode != 0:
                return (
                    False,
                    f"Exit code: {result.returncode}; stderr: {result.stderr[:200]}",
                )

            # Check if collected_data.json was created
            data_file = workdir / "collected_data.json"
            if not data_file.exists():
                return False, "collected_data.json not created"

            # Validate JSON
            try:
                with open(data_file) as f:
                    json.load(f)
            except json.JSONDecodeError as e:
                return False, f"collected_data.json is not valid JSON: {e}"

            return True, "Exit code: 0; collected_data.json successfully generated"

        except subprocess.TimeoutExpired:
            return False, f"fetch.sh execution timed out after {self.TIMEOUT_SECONDS}s"
        except Exception as e:
            return False, f"fetch.sh execution failed: {e}"

    def _run_opa_eval(self, workdir: Path) -> Tuple[Optional[bool], str]:
        """Run OPA evaluation and return result."""
        try:
            result = subprocess.run(
                [
                    "opa",
                    "eval",
                    "-d",
                    "policy.rego",
                    "-i",
                    "collected_data.json",
                    "data.check.result",
                ],
                cwd=workdir,
                capture_output=True,
                text=True,
                timeout=self.TIMEOUT_SECONDS,
            )

            if result.returncode != 0:
                return None, f"OPA evaluation failed: {result.stderr[:200]}"

            # Parse OPA output
            try:
                output = json.loads(result.stdout)
                opa_result = output.get("result", [{}])[0].get("expressions", [{}])[
                    0
                ].get("value")

                if opa_result is None:
                    return None, "OPA result is null/undefined"

                expected = "false for violations" if opa_result is False else "true for compliance"
                return opa_result, f"OPA result: {opa_result} (expected: {expected})"

            except (json.JSONDecodeError, KeyError, IndexError) as e:
                return None, f"Failed to parse OPA output: {e}"

        except subprocess.TimeoutExpired:
            return None, f"OPA evaluation timed out after {self.TIMEOUT_SECONDS}s"
        except FileNotFoundError:
            return (
                None,
                "OPA not found. Install with: brew install opa (macOS) or see https://www.openpolicyagent.org/",
            )
        except Exception as e:
            return None, f"OPA evaluation error: {e}"

    def _replace_with_compliant_resources(
        self, workdir: Path, compliant_resources: Path
    ) -> bool:
        """Replace static-resources with compliant version."""
        try:
            static_resources = workdir / "static-resources"

            # Remove existing static-resources
            if static_resources.exists():
                shutil.rmtree(static_resources)

            # Copy compliant resources
            shutil.copytree(compliant_resources, static_resources)

            return True
        except Exception as e:
            logger.error(f"Failed to replace resources: {e}")
            return False
