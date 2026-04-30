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

        # TODO: Workaround to align the framework
        # Copy static-resources to agent_workdir
        static_resources = scenario_dir / "static-resources"
        if (agent_workdir / "static-resources").exists():
            shutil.rmtree(agent_workdir / "static-resources")

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

            temp_dir = tempfile.mkdtemp(prefix="ciso_eval_")
            def fetch_and_check(static_resources_dir: Path):
                # Create temporary working directory
                each_scores = {}
                each_details = {}
                work_path = Path(temp_dir) / static_resources_dir.name
                shutil.copytree(agent_workdir, work_path, ignore=shutil.ignore_patterns(".git"))
                shutil.copytree(static_resources_dir, work_path / "static-resources")
                each_details["temp_dir"] = temp_dir
                each_details["work_path"] = str(work_path)

                # TODO: Workaround to align the frameworl
                # Evaluation isolates each run by copying the full snapshot (agent-workdir + static-resources +
                # static-resources-compliant) into a fresh working directory. Ideally AUB (Agent under Bench) would run inside that
                # complete workdir so fetch.sh needs no fixup. Currently AUB runs against snapshot paths directly,
                # so fetch.sh embeds the snapshot path — replaced here at evaluation time.
                fetch_script = work_path / "fetch.sh"
                content = fetch_script.read_text()
                content = content.replace(str(static_resources), "static-resources")
                content = content.replace(str(agent_workdir), ".")
                fetch_script.write_text(content)

                # Criterion 2: Execute fetch.sh and check for collected_data.json
                fetch_success, fetch_details = self._run_fetch_script(work_path)
                each_scores["fetch_generates_data"] = {
                    "calculation": 1 if fetch_success else 0,
                    "justification": fetch_details,
                    "details": {},
                }

                each_details["scores"] = each_scores
                if each_scores["fetch_generates_data"]["calculation"] == 0:
                    return each_details

                # Criterion 3: OPA violation detection (should return false)
                violation_result, violation_details = self._run_opa_eval(work_path)
                violation_pass = violation_result is False
                each_scores["violation_detection"] = {
                    "calculation": 1 if violation_pass else 0,
                    "justification": violation_details,
                    "details": {"opa_result": violation_result},
                }

                each_details["violation_check_result"] = violation_result
                return each_details

            violation_dirs = sorted(scenario_dir.glob("static-resources-violation-*"))
            # Fallback to default static-resources directory (legacy scenarios)
            if not violation_dirs:
                default_dir = scenario_dir / "static-resources"
                if default_dir.exists():
                    violation_dirs = [default_dir]

            violation_details_list = []
            for violation_dir in violation_dirs:
                details_per_violation = fetch_and_check(violation_dir)
                violation_details_list.append(details_per_violation)

            # Aggregate fetch_generates_data scores
            all_fetch_success = all(
                d["scores"]["fetch_generates_data"]["calculation"] == 1
                for d in violation_details_list
            )
            fetch_justifications = [
                d["scores"]["fetch_generates_data"]["justification"]
                for d in violation_details_list
            ]
            combined_fetch_justification = "; ".join(fetch_justifications) if len(fetch_justifications) > 1 else fetch_justifications[0]
            
            scores["fetch_generates_data"] = {
                "calculation": 1 if all_fetch_success else 0,
                "justification": combined_fetch_justification,
                "details": {},
            }

            if not all_fetch_success:
                return self._create_result(scenario_id, trial_id, scores, details)

            # Aggregate violation_detection scores
            # All violations must pass for overall success
            all_violations_pass = all(
                d["scores"]["violation_detection"]["calculation"] == 1
                for d in violation_details_list
            )
            
            # Collect justifications from all violation checks
            violation_justifications = [
                d["scores"]["violation_detection"]["justification"]
                for d in violation_details_list
            ]
            combined_violation_justification = "; ".join(violation_justifications) if len(violation_justifications) > 1 else violation_justifications[0]
            
            # Collect all OPA results
            opa_results = [d["violation_check_result"] for d in violation_details_list]
            
            scores["violation_detection"] = {
                "calculation": 1 if all_violations_pass else 0,
                "justification": combined_violation_justification,
                "details": {"opa_result": opa_results[0] if len(opa_results) == 1 else opa_results},
            }
            
            # For backward compatibility: single result if only one dir, array if multiple
            if len(violation_details_list) == 1:
                details["violation_check_result"] = violation_details_list[0]["violation_check_result"]
            else:
                details["violation_check_result"] = opa_results
            
            details["violation_details_list"] = violation_details_list

            # Criterion 4: Replace resources and verify compliance
            compliance_details_result = fetch_and_check(compliant_resources)
            
            # Check if fetch succeeded for compliant resources
            compliance_fetch_success = (
                compliance_details_result["scores"]["fetch_generates_data"]["calculation"] == 1
            )
            
            if not compliance_fetch_success:
                scores["compliance_validation"] = {
                    "calculation": 0,
                    "justification": compliance_details_result["scores"]["fetch_generates_data"]["justification"],
                    "details": {},
                }
                return self._create_result(scenario_id, trial_id, scores, details)

            # Use the compliance check result from fetch_and_check
            compliance_result = compliance_details_result["violation_check_result"]
            compliance_pass = compliance_result is True
            scores["compliance_validation"] = {
                "calculation": 1 if compliance_pass else 0,
                "justification": compliance_details_result["scores"]["violation_detection"]["justification"],
                "details": {"opa_result": compliance_result},
            }

            details["compliance_check_result"] = compliance_result
            details["compliance_details"] = compliance_details_result

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
        for criteria in CISO_EVAL_CRITERIA:
            if criteria not in scores:
                scores[criteria] = {
                    "calculation": 0,
                    "justification": "Not attempted because the prerequisite was not met.",
                    "details": {},
                }

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
