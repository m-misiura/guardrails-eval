"""
Detector client for benchmarking guardrails using Detectors API.

Supports:
- /api/v1/text/contents endpoint only
- Optional Bearer token authentication
- Clean, minimal interface
"""

import os
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional

import requests


@dataclass
class DetectionResult:
    """Standardized detection result."""

    detected: bool  # Was a threat detected?
    score: float  # Highest detection score (0-1)
    detection_type: str  # Primary detection type
    detections: List[Dict[str, Any]] = field(default_factory=list)  # All detections
    error: Optional[str] = None  # Error message if detection failed
    latency_ms: float = 0.0  # Request latency in milliseconds


class DetectorClient:
    """
    Client for Detectors API /api/v1/text/contents endpoint.

    Supports:
    - ContentAnalysisHttpRequest: {"contents": [str], "detector_params": {}}
    - Response: List[List[ContentAnalysisResponse]]
    - Optional Bearer token authentication from environment variable
    """

    def __init__(self, name: str, config: Dict[str, Any]):
        self.name = name
        self.config = config
        self.logger = logging.getLogger(f"detector.{name}")

        # Required config
        self.api_url = config["api_url"]
        self.detector_id = config["detector_id"]

        # Optional config
        self.threshold = config.get("threshold", 0.0)
        self.timeout = config.get("timeout", 30)

        # Optional bearer token authentication
        self.auth_token_env = config.get("auth_token_env", None)

        self.logger.info(
            f"Initialized {name} (detector_id={self.detector_id}) at {self.api_url}/api/v1/text/contents "
            f"(auth: {'yes' if self.auth_token_env else 'no'})"
        )

    def _get_headers(self) -> Dict[str, str]:
        """Build request headers with optional bearer token and required detector-id."""
        headers = {
            "Content-Type": "application/json",
            "accept": "application/json",
            "detector-id": self.detector_id,
        }

        # Add bearer token if configured
        if self.auth_token_env:
            token = os.getenv(self.auth_token_env)
            if token:
                headers["Authorization"] = f"Bearer {token}"
                self.logger.debug(f"Using bearer token from {self.auth_token_env}")
            else:
                self.logger.warning(
                    f"Auth configured but env var {self.auth_token_env} not set"
                )

        return headers

    def detect(self, prompt: str) -> DetectionResult:
        """Run detection on a single prompt using /api/v1/text/contents endpoint."""
        start_time = time.time()

        try:
            url = f"{self.api_url}/api/v1/text/contents"
            headers = self._get_headers()

            # ContentAnalysisHttpRequest
            body = {
                "contents": [prompt],
                "detector_params": {},
            }

            # Make request
            self.logger.debug(f"POST {url} (detector={self.name})")
            response = requests.post(
                url, headers=headers, json=body, timeout=self.timeout
            )
            response.raise_for_status()

            data = response.json()

            # Calculate latency
            latency_ms = (time.time() - start_time) * 1000

            # Parse response
            result = self._parse_response(data)
            result.latency_ms = latency_ms
            return result

        except requests.exceptions.RequestException as e:
            latency_ms = (time.time() - start_time) * 1000
            self.logger.error(f"Request failed: {e}")
            return DetectionResult(
                detected=False,
                score=0.0,
                detection_type="error",
                error=str(e),
                latency_ms=latency_ms,
            )
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            self.logger.error(f"Unexpected error: {e}")
            return DetectionResult(
                detected=False,
                score=0.0,
                detection_type="error",
                error=str(e),
                latency_ms=latency_ms,
            )

    def _parse_response(self, data: Any) -> DetectionResult:
        """
        Parse /api/v1/text/contents response.

        Response format: List[List[ContentAnalysisResponse]]
        """
        detections = []
        if isinstance(data, list) and len(data) > 0:
            content_detections = data[0]
            if isinstance(content_detections, list):
                detections = content_detections
        if detections:
            max_score = max(d.get("score", 0.0) for d in detections)
            detected = max_score >= self.threshold
            primary_type = detections[0].get("detection_type", "unknown")
            return DetectionResult(
                detected=detected,
                score=max_score,
                detection_type=primary_type,
                detections=detections,
            )
        else:
            return DetectionResult(
                detected=False,
                score=0.0,
                detection_type="none",
                detections=[],
            )


def create_detector_client(
    name: str, detector_config: Dict[str, Any]
) -> DetectorClient:
    """
    Create detector client from configuration.

    Args:
        name: Detector name
        detector_config: Configuration dict (expects 'config' key with detector settings)

    Returns:
        DetectorClient instance
    """
    config = detector_config.get("config", {})
    return DetectorClient(name, config)
