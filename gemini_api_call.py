from __future__ import annotations

import json
import mimetypes
import os
from datetime import date
from pathlib import Path
from typing import Any, Literal, Sequence, Optional

from google import genai
from google.genai import types
from pydantic import BaseModel, ConfigDict, Field, field_validator

Role = Literal["user", "model"]
ExecutionMode = Literal["generate_content", "chat"]


# Local lifecycle metadata used to perform basic validation of model names.
# Unknown models are still allowed because the live API may support names that
# have not yet been added to this local catalog.
MODEL_CATALOG: dict[str, dict[str, Any]] = {
    "gemini-3.1-pro-preview": {
        "family": "pro",
        "preview": True,
        "available_from": date(2026, 2, 19),
        "deprecation_date": None,
        "status": "preview",
    },
    "gemini-3.1-flash-lite-preview": {
        "family": "flash",
        "preview": True,
    },
    "gemini-3-flash-preview": {
        "family": "flash",
        "preview": True,
        "available_from": date(2026, 2, 5),
        "deprecation_date": None,
        "status": "preview",
    },
}

SUPPORTED_UPLOAD_MIME_PREFIXES = ("application/pdf", "audio/", "image/", "text/")


class GeminiConfigurationError(ValueError):
    """Raised when a request configuration is invalid."""


class ValidationReport(BaseModel):
    """Collects validation errors and warnings for request-related objects."""

    model_config = ConfigDict(extra="forbid", strict=True)
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    @property
    def ok(self) -> bool:
        """Return ``True`` when validation produced no errors."""
        return not self.errors

    def raise_if_invalid(self) -> None:
        """Raise ``GeminiConfigurationError`` if any validation errors exist."""
        if self.errors:
            message = "Invalid Gemini request configuration:\n- " + "\n- ".join(self.errors)
            raise GeminiConfigurationError(message)

class StructuredOutputConfig(BaseModel):
    """Controls optional JSON-formatted structured output.

    Attributes:
        enabled: Enables structured-output settings when ``True``.
        json_schema: Optional schema used to constrain JSON generation.
        mime_type: Response MIME type requested from the model.
    """

    model_config = ConfigDict(extra="forbid", strict=True, arbitrary_types_allowed=True)
    enabled: bool = False
    json_schema: Optional[dict[str, Any]] | types.Schema | None = None
    mime_type: str = "application/json"

    def validate(self, report: ValidationReport) -> None:
        """Validate structured-output settings and append issues to ``report``."""
        if not self.enabled:
            return
        if self.mime_type != "application/json":
            report.errors.append("Structured output requires mime_type='application/json'.")
        if self.json_schema is not None and not isinstance(self.json_schema, (dict, types.Schema)):
            report.errors.append("Structured output schema must be a dict or google.genai.types.Schema.")


class AttachmentConfig(BaseModel):
    """Describes a local file that should be uploaded with the request.

    Attributes:
        file_path: Local path to the file on disk.
        mime_type: Optional explicit MIME type. If omitted, it is inferred.
        display_name: Optional human-readable file name sent to the API.
    """

    model_config = ConfigDict(extra="forbid", strict=True)
    file_path: Path
    mime_type: str | None = None
    display_name: str | None = None

    @field_validator("file_path", mode="before")
    @classmethod
    def _coerce_path(cls, value: Any) -> Path:
        """Convert incoming path-like values into ``Path`` objects."""
        return Path(value)

    def resolved_path(self) -> Path:
        """Return the absolute path after expanding ``~`` and resolving symlinks."""
        return Path(self.file_path).expanduser().resolve()

    def resolved_mime_type(self) -> str:
        """Return the explicit or inferred MIME type for this attachment."""
        if self.mime_type:
            return self.mime_type
        guessed, _ = mimetypes.guess_type(str(self.resolved_path()))
        if guessed:
            return guessed
        raise GeminiConfigurationError(f"Could not infer MIME type for '{self.file_path}'.")

    def validate(self, report: ValidationReport) -> None:
        """Validate file existence and whether its MIME type is uploadable."""
        path = self.resolved_path()
        if not path.exists():
            report.errors.append(f"Attachment not found: {path}")
            return
        if not path.is_file():
            report.errors.append(f"Attachment is not a file: {path}")
            return
        mime_type = self.mime_type
        if mime_type is None:
            guessed, _ = mimetypes.guess_type(str(path))
            mime_type = guessed
        if mime_type is None:
            report.errors.append(f"Could not infer MIME type for attachment: {path}")
            return
        if not mime_type.startswith(SUPPORTED_UPLOAD_MIME_PREFIXES):
            report.errors.append(
                f"Unsupported attachment MIME type '{mime_type}' for {path}. "
                "Supported types are PDF, audio, images, and text."
            )


class HistoryTurn(BaseModel):
    """Represents one conversational turn stored in plain text."""

    model_config = ConfigDict(extra="forbid", strict=True)
    role: Role
    text: str

    def validate(self, report: ValidationReport, *, field_name: str) -> None:
        """Validate the role and text content for one history entry."""
        if self.role not in ("user", "model"):
            report.errors.append(f"{field_name}.role must be 'user' or 'model'.")
        if not self.text.strip():
            report.errors.append(f"{field_name}.text must not be empty.")

    def to_content(self) -> types.Content:
        """Convert the turn into the SDK content object expected by Gemini."""
        if self.role == "user":
            return types.UserContent(parts=[types.Part.from_text(text=self.text)])
        return types.ModelContent(parts=[types.Part.from_text(text=self.text)])


class GenerationControls(BaseModel):
    """Holds optional sampling and output-length settings for generation."""

    model_config = ConfigDict(extra="forbid", strict=True)
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    candidate_count: int | None = None
    max_output_tokens: int | None = None

    def validate(self, report: ValidationReport) -> None:
        """Validate numeric ranges for generation parameters."""
        if self.temperature is not None and not 0 <= self.temperature <= 2:
            report.errors.append("temperature must be between 0 and 2.")
        if self.top_p is not None and not 0 <= self.top_p <= 1:
            report.errors.append("top_p must be between 0 and 1.")
        if self.top_k is not None and self.top_k < 1:
            report.errors.append("top_k must be at least 1.")
        if self.candidate_count is not None and self.candidate_count < 1:
            report.errors.append("candidate_count must be at least 1.")
        if self.max_output_tokens is not None and self.max_output_tokens < 1:
            report.errors.append("max_output_tokens must be at least 1.")


class GeminiRequestConfig(BaseModel):
    """Defines all inputs needed to execute a single Gemini request.

    Attributes:
        model_name: Gemini model identifier to call.
        user_input: Main user prompt sent in the current request.
        system_instruction: Optional system-level instruction for the model.
        history: Prior conversation turns to include before the new prompt.
        attachments: Local files uploaded and attached to the prompt.
        structured_output: JSON output configuration.
        enable_google_search: Enables Google Search grounding when supported.
        execution_mode: Chooses between ``generate_content`` and chat mode.
        generation: Sampling and token-limit controls.
        labels: Optional request labels forwarded to the SDK.
    """

    model_config = ConfigDict(extra="forbid", strict=True)
    model_name: str
    user_input: str
    system_instruction: str | None = None
    history: list[HistoryTurn] = Field(default_factory=list)
    attachments: list[AttachmentConfig] = Field(default_factory=list)
    structured_output: StructuredOutputConfig = Field(default_factory=StructuredOutputConfig)
    enable_google_search: bool = False
    execution_mode: ExecutionMode = "generate_content"
    generation: GenerationControls = Field(default_factory=GenerationControls)
    labels: dict[str, str] = Field(default_factory=dict)

    def validate(self, *, today: date | None = None) -> ValidationReport:
        """Validate the request and return a report of errors and warnings.

        Args:
            today: Optional override used for date-based model lifecycle checks.

        Returns:
            A ``ValidationReport`` with all detected issues.
        """
        report = ValidationReport()
        today = today or date.today()

        if not self.model_name.strip():
            report.errors.append("model_name must not be empty.")
        else:
            model_meta = MODEL_CATALOG.get(self.model_name)
            if model_meta is None:
                report.warnings.append(
                    f"Model '{self.model_name}' is not in the local lifecycle catalog. "
                    "Execution may still work if the API accepts it."
                )
            else:
                available_from: date | None = model_meta.get("available_from")
                deprecation_date: date | None = model_meta.get("deprecation_date")
                if available_from and today < available_from:
                    report.errors.append(
                        f"Model '{self.model_name}' is not available before {available_from.isoformat()}."
                    )
                if deprecation_date and today > deprecation_date:
                    report.errors.append(
                        f"Model '{self.model_name}' is past its deprecation date "
                        f"({deprecation_date.isoformat()})."
                    )
                elif deprecation_date:
                    report.warnings.append(
                        f"Model '{self.model_name}' is scheduled to deprecate on "
                        f"{deprecation_date.isoformat()}."
                    )

        if not self.user_input.strip():
            report.errors.append("user_input must not be empty.")
        if self.execution_mode not in ("generate_content", "chat"):
            report.errors.append("execution_mode must be 'generate_content' or 'chat'.")

        for index, turn in enumerate(self.history):
            turn.validate(report, field_name=f"history[{index}]")

        for attachment in self.attachments:
            attachment.validate(report)

        self.structured_output.validate(report)
        self.generation.validate(report)

        if self.structured_output.enabled and self.generation.candidate_count not in (None, 1):
            report.errors.append("Structured output should be used with candidate_count=1.")

        if self.enable_google_search and self.structured_output.enabled:
            report.warnings.append(
                "Structured output with Google Search grounding can be brittle if the model returns citations."
            )

        return report

    def build_generate_config(self) -> types.GenerateContentConfig:
        """Build the SDK config object from this request definition."""
        config_kwargs: dict[str, Any] = {
            "temperature": self.generation.temperature,
            "top_p": self.generation.top_p,
            "top_k": self.generation.top_k,
            "candidate_count": self.generation.candidate_count,
            "max_output_tokens": self.generation.max_output_tokens,
            "system_instruction": self.system_instruction,
            "labels": self.labels or None,
        }

        if self.enable_google_search:
            config_kwargs["tools"] = [types.Tool(google_search=types.GoogleSearch())]

        if self.structured_output.enabled:
            config_kwargs["response_mime_type"] = self.structured_output.mime_type
            if self.structured_output.json_schema is not None:
                config_kwargs["response_json_schema"] = self.structured_output.json_schema

        return types.GenerateContentConfig(**{k: v for k, v in config_kwargs.items() if v is not None})


class ExecutionResult(BaseModel):
    """Normalized output returned after one request is executed.

    Attributes:
        model_name: Model used to generate the response.
        text: Best-effort plain-text response extracted from the SDK object.
        parsed: Parsed structured output, when available.
        raw_response: Original SDK response object.
        response_id: Response identifier returned by the API, if present.
        usage_metadata: Usage metadata returned by the API, if present.
        updated_history: Conversation history including this request/response.
        uploaded_files: API file handles created for request attachments.
        validation_warnings: Non-fatal warnings produced during validation.
    """

    model_config = ConfigDict(extra="forbid", strict=True, arbitrary_types_allowed=True)
    model_name: str
    text: str
    parsed: Any
    raw_response: types.GenerateContentResponse
    response_id: str | None
    usage_metadata: Any
    updated_history: list[HistoryTurn]
    uploaded_files: list[types.File]
    validation_warnings: list[str]


class GeminiRequestExecutor:
    """Builds, validates, and executes Gemini API requests."""

    def __init__(self, api_key: str | None = None) -> None:
        """Create an executor using ``api_key`` or ``GEMINI_API_KEY``."""
        resolved_api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not resolved_api_key:
            raise GeminiConfigurationError("GEMINI_API_KEY is not set.")
        self.client = genai.Client(api_key=resolved_api_key)
        self._attachment_cache: dict[tuple[str, str, str | None], types.File] = {}

    def execute(self, request: GeminiRequestConfig) -> ExecutionResult:
        """Execute one request and return a normalized result object.

        Args:
            request: Request configuration to validate and execute.

        Returns:
            An ``ExecutionResult`` with response text, metadata, and history.
        """
        report = request.validate()
        report.raise_if_invalid()

        uploaded_files = self._upload_attachments(request.attachments)
        message_parts = [types.Part.from_text(text=request.user_input)]
        message_parts.extend(types.Part.from_uri(file_uri=file.uri, mime_type=file.mime_type) for file in uploaded_files)
        message_payload = message_parts if len(message_parts) > 1 else request.user_input
        config = request.build_generate_config()

        if request.execution_mode == "chat":
            chat = self.client.chats.create(
                model=request.model_name,
                config=config,
                history=[turn.to_content() for turn in request.history] or None,
            )
            response = chat.send_message(message=message_payload, config=config)
        else:
            contents: list[types.Content] = [turn.to_content() for turn in request.history]
            contents.append(types.UserContent(parts=message_parts))
            response = self.client.models.generate_content(
                model=request.model_name,
                contents=contents,
                config=config,
            )

        response_text = getattr(response, "text", "") or self._extract_text(response)
        updated_history = self._build_updated_history(request.history, request.user_input, response_text)

        return ExecutionResult(
            model_name=request.model_name,
            text=response_text,
            parsed=getattr(response, "parsed", None),
            raw_response=response,
            response_id=getattr(response, "response_id", None),
            usage_metadata=getattr(response, "usage_metadata", None),
            updated_history=updated_history,
            uploaded_files=uploaded_files,
            validation_warnings=report.warnings,
        )

    def _upload_attachments(self, attachments: Sequence[AttachmentConfig]) -> list[types.File]:
        """Upload all attachments and return the resulting API file objects."""
        uploaded_files: list[types.File] = []
        for attachment in attachments:
            resolved_path = attachment.resolved_path()
            mime_type = attachment.resolved_mime_type()
            cache_key = (str(resolved_path), mime_type, attachment.display_name)
            cached_file = self._attachment_cache.get(cache_key)
            if cached_file is None:
                cached_file = self.client.files.upload(
                    file=str(resolved_path),
                    config=types.UploadFileConfig(
                        mime_type=mime_type,
                        display_name=attachment.display_name,
                    ),
                )
                self._attachment_cache[cache_key] = cached_file
            uploaded_files.append(cached_file)
        return uploaded_files

    @staticmethod
    def _extract_text(response: types.GenerateContentResponse) -> str:
        """Extract text parts from all response candidates as a fallback."""
        texts: list[str] = []
        for candidate in response.candidates or []:
            content = getattr(candidate, "content", None)
            if content is None:
                continue
            for part in content.parts or []:
                text = getattr(part, "text", None)
                if text:
                    texts.append(text)
        return "\n".join(texts).strip()

    @staticmethod
    def _build_updated_history(history: Sequence[HistoryTurn], user_input: str, model_output: str) -> list[HistoryTurn]:
        """Return a new history list extended with the latest exchange."""
        updated = list(history)
        updated.append(HistoryTurn(role="user", text=user_input))
        if model_output.strip():
            updated.append(HistoryTurn(role="model", text=model_output))
        return updated


class FlowStep(BaseModel):
    """Defines one step in a multi-request Gemini workflow.

    Attributes:
        step_id: Stable identifier for the step.
        request: Base request to execute for this step.
        history_mode: Whether to inherit the orchestrator history or replace it.
        inject_previous_response_template: Optional template used to fold the
            previous model response into the next prompt.
    """

    model_config = ConfigDict(extra="forbid", strict=True)
    step_id: str
    request: GeminiRequestConfig
    history_mode: Literal["inherit", "replace"] = "inherit"
    inject_previous_response_template: str | None = None


class FlowExecutionResult(BaseModel):
    """Aggregates per-step results together with the final conversation history."""

    model_config = ConfigDict(extra="forbid", strict=True, arbitrary_types_allowed=True)
    step_results: list[ExecutionResult]
    final_history: list[HistoryTurn]


class GeminiFlowOrchestrator:
    """Executes multiple request configs while allowing explicit history edits."""

    def __init__(self, executor: GeminiRequestExecutor) -> None:
        """Initialize the orchestrator with an executor and empty history."""
        self.executor = executor
        self._history: list[HistoryTurn] = []

    @property
    def history(self) -> list[HistoryTurn]:
        """Return a copy of the current internal history."""
        return list(self._history)

    def set_history(self, history: Sequence[HistoryTurn]) -> None:
        """Replace the full history after validating each provided turn."""
        validation = ValidationReport()
        for index, turn in enumerate(history):
            turn.validate(validation, field_name=f"history[{index}]")
        validation.raise_if_invalid()
        self._history = list(history)

    def append_history(self, role: Role, text: str) -> None:
        """Append one validated turn to the existing history."""
        turn = HistoryTurn(role=role, text=text)
        validation = ValidationReport()
        turn.validate(validation, field_name="history_append")
        validation.raise_if_invalid()
        self._history.append(turn)

    def replace_history_turn(self, index: int, role: Role | None = None, text: str | None = None) -> None:
        """Replace one history item while preserving unspecified fields."""
        original = self._history[index]
        updated = HistoryTurn(role=role or original.role, text=text or original.text)
        validation = ValidationReport()
        updated.validate(validation, field_name=f"history[{index}]")
        validation.raise_if_invalid()
        self._history[index] = updated

    def delete_history_turn(self, index: int) -> None:
        """Delete the history item at ``index``."""
        del self._history[index]

    def execute_step(self, step: FlowStep) -> ExecutionResult:
        """Execute one flow step and persist the returned history."""
        request = step.request
        base_history = self._history if step.history_mode == "inherit" else []
        request_with_history = request.model_copy(
            update={"history": list(base_history) + list(request.history)}
        )

        if step.inject_previous_response_template and self._history:
            latest_model_turn = next((turn for turn in reversed(self._history) if turn.role == "model"), None)
            if latest_model_turn is not None:
                request_with_history = request_with_history.model_copy(
                    update={
                        "user_input": step.inject_previous_response_template.format(
                        previous_response=latest_model_turn.text,
                        user_input=request_with_history.user_input,
                        )
                    },
                )

        result = self.executor.execute(request_with_history)
        self._history = result.updated_history
        return result

    def run(self, steps: Sequence[FlowStep]) -> FlowExecutionResult:
        """Execute all steps sequentially and return the aggregated result."""
        results: list[ExecutionResult] = []
        for step in steps:
            results.append(self.execute_step(step))
        return FlowExecutionResult(step_results=results, final_history=self.history)


def build_three_model_demo_flow(
    *,
    pdf_path: Path | None = None,
    image_path: Path | None = None,
    audio_path: Path | None = None,
) -> list[FlowStep]:
    """Build a sample three-step flow that chains collection, structuring, and delivery.

    Args:
        pdf_path: Optional PDF attachment for the collection step.
        image_path: Optional image attachment for the collection step.
        audio_path: Optional audio attachment for the collection step.

    Returns:
        A list of ``FlowStep`` instances that can be executed by the orchestrator.
    """
    attachments = [
        AttachmentConfig(file_path=path)
        for path in (pdf_path, image_path, audio_path)
        if path is not None
    ]

    structured_schema = {
        "type": "object",
        "properties": {
            "summary": {"type": "string"},
            "risks": {"type": "array", "items": {"type": "string"}},
            "next_actions": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["summary", "risks", "next_actions"],
    }

    return [
        FlowStep(
            step_id="collect",
            request=GeminiRequestConfig(
                model_name="gemini-3.1-pro-preview",
                system_instruction=(
                    "You are the planning model. Inspect all provided materials and extract decision-ready facts."
                ),
                user_input=(
                    "Read the uploaded materials and produce a detailed factual briefing. "
                    "Use Google Search only when external verification materially improves the answer."
                ),
                attachments=attachments,
                enable_google_search=True,
                execution_mode="generate_content",
                generation=GenerationControls(temperature=0.2, max_output_tokens=2048),
            ),
        ),
        FlowStep(
            step_id="structure",
            request=GeminiRequestConfig(
                model_name="gemini-3-pro-preview",
                system_instruction=(
                    "You are the synthesis model. Convert the working notes into strict structured output."
                ),
                user_input=(
                    "Transform the prior discussion into the requested JSON object. "
                    "Do not add fields outside the schema."
                ),
                structured_output=StructuredOutputConfig(enabled=True, json_schema=structured_schema),
                execution_mode="chat",
                generation=GenerationControls(temperature=0.0, max_output_tokens=1024),
            ),
            inject_previous_response_template=(
                "Previous response:\n{previous_response}\n\nCurrent instruction:\n{user_input}"
            ),
        ),
        FlowStep(
            step_id="deliver",
            request=GeminiRequestConfig(
                model_name="gemini-3-flash-preview",
                system_instruction=(
                    "You are the delivery model. Produce a concise end-user answer grounded in the prior steps."
                ),
                user_input=(
                    "Turn the prior structured result into a concise action plan for the user. "
                    "Preserve the important risks."
                ),
                execution_mode="chat",
                generation=GenerationControls(temperature=0.3, max_output_tokens=768),
            ),
            inject_previous_response_template=(
                "Use this structured result as the main source:\n{previous_response}\n\nTask:\n{user_input}"
            ),
        ),
    ]


def run_demo_flow(
    *,
    api_key: str | None = None,
    pdf_path: Path | None = None,
    image_path: Path | None = None,
    audio_path: Path | None = None,
) -> FlowExecutionResult:
    """Execute the sample flow end-to-end with optional attachments."""
    executor = GeminiRequestExecutor(api_key=api_key)
    orchestrator = GeminiFlowOrchestrator(executor)
    steps = build_three_model_demo_flow(pdf_path=pdf_path, image_path=image_path, audio_path=audio_path)
    return orchestrator.run(steps)


def print_flow_result(result: FlowExecutionResult) -> None:
    """Print step outputs, warnings, and parsed JSON to standard output."""
    for index, step_result in enumerate(result.step_results, start=1):
        print(f"[Step {index}] model={step_result.model_name}")
        if step_result.validation_warnings:
            print("Validation warnings:")
            for warning in step_result.validation_warnings:
                print(f"- {warning}")
        print(step_result.text)
        if step_result.parsed is not None:
            print(json.dumps(step_result.parsed, ensure_ascii=False, indent=2))
        print("-" * 80)


if __name__ == "__main__":
    demo_pdf = os.environ.get("GEMINI_DEMO_PDF")
    demo_image = os.environ.get("GEMINI_DEMO_IMAGE")
    demo_audio = os.environ.get("GEMINI_DEMO_AUDIO")
    demo_result = run_demo_flow(
        pdf_path=Path(demo_pdf) if demo_pdf else None,
        image_path=Path(demo_image) if demo_image else None,
        audio_path=Path(demo_audio) if demo_audio else None,
    )
    print_flow_result(demo_result)
