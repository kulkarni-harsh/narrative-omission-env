"""FastAPI application for the Narrative Omission Detection Environment."""

import os

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:  # pragma: no cover
    raise ImportError(
        "openenv is required. Install dependencies with 'uv sync'"
    ) from e

try:
    from ..models import NarrativeAction, NarrativeObservation
    from .narrative_omission_env_environment import NarrativeEnvironment
    from .gradio_app import build_narrative_gradio_app
except ImportError:
    from models import NarrativeAction, NarrativeObservation
    from server.narrative_omission_env_environment import NarrativeEnvironment
    from server.gradio_app import build_narrative_gradio_app


def env_factory() -> NarrativeEnvironment:
    task = os.getenv("TASK_NAME", "easy")
    return NarrativeEnvironment(task_name=task)


app = create_app(
    env_factory,
    NarrativeAction,
    NarrativeObservation,
    env_name="narrative_omission",
    max_concurrent_envs=4,
    gradio_builder=build_narrative_gradio_app,
)


def main(host: str = "0.0.0.0", port: int = int(os.getenv("PORT", "8000"))) -> None:
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
