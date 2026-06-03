from __future__ import annotations

import os


def main() -> None:
    """Run the claim-extraction GUI with uvicorn.

    Configure with FACTICLI_WEB_HOST / FACTICLI_WEB_PORT (defaults 127.0.0.1:8000).
    """
    import uvicorn

    host = os.getenv("FACTICLI_WEB_HOST", "127.0.0.1")
    port = int(os.getenv("FACTICLI_WEB_PORT", "8000"))
    uvicorn.run("facticli.web.app:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    main()
