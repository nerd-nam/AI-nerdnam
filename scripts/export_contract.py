"""계약 C-04 내보내기: app/schemas.py → /openapi.json → SPEC contracts/api-ai.openapi.yaml
사용: uv run python -m scripts.export_contract ../SPEC-nerdnam/contracts/api-ai.openapi.yaml"""
import sys

import yaml

from app.main import app

with open(sys.argv[1], "w", encoding="utf-8") as f:
    yaml.safe_dump(app.openapi(), f, allow_unicode=True, sort_keys=False, width=10_000)
