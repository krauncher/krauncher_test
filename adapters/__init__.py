"""Service adapters, by name (run.py --service <name>)."""

from adapters.krauncher import KrauncherAdapter

ADAPTERS = {a.name: a for a in (KrauncherAdapter(),)}
