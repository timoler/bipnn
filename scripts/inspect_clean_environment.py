"""Record and enforce isolation for a newly installed Windows project environment."""
import argparse
import hashlib
import importlib
import importlib.metadata
import json
import platform
import site
import sys
from datetime import datetime, timezone
from pathlib import Path


EXPECTED = {
    "torch": "2.8.0+cu128", "torchvision": "0.23.0+cu128", "dhg": "0.9.5",
    "numpy": "1.26.4", "scipy": "1.13.1", "matplotlib": "3.9.4",
    "optuna": "4.9.0", "requests": "2.32.5", "scikit-learn": "1.6.1",
    "pyscipopt": "5.3.0",
    "torch-geometric": "2.6.1", "dimod": "0.12.18", "dwave-neal": "0.6.0",
    "dwave-samplers": "1.6.0", "pyqubo": "1.5.0",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prefix = Path(sys.prefix).resolve()
    root = Path(__file__).resolve().parents[1]
    assert platform.python_version() == "3.9.23", sys.version
    assert sys.prefix != sys.base_prefix, "This is not a virtual environment"
    assert not site.ENABLE_USER_SITE, "User site packages are enabled"
    configuration = (prefix / "pyvenv.cfg").read_text(encoding="utf-8")
    assert "include-system-site-packages = false" in configuration.lower()
    versions, origins = {}, {}
    for distribution, expected in EXPECTED.items():
        versions[distribution] = importlib.metadata.version(distribution)
        assert versions[distribution] == expected, (distribution, versions[distribution], expected)
        module_name = {"scikit-learn": "sklearn", "torch-geometric": "torch_geometric",
                       "dwave-neal": "neal", "dwave-samplers": "dwave.samplers"}.get(distribution, distribution)
        module = importlib.import_module(module_name)
        origin = Path(module.__file__).resolve()
        assert prefix in origin.parents, (distribution, str(origin))
        origins[distribution] = str(origin.relative_to(prefix))
    for entry in sys.path:
        path = Path(entry).resolve()
        if "site-packages" in path.parts:
            assert prefix in path.parents, ("External site packages", entry)
    torch = importlib.import_module("torch")
    assert torch.version.cuda == "12.8", torch.version.cuda
    source_names = ["scripts/setup_mis.ps1", "scripts/reproduce_clean.ps1",
                    "scripts/inspect_clean_environment.py", "scripts/clean_scip_smoke.py",
                    "requirements-mis-support.txt"]
    evidence = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(), "executable": sys.executable,
        "sys_prefix": sys.prefix, "sys_base_prefix": sys.base_prefix,
        "include_system_site_packages": False, "enable_user_site": site.ENABLE_USER_SITE,
        "python_isolated_flag": sys.flags.isolated, "sys_path": sys.path,
        "pyvenv_configuration": configuration, "versions": versions,
        "module_origins_relative_to_environment": origins, "torch_cuda": torch.version.cuda,
        "source_sha256": {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                          for name in source_names},
        "test_source_sha256": {str(path.relative_to(root)).replace("\\", "/"):
                               hashlib.sha256(path.read_bytes()).hexdigest()
                               for path in sorted((root / "tests").glob("test_*.py"))},
        "checks_passed": True,
        "limitation": "DHG torch<2 metadata is explicitly overridden; only project operations are tested.",
    }
    args.output.write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"checks_passed": True, "python": evidence["python"],
                      "versions": versions, "output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
