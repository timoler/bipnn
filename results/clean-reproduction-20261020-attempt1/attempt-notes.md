# First clean installation attempt

This attempt created `.venv-clean-20261020` with `include-system-site-packages = false`.
It exposed an incomplete original support package list. The tool's native output
reported `Ran 41 tests` and `FAILED (errors=4)`: importing `test_anytime`,
`test_completion`, `test_controlled` and `test_fast_decode` failed with
`ModuleNotFoundError: No module named 'torch_geometric'` through
`src/__init__.py -> src/core.py`. Inspection of the same upstream initialization
also identified the dimod, dwave-neal, dwave-samplers and pyqubo imports.

The original environment evidence and package freeze are retained. `prepare.txt`
is an incomplete PowerShell transcript: this host did not capture native process
stdout/stderr. It must not be described as a complete installation or failure log.
No successful completion marker or GPU smoke was produced for this attempt.
The native test outcome above is a note transcribed from the actual tool output,
not a replacement raw log. Identity headers were removed from the public transcript;
an unredacted local copy is retained outside the repository as noted separately.
