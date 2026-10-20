# Second clean installation attempt

This attempt created a new `.venv-clean-20261020-r2` after adding the fixed
upstream import dependencies. The Prepare and Smoke phases both exited
successfully. The actual tool output reported 61 tests passing without skips,
and weak/strong SCIP toy optima 9/6. The saved GPU reports contain three
200-epoch neural runs per definition, with sizes 9/9/9 and 6/6/6; all six raw
threshold solutions were feasible. Saved solution reports and independent
verification output were produced.

During evidence assembly, it was discovered that this PowerShell host's
`Start-Transcript` did not capture native uv/Python stdout/stderr.
`prepare.txt` and `smoke.txt` are retained as incomplete transcripts, not full
installation/test logs. The JSON reports, isolation checks, package freeze and
completion markers are genuine output from this attempt. The final reproduction
uses a third new environment and explicitly saves each native command's merged
stdout/stderr with `Tee-Object`, so the final installation and tests have complete
on-disk native output rather than reconstructed logs.
