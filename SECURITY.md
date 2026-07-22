# Security policy

## Reporting

Use GitHub's private security-advisory feature for credential exposure, unsafe deserialization, path traversal, dependency compromise, or unintended disclosure of account/data artifacts. Do not open a public issue containing exploit details or secrets.

## Credential model

TensorAlpha reads provider credentials only from environment variables. The repository contains no credential store and no broker connectivity. `.env`, model, data, output, and paper-account paths are ignored.

## Release checks

`python scripts/check_release.py .` scans for known credential formats, non-placeholder secret assignments, email addresses (including commit metadata), personal absolute paths, forbidden runtime directories, model-key extensions, and files larger than 5 MiB. Public commits must use a GitHub `noreply` address.

If a real credential is ever committed, revoke or rotate it first. Deleting the current file does not remove a credential from Git history.
