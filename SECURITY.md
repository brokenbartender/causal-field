# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| 1.x     | Yes       |

## Reporting a Vulnerability

Do not open a public issue for security vulnerabilities.

Report privately via GitHub Security Advisories:
[Report a vulnerability](https://github.com/brokenbartender/causal-field/security/advisories/new)

Or email: codymckenzie23@gmail.com

Response within 48 hours, patch within 7 days for confirmed issues.

## Scope

causal-field operates entirely in-process. It does not make network requests,
execute external code, or persist state beyond the in-memory field object.
The optional Ollama integration communicates only with a local Ollama instance.
