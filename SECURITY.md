# Security policy

ReadCue is an early research project. Security fixes target the current `main`
branch; there are no maintained release branches or published ReadCue weights.

Please report vulnerabilities privately using
[GitHub's vulnerability reporting form](https://github.com/andyandymike/ReadCue/security/advisories/new).
Include the affected commit, environment, impact, and a minimal reproduction
using synthetic inputs. Do not include credentials, private data, or model
weights. If the form is unavailable, open an issue asking for a private contact
without disclosing the vulnerability.

Ordinary bugs and model-quality questions belong in
[Issues](https://github.com/andyandymike/ReadCue/issues).
This project has no guaranteed response time or production support commitment.

Run commands only against a trusted checkout: verification establishes recorded
file consistency, not that an arbitrary repository is safe to execute. Baseline
model terms and optional inference dependencies remain separate from the core
package. Historic frozen experiments retain their original records; security
fixes must not rewrite old evidence or encourage running a known unsafe version.
