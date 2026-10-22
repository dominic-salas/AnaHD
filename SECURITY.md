# Security

## Supported versions

Only the latest release gets fixes. AnaHD works offline: it never uploads footage or
sends data anywhere. From source, the only network request is a one-time model download
from this repository's releases, checked against a pinned SHA-256.

## Reporting a vulnerability

Please don't open a public issue. Report it privately through the repository's
**Security** tab, **Report a vulnerability**. That form reaches only the maintainer.

Include what an attacker could do, the AnaHD version, and steps or a file that shows it.
Examples that count:
- a recording or settings file that makes AnaHD run code, or write or delete files
  outside its output;
- the installer or app loading code from a place a normal user can write to;
- a model or download that is used without its checksum being checked.

You'll get an answer within two weeks. A confirmed problem is fixed in a new release,
and the report is credited there unless you'd rather not be named.

## Out of scope

- Problems in FFmpeg, PyTorch, Qt or NVIDIA's libraries themselves: report those to
  their projects (AnaHD will update the bundled version once a fix is out).
- Running AnaHD on a recording you trust less than the PC it runs on is like opening it
  in any video player: FFmpeg reads it.
