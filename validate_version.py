import semver
import sys

input = sys.argv[1]
ver = semver.Version.parse(input)
if ver.prerelease:
    prefix = ver.prerelease.split('.')[0]
    if not prefix in ('alpha', 'beta', 'rc'):
        raise ValueError(f"Invalid pre-release. (alpha|beta|rc) accepted.")
