"""Check installed distribution requirements reachable from explicit roots."""


def check_dependency_closure(roots, *, lookup=None):
    """Check every active transitive requirement, including requested extras.

    Unrelated installed distributions are outside the scope. Import/API behavior
    is a separate check. ``lookup`` is for model-free metadata fixtures only.
    """
    import importlib.metadata as metadata
    from packaging.requirements import InvalidRequirement, Requirement
    from packaging.utils import canonicalize_name

    lookup = metadata.distribution if lookup is None else lookup
    queue = [(canonicalize_name(name), frozenset()) for name in roots]
    seen = set()
    checked = {}
    errors = []
    while queue:
        name, extras = queue.pop()
        if (name, extras) in seen:
            continue
        seen.add((name, extras))
        try:
            dist = lookup(name)
        except (metadata.PackageNotFoundError, KeyError):
            errors.append({"package": name, "reason": "distribution missing"})
            continue
        checked[name] = dist.version
        for text in dist.requires or ():
            try:
                req = Requirement(text)
                active = req.marker is None or any(req.marker.evaluate({"extra": extra}) for extra in ("", *extras))
            except (InvalidRequirement, ValueError) as exc:
                errors.append({"package": name, "requirement": text, "reason": type(exc).__name__})
                continue
            if not active:
                continue
            child = canonicalize_name(req.name)
            try:
                child_dist = lookup(child)
            except (metadata.PackageNotFoundError, KeyError):
                errors.append({"package": name, "requirement": text, "reason": "dependency missing"})
                continue
            if req.url:
                # A version comparison cannot establish a URL/commit identity.
                errors.append({"package": name, "requirement": text,
                               "reason": "direct URL requirement needs separate source verification"})
            elif not req.specifier.contains(child_dist.version, prereleases=True):
                errors.append({"package": name, "requirement": text, "installed": child_dist.version,
                               "reason": "version mismatch"})
            queue.append((child, frozenset(req.extras)))
    return {
        "artifact_kind": "setup-smoke", "split": "example", "passed": not errors,
        "scope": "transitive_declared_dependencies", "roots": list(roots),
        "checked_versions": dict(sorted(checked.items())), "errors": errors,
        "limitations": ["Unrelated packages are not certified", "Metadata compatibility is not API or model verification"],
    }
