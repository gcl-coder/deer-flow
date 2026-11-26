"""Shared source-selection rules for the bundled web-search providers.

``include_domains`` and ``exclude_domains`` started in Serper (#6213) and were
extracted here so a provider without a native filter can enforce the same
rules on its own rows. The functions are pure: they read a configured value and
a result URL, and never resolve, fetch or connect.

Domain validation follows Python's IDNA codec so an international name and its
A-label form select the same rows, and a host that is really an IP literal is
refused: the lists describe *sites*, not addresses.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from ipaddress import IPv4Address, ip_address
from urllib.parse import urlparse

# A configured list longer than this is a mistake, not a preference.
MAX_DOMAINS = 10

_DOMAIN_LABEL = re.compile(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?")

# Two-label suffixes whose registrable domain needs three labels. Grouping the
# per-domain cap by the last two labels alone would let one ``example.co.uk``
# crowd out an unrelated ``other.co.uk``.
_MULTI_LABEL_SUFFIXES = frozenset(
    {
        "ac.uk",
        "co.in",
        "co.jp",
        "co.kr",
        "co.nz",
        "co.uk",
        "co.za",
        "com.au",
        "com.br",
        "com.cn",
        "com.mx",
        "com.sg",
        "com.tr",
        "com.tw",
        "edu.cn",
        "gov.cn",
        "net.cn",
        "org.uk",
    }
)


def normalize_domain(value: object) -> str:
    """Accept a bare DNS domain, in Unicode or A-label form.

    Rejects a scheme, port, path, wildcard, user info or an IP literal, because
    each of those means the operator wrote something other than a hostname and
    silently interpreting it would select the wrong rows.
    """
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError("expected a domain string")
    domain = value.removesuffix(".").lower()
    try:
        domain = domain.encode("idna").decode("ascii")
        # Validate existing A-labels too (the encoder otherwise passes them through).
        domain.encode("ascii").decode("idna")
    except UnicodeError as exc:
        raise ValueError("invalid IDNA domain") from exc
    labels = domain.split(".")
    if len(domain) > 253 or len(labels) < 2 or any(not _DOMAIN_LABEL.fullmatch(label) for label in labels):
        raise ValueError("expected a DNS domain without scheme, port, path or operators")
    try:
        ip_address(domain)
    except ValueError:
        if _decode_ipv4(domain) is None:
            return domain
    raise ValueError("IP literals are not domains")


def domain_list(extra: Mapping[str, object], name: str) -> list[str]:
    """Read one configured domain list, normalized and de-duplicated in order."""
    values = extra.get(name, [])
    if not isinstance(values, list) or len(values) > MAX_DOMAINS:
        raise ValueError(f"{name} must be a list of at most {MAX_DOMAINS} domains")
    try:
        return list(dict.fromkeys(normalize_domain(value) for value in values))
    except ValueError as exc:
        raise ValueError(f"Invalid {name}: {exc}") from exc


def domain_scope(extra: Mapping[str, object]) -> tuple[list[str], list[str]]:
    """Read both configured lists together; a bad value raises before any work."""
    return domain_list(extra, "include_domains"), domain_list(extra, "exclude_domains")


def matches_domain_scope(value: object, include: list[str], exclude: list[str]) -> bool:
    """Decide whether one result URL belongs to the configured source scope.

    This is source selection, not DNS resolution or a fetch permission check, so
    it never raises: a URL that cannot be read as a plain HTTP(S) address with a
    DNS host is simply outside every list.
    """
    if not isinstance(value, str) or any(ord(char) <= 32 or ord(char) == 127 for char in value) or "\\" in value:
        return False
    try:
        parsed = urlparse(value)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc or parsed.username is not None or parsed.password is not None:
            return False
        _ = parsed.port  # Reject malformed/out-of-range ports.
        host = normalize_domain(parsed.hostname)
    except ValueError:
        return False

    def matches(domains: list[str]) -> bool:
        return any(host == domain or host.endswith("." + domain) for domain in domains)

    return not matches(exclude) and (not include or matches(include))


def coerce_domain_cap(value: object) -> int:
    """Read ``dedupe_by_domain`` as a non-negative int, 0 meaning off.

    A silently ignored malformed value would look like the cap working, so this
    raises like the domain lists do instead of falling back to a default.
    """
    if value is None:
        return 0
    if isinstance(value, bool):
        raise ValueError("dedupe_by_domain must be an integer")
    if isinstance(value, float):
        if not value.is_integer():
            raise ValueError("dedupe_by_domain must be an integer")
        value = int(value)
    if isinstance(value, str):
        text = value.strip()
        if not text.isdigit():
            raise ValueError("dedupe_by_domain must be an integer")
        value = int(text)
    if not isinstance(value, int):
        raise ValueError("dedupe_by_domain must be an integer")
    if value < 0:
        raise ValueError("dedupe_by_domain must not be negative")
    return value


def cap_results_per_domain(results: list[dict], limit: int) -> tuple[list[dict], int]:
    """Keep at most ``limit`` results per registrable domain, in provider order.

    Returns the kept rows and how many were dropped. ``limit`` of 0 or less
    disables the cap. The registrable domain is the host minus its public
    suffix, approximated by the last two labels plus a small table of common
    two-label suffixes; it is a grouping key for this cap, not a PSL lookup.
    """
    if limit <= 0:
        return results, 0
    kept: list[dict] = []
    counts: dict[str, int] = {}
    for row in results:
        key = _registrable_domain(row.get("url"))
        if key is not None:
            counts[key] = counts.get(key, 0) + 1
            if counts[key] > limit:
                continue
        kept.append(row)
    return kept, len(results) - len(kept)


def _registrable_domain(value: object) -> str | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        host = (urlparse(value).hostname or "").rstrip(".").lower()
    except ValueError:
        return None
    if not host:
        return None
    labels = host.split(".")
    if len(labels) <= 2:
        return host
    if ".".join(labels[-2:]) in _MULTI_LABEL_SUFFIXES:
        return ".".join(labels[-3:])
    return ".".join(labels[-2:])


def _decode_ipv4(host: str) -> IPv4Address | None:
    """Decode obfuscated IPv4 literals that ``ip_address`` rejects.

    Mirrors the permissive ``inet_aton`` parsing many HTTP clients use, so that
    integer (``2130706433``), hex (``0x7f000001``) and octal (``0177.0.0.1``)
    encodings of an address are recognized. Returns an ``IPv4Address`` when the
    host decodes to one, otherwise ``None`` (e.g. real domains like
    ``cafe.com`` fail to decode and are left for the caller to treat as a host).
    """
    parts = host.split(".")
    if not 1 <= len(parts) <= 4:
        return None

    values: list[int] = []
    for part in parts:
        if not part:
            return None
        try:
            if part.startswith(("0x", "0X")):
                values.append(int(part, 16))
            elif part.startswith("0") and len(part) > 1:
                values.append(int(part, 8))
            else:
                values.append(int(part, 10))
        except ValueError:
            return None

    *leading, last = values
    for value in leading:
        if not 0 <= value <= 0xFF:
            return None
    max_last = (1 << (8 * (4 - len(leading)))) - 1
    if not 0 <= last <= max_last:
        return None

    result = 0
    for value in leading:
        result = (result << 8) | value
    result = (result << (8 * (4 - len(leading)))) | last
    return ip_address(result)


def result_envelope(query: str, provider: str, rows: list[dict], dropped: int = 0) -> dict:
    """Assemble the payload every bundled ``web_search`` provider returns.

    Callers, the trace view and citation handling each used to special-case the
    provider, because Tavily answered with a bare list and the text field was
    named ``content`` here and ``snippet`` there. One envelope with one name
    fixes that; ``content`` stays as an alias for a release so stored prompts
    and skills that read it keep working.
    """
    results = []
    for row in rows:
        text = row.get("snippet")
        if not text:
            text = row.get("content") or ""
        results.append(
            {
                "title": row.get("title") or "",
                "url": row.get("url") or "",
                "snippet": text,
                "content": text,
            }
        )
    return {
        "query": query,
        "provider": provider,
        "total_results": len(results),
        "results": results,
        "dropped_results": dropped,
    }
