"""Shared source-selection rules used by the bundled web-search providers.

``include_domains``/``exclude_domains`` were Serper-only (#6213). These cases
pin the extracted rules at the layer the providers now call, so a provider
without a native filter selects the same rows as one that has it.
"""

import pytest

from deerflow.community.search_filters import (
    cap_results_per_domain,
    coerce_domain_cap,
    domain_list,
    domain_scope,
    matches_domain_scope,
    normalize_domain,
)


def test_normalize_domain_accepts_a_bare_host_and_its_idna_form() -> None:
    assert normalize_domain("Example.COM.") == "example.com"
    assert normalize_domain("bücher.de") == "xn--bcher-kva.de"
    assert normalize_domain("xn--bcher-kva.de") == "xn--bcher-kva.de"


@pytest.mark.parametrize(
    "value",
    [
        None,
        1,
        "",
        " example.com",
        "example.com ",
        "https://example.com",
        "*.example.com",
        "example.com/path",
        "example.com:443",
        "a@b.com",
        "a..com",
        "-a.com",
        "a_.com",
        "localhost",
        "127.0.0.1",
        "2130706433",
        "0x7f000001",
        "0177.0.0.1",
        "example.com OR site:evil.com",
        "a" * 64 + ".com",
    ],
)
def test_normalize_domain_rejects_anything_that_is_not_a_hostname(value) -> None:
    with pytest.raises(ValueError):
        normalize_domain(value)


def test_domain_list_deduplicates_in_configured_order() -> None:
    extra = {"include_domains": ["Example.COM", "docs.python.org", "example.com"]}
    assert domain_list(extra, "include_domains") == ["example.com", "docs.python.org"]


@pytest.mark.parametrize("value", [None, "example.com", {}, True, [None], [1], [""], ["a.com"] * 11])
def test_domain_list_rejects_a_value_that_is_not_a_bounded_list(value) -> None:
    with pytest.raises(ValueError):
        domain_list({"include_domains": value}, "include_domains")


def test_domain_scope_reports_which_list_holds_the_bad_value() -> None:
    with pytest.raises(ValueError, match="exclude_domains"):
        domain_scope({"include_domains": ["example.com"], "exclude_domains": ["https://bad.example"]})


def test_unset_lists_select_every_row() -> None:
    include, exclude = domain_scope({})
    assert (include, exclude) == ([], [])
    assert matches_domain_scope("https://anything.example/a", include, exclude)


def test_scope_matches_the_host_and_its_subdomains_but_not_a_lookalike() -> None:
    include, exclude = domain_scope({"include_domains": ["example.com"], "exclude_domains": ["bad.example.com"]})
    assert matches_domain_scope("https://EXAMPLE.com./a", include, exclude)
    assert matches_domain_scope("https://docs.example.com/a", include, exclude)
    assert not matches_domain_scope("https://bad.example.com/a", include, exclude)
    assert not matches_domain_scope("https://deep.bad.example.com/a", include, exclude)
    assert not matches_domain_scope("https://notexample.com/a", include, exclude)
    assert not matches_domain_scope("https://example.com.evil.com/a", include, exclude)


def test_scope_treats_a_unicode_name_and_its_a_label_as_one_site() -> None:
    include, _ = domain_scope({"include_domains": ["bücher.de"]})
    assert matches_domain_scope("https://bücher.de/a", include, [])
    assert matches_domain_scope("https://xn--bcher-kva.de/b", include, [])


@pytest.mark.parametrize(
    "value",
    [
        None,
        "",
        "not a URL",
        "ftp://example.com/a",
        "https://user:pass@example.com/a",
        "https://example.com:70000/a",
        "https://example.com\\a",
        "https://127.0.0.1/a",
    ],
)
def test_scope_declines_a_row_it_cannot_read_as_a_public_url(value) -> None:
    include, _ = domain_scope({"include_domains": ["example.com"]})
    assert not matches_domain_scope(value, include, [])


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, 0),
        (0, 0),
        (1, 1),
        (3, 3),
        ("2", 2),
        (2.0, 2),
    ],
)
def test_coerce_domain_cap_accepts_an_absent_or_non_negative_count(value, expected) -> None:
    assert coerce_domain_cap(value) == expected


@pytest.mark.parametrize("value", [True, False, "two", "1.5", 1.5, -1, {}, []])
def test_coerce_domain_cap_rejects_a_value_that_would_be_ignored(value) -> None:
    with pytest.raises(ValueError):
        coerce_domain_cap(value)


def test_cap_keeps_the_first_rows_of_each_site_and_reports_the_drops() -> None:
    rows = [
        {"url": "https://a.example.com/1"},
        {"url": "https://b.other.test/1"},
        {"url": "https://b.example.com/2"},
        {"url": "https://a.example.com/3"},
        {"url": "https://other.test/2"},
    ]
    kept, dropped = cap_results_per_domain(rows, 2)
    assert [row["url"] for row in kept] == [
        "https://a.example.com/1",
        "https://b.other.test/1",
        "https://b.example.com/2",
        "https://other.test/2",
    ]
    assert dropped == 1


def test_cap_groups_a_two_label_public_suffix_by_its_registrable_domain() -> None:
    rows = [{"url": "https://a.example.co.uk/1"}, {"url": "https://b.example.co.uk/2"}, {"url": "https://other.co.uk/3"}]
    kept, dropped = cap_results_per_domain(rows, 1)
    assert [row["url"] for row in kept] == ["https://a.example.co.uk/1", "https://other.co.uk/3"]
    assert dropped == 1


def test_cap_off_returns_the_same_rows_in_order() -> None:
    rows = [{"url": "https://a.com/1"}, {"url": "https://a.com/2"}]
    assert cap_results_per_domain(rows, 0) == (rows, 0)
    assert cap_results_per_domain(rows, -1) == (rows, 0)


def test_cap_keeps_rows_without_a_readable_host() -> None:
    rows = [{"url": ""}, {"url": "not a URL"}, {"url": "https://a.com/1"}, {"url": "https://a.com/2"}]
    kept, dropped = cap_results_per_domain(rows, 1)
    assert [row["url"] for row in kept] == ["", "not a URL", "https://a.com/1"]
    assert dropped == 1
