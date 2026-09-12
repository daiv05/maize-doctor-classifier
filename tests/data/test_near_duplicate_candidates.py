import numpy as np

from scripts.checks.audit_near_duplicates import candidate_pairs


def test_candidate_search_matches_brute_force():
    random = np.random.default_rng(42)
    hashes = [int(value) for value in random.integers(0, 2**63, 70, dtype=np.int64)]
    hashes.extend([hashes[0], hashes[0] ^ 15, hashes[1] ^ 1, hashes[2] ^ 255])
    expected = [
        (a, b, (hashes[a] ^ hashes[b]).bit_count())
        for b in range(len(hashes))
        for a in range(b)
        if (hashes[a] ^ hashes[b]).bit_count() <= 4
    ]
    assert list(candidate_pairs(hashes, radius=4)) == expected
