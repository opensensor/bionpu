from bionpu.genomics.offtarget_seed import (
    encode_seed_2bit,
    prefilter_offtargets,
    reverse_complement,
)


def _coords(candidates):
    return [(c.guide_id, c.ref_name, c.position, c.strand) for c in candidates]


def test_prefilter_reports_perfect_plus_seed_hit_with_metadata():
    guide = "AAAACCCCGGGGTTTTAAAA"
    ref = "CCCC" + guide + "AGG" + "TTTT"

    candidates = prefilter_offtargets(
        {"g1": guide},
        {"chrSynthetic": ref},
        seed_length=8,
        pam="NGG",
    )

    assert _coords(candidates) == [("g1", "chrSynthetic", 4, "+")]
    hit = candidates[0]
    assert hit.seed == "TTTTAAAA"
    assert hit.target_seed == "TTTTAAAA"
    assert hit.seed_key == encode_seed_2bit("TTTTAAAA")
    assert hit.target_seed_key == hit.seed_key
    assert hit.seed_mismatches == 0
    assert hit.seed_mismatch_positions == ()
    assert hit.seed_hit_count == 1
    assert hit.pam == "AGG"


def test_prefilter_reports_reverse_complement_seed_hit():
    guide = "AAAACCCCGGGGTTTTAAAA"
    genomic_protospacer = reverse_complement(guide)
    ref = "CCT" + genomic_protospacer + "CCCC"

    candidates = prefilter_offtargets(
        {"g1": guide},
        {"chrSynthetic": ref},
        seed_length=8,
        pam="NGG",
    )

    assert _coords(candidates) == [("g1", "chrSynthetic", 3, "-")]
    assert candidates[0].seed == "TTTTAAAA"
    assert candidates[0].target_seed == "TTTTAAAA"
    assert candidates[0].pam == "AGG"


def test_pam_aware_filtering_can_exclude_or_include_same_seed():
    guide = "AAAACCCCGGGGTTTTAAAA"
    ref = "CCCC" + guide + "ATG" + "TTTT"

    assert (
        prefilter_offtargets(
            {"g1": guide},
            {"chrSynthetic": ref},
            seed_length=8,
            pam="NGG",
            pam_aware=True,
        )
        == []
    )

    candidates = prefilter_offtargets(
        {"g1": guide},
        {"chrSynthetic": ref},
        seed_length=8,
        pam="NGG",
        pam_aware=False,
    )

    assert _coords(candidates) == [("g1", "chrSynthetic", 4, "+")]
    assert candidates[0].pam is None


def test_reference_and_guide_n_bases_do_not_emit_candidates():
    guide = "AAAACCCCGGGGTTTTAAAA"
    ref_with_n_seed = "CCCC" + guide[:-1] + "N" + "AGG" + "TTTT"
    ref_with_n_outside_seed = "CCCC" + "N" + guide[1:] + "AGG" + "TTTT"

    assert (
        prefilter_offtargets(
            {"g1": guide},
            {"chrSynthetic": ref_with_n_seed},
            seed_length=8,
            pam="NGG",
        )
        == []
    )
    assert (
        prefilter_offtargets(
            {"g1": guide},
            {"chrSynthetic": ref_with_n_outside_seed},
            seed_length=8,
            pam="NGG",
        )
        == []
    )

    guide_with_n_seed = guide[:-1] + "N"
    guide_with_n_outside_seed = "N" + guide[1:]
    ref = "CCCC" + guide + "AGG" + "TTTT"
    assert (
        prefilter_offtargets(
            {"gN": guide_with_n_seed},
            {"chrSynthetic": ref},
            seed_length=8,
            pam="NGG",
        )
        == []
    )
    assert (
        prefilter_offtargets(
            {"gN": guide_with_n_outside_seed},
            {"chrSynthetic": ref},
            seed_length=8,
            pam="NGG",
        )
        == []
    )


def test_seed_mismatch_threshold_reports_positions_and_count():
    guide = "AAAACCCCGGGGTTTTAAAA"
    target = "AAAACCCCGGGGTTTTAAAC"
    ref = "CCCC" + target + "AGG"

    assert (
        prefilter_offtargets(
            {"g1": guide},
            {"chrSynthetic": ref},
            seed_length=8,
            max_seed_mismatches=0,
            pam="NGG",
        )
        == []
    )

    candidates = prefilter_offtargets(
        {"g1": guide},
        {"chrSynthetic": ref},
        seed_length=8,
        max_seed_mismatches=1,
        pam="NGG",
    )

    assert _coords(candidates) == [("g1", "chrSynthetic", 4, "+")]
    assert candidates[0].target_seed == "TTTTAAAC"
    assert candidates[0].seed_mismatches == 1
    assert candidates[0].seed_mismatch_positions == (7,)
