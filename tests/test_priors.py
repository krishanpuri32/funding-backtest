import pandas as pd

from fb import priors


def test_collapse_phase():
    s = pd.Series(["EARLY_PHASE1", "PHASE1", "PHASE1/PHASE2", "PHASE2", "PHASE2/PHASE3", "PHASE3", "PHASE4", "NA", None, "WEIRD"])
    assert priors.collapse_phase(s).tolist() == ["Early 1", "1", "1-2", "2", "2-3", "3", "4", "N/A", "missing", "missing"]


def test_mesh_collapse_known_terms_and_priority():
    bc = pd.DataFrame([
        # T1: colorectal cancer -> Neoplasms + Digestive -> Neoplasms wins
        ("T1", "Colorectal Neoplasms", "mesh-list"), ("T1", "Neoplasms", "mesh-ancestor"), ("T1", "Digestive System Diseases", "mesh-ancestor"),
        # T2: HIV -> Infections + Immune System Diseases + Sexually Transmitted -> Infections wins
        ("T2", "HIV Infections", "mesh-list"), ("T2", "Infections", "mesh-ancestor"), ("T2", "Immune System Diseases", "mesh-ancestor"),
        # T3: smoking cessation -> only F01 'Behavior' ancestor -> Mental & behavioral
        ("T3", "Smoking Cessation", "mesh-list"), ("T3", "Behavior", "mesh-ancestor"),
        # T4: listed term IS a top-level heading, no ancestor rows
        ("T4", "Mental Disorders", "mesh-list"),
        # T5: rheumatoid arthritis -> Musculoskeletal + Skin/Connective + Immune -> Hemic/immune wins (priority 10 < 11)
        ("T5", "Arthritis, Rheumatoid", "mesh-list"), ("T5", "Musculoskeletal Diseases", "mesh-ancestor"),
        ("T5", "Skin and Connective Tissue Diseases", "mesh-ancestor"), ("T5", "Immune System Diseases", "mesh-ancestor"),
        # T6: only 'Pathological Conditions, Signs and Symptoms' -> Other
        ("T6", "Pain", "mesh-list"), ("T6", "Pathological Conditions, Signs and Symptoms", "mesh-ancestor"),
        # T7: MeSH rows but no heading from the table -> Other
        ("T7", "Motor Activity", "mesh-list"),
        # T8: pre-2023 C13 spelling
        ("T8", "Pre-Eclampsia", "mesh-list"), ("T8", "Female Urogenital Diseases and Pregnancy Complications", "mesh-ancestor"),
    ], columns=["nct_id", "mesh_term", "mesh_type"])
    out = priors.mesh_groups_for_trials(bc, pd.Series(["T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8", "T9"])).set_index("nct_id")
    assert out.loc["T1", "condition_group"] == "Neoplasms" and out.loc["T1", "n_top_headings"] == 2
    assert out.loc["T2", "condition_group"] == "Infections"
    assert out.loc["T3", "condition_group"] == "Mental & behavioral"
    assert out.loc["T4", "condition_group"] == "Mental & behavioral"
    assert out.loc["T5", "condition_group"] == "Hemic, lymphatic & immune"
    assert out.loc["T6", "condition_group"] == "Other / symptoms only"
    assert out.loc["T7", "condition_group"] == "Other / symptoms only"
    assert out.loc["T8", "condition_group"] == "Urogenital & pregnancy"
    assert out.loc["T9", "condition_group"] == "No MeSH term"  # no browse_conditions row at all
    assert len(priors.MESH_GROUP_PRIORITY) <= 15
    assert len(set(priors.MESH_GROUP_PRIORITY)) == len(priors.MESH_GROUP_PRIORITY)


def test_mesh_group_map_covers_every_group():
    m = priors.mesh_group_map()
    assert set(priors.MESH_GROUP_PRIORITY) <= set(m["group"])
    assert priors.MESH_GROUP_NONE in set(m["group"])


def test_activity_class_and_modal_tiebreak():
    s = pd.Series(["R01", "R21", "U10", "P30", "K23", "F32", "T32", None, "u01"])
    assert priors.activity_class(s).tolist() == ["R", "R", "U", "P", "K", "other", "other", "other", "U"]
    assert priors.modal_activity_class(pd.Series(["U", "U", "R"])) == "U"
    assert priors.modal_activity_class(pd.Series(["K", "P"])) == "P"        # tie -> P before K
    assert priors.modal_activity_class(pd.Series(["other", "U"])) == "U"
    assert priors.modal_activity_class(pd.Series([], dtype=object)) == "none"


def test_modal_value_and_ic_collapse():
    assert priors.modal_value(pd.Series(["HL", "CA", "CA", None])) == "CA"
    assert priors.modal_value(pd.Series(["HL", "CA"])) == "CA"  # alphabetical tie-break
    assert priors.modal_value(pd.Series([None, None])) is None
    out = priors.collapse_ic(pd.Series(["CA", "ca", "ZZ", None, ""]), top=["CA", "HL"])
    assert out.tolist() == ["CA", "CA", "other", "none", "none"]


def test_sponsor_class():
    assert priors.sponsor_class(pd.Series(["NIH", "Other", None])).tolist() == ["NIH", "OTHER", "missing"]
