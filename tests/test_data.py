from faircredit.data import AUDITED, CATEGORICAL, FEATURES, PROTECTED, TARGET


def test_shape_and_target(df):
    assert len(df) == 1000
    assert df[TARGET].sum() == 300  # 300 bad, 700 good


def test_all_codes_decoded(df):
    for col in CATEGORICAL:
        assert df[col].notna().all(), col
        assert not df[col].astype(str).str.match(r"^A\d+$").any(), col


def test_protected_attributes_not_used_as_features():
    assert not set(PROTECTED) & set(FEATURES)
    assert "age" not in FEATURES and "personal_status" not in FEATURES
    assert set(AUDITED) <= set(PROTECTED)


def test_derived_groups(df):
    assert set(df["sex"]) == {"male", "female"}
    assert set(df["age_group"]) == {"under 25", "25 and over"}
    assert (df.loc[df["age_group"] == "under 25", "age"] < 25).all()
