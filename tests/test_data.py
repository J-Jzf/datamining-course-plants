from pathlib import Path

import numpy as np


def test_cleaning_merges_infraspecific_records_and_legacy_province(tmp_path):
    # Keeping genus summaries, truncating hybrids, or treating AB as Alabama breaks this.
    (tmp_path / "plants.data").write_text(
        "acer,al,ab\n"
        "acer rubrum,al,nf\n"
        "acer rubrum var. drummondii,ab,lb\n"
        "quercus x hybrida,ca\n"
        "quercus x hybrida var. alba,or\n"
        "hibiscus tropicalis,pr,vi\n"
        "testus empty,gl\n", encoding="latin1")
    from plantmine.data import load_plants
    data = load_plants(tmp_path)
    assert data.names.tolist() == ["acer rubrum", "quercus x hybrida"]
    assert data.genera.tolist() == ["acer", "quercus"]
    assert data.matrix.dtype == np.bool_
    assert data.matrix.shape == (2, 64)
    recorded = [data.regions[i] for i in np.flatnonzero(data.matrix[0])]
    assert recorded == ["AB", "AL", "NL"]
    assert [data.regions[i] for i in np.flatnonzero(data.matrix[1])] == ["CA", "OR"]
    info = data.region_info.set_index("code")
    assert info.loc["AB", "name"] == "Alberta"
    assert info.loc["AL", "name"] == "Alabama"
    assert info.loc["PE", "name"] == "Prince Edward Island"
    assert data.audit["raw_rows"] == 7
    assert data.audit["genus_rows_removed"] == 1
    assert data.audit["species_without_in_scope_presence"] == 2
    assert data.audit["clean_species"] == 2
    assert data.audit["excluded_region_codes"]["pr"] == 1
    assert len(data.lineage) == 7


def test_preprocessing_is_row_order_invariant(tmp_path):
    from plantmine.data import load_plants
    lines = ["a b,ca", "a b var. c,or", "a c,al", "a,al,ca"]
    (tmp_path / "plants.data").write_text("\n".join(lines), encoding="latin1")
    first = load_plants(tmp_path)
    (tmp_path / "plants.data").write_text("\n".join(lines[::-1]), encoding="latin1")
    second = load_plants(tmp_path)
    np.testing.assert_array_equal(first.names, second.names)
    np.testing.assert_array_equal(first.matrix, second.matrix)

