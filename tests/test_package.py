import bipl5


def test_package_imports():
    assert hasattr(bipl5, "ordination")
    assert isinstance(bipl5.__version__, str)
