import pupil_labs.dynamic_content_on_rim as this_project


def test_package_metadata() -> None:
    assert hasattr(this_project, "__version__")
