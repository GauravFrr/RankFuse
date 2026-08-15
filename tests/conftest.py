import shutil
import tempfile

import pytest


@pytest.fixture
def temp_dir():
    """Fixture to provide a temporary directory that is automatically cleaned up."""
    tmpdir = tempfile.mkdtemp()
    yield tmpdir
    shutil.rmtree(tmpdir, ignore_errors=True)
