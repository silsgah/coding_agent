"""Tests for infrastructure/tools.py — validates safety and correctness."""

from pathlib import Path

from infrastructure.tools import list_files, read_file, _is_safe_path


def test_is_safe_path_inside(tmp_path):
    """A file inside an allowed root should be safe."""
    child = tmp_path / "src" / "main.py"
    child.parent.mkdir(parents=True, exist_ok=True)
    child.touch()
    assert _is_safe_path(child, [str(tmp_path)]) is True


def test_is_safe_path_outside(tmp_path):
    """A file outside all allowed roots should be rejected."""
    outside = Path("/etc/passwd")
    assert _is_safe_path(outside, [str(tmp_path)]) is False


def test_read_file_blocked(tmp_path):
    """read_file should deny access to files outside allowed roots."""
    result = read_file("/etc/hosts", allowed_roots=[str(tmp_path)])
    assert "Access denied" in result


def test_read_file_success(tmp_path):
    """read_file should return contents for files inside allowed roots."""
    f = tmp_path / "hello.py"
    f.write_text("print('hello')")
    result = read_file(str(f), allowed_roots=[str(tmp_path)])
    assert result == "print('hello')"


def test_read_file_not_found():
    """read_file should handle missing files gracefully."""
    result = read_file("/nonexistent/file.py")
    assert "File not found" in result


def test_list_files(tmp_path):
    """list_files should find files matching the given extensions."""
    (tmp_path / "a.py").write_text("pass")
    (tmp_path / "b.txt").write_text("hello")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "c.py").write_text("pass")

    files = list_files([str(tmp_path)], [".py"])
    names = {f.name for f in files}
    assert "a.py" in names
    assert "c.py" in names
    assert "b.txt" not in names


def test_list_files_nonexistent_path():
    """list_files should handle missing paths gracefully."""
    files = list_files(["/nonexistent/repo"], [".py"])
    assert files == []
