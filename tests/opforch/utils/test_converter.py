import shutil

from opforch.utils import converter


def test_opf2txt(tmp_path):
    output = tmp_path / "boat.txt"
    converter.opf2txt("data/boat.dat", str(output))
    assert output.is_file()


def test_opf2csv(tmp_path):
    output = tmp_path / "boat.csv"
    converter.opf2csv("data/boat.dat", str(output))
    assert output.is_file()


def test_opf2json(tmp_path):
    output = tmp_path / "boat.json"
    converter.opf2json("data/boat.dat", str(output))
    assert output.is_file()


def test_default_output_path(tmp_path):
    source = tmp_path / "boat.dat"
    shutil.copyfile("data/boat.dat", source)

    converter.opf2csv(str(source))

    assert source.with_suffix(".csv").is_file()
