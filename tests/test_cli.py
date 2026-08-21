from pathlib import Path

from battery_lab.cli import _cmd_validate_configs, build_parser


def test_cli_parser_exposes_doctor_and_validation():
    parser = build_parser()
    assert parser.prog == "battery-lab"


def test_all_repository_configs_validate():
    root = Path(__file__).resolve().parents[1] / "configs"
    args = type("Args", (), {"path": str(root)})()
    assert _cmd_validate_configs(args) == 0
