"""CLI smoke tests for the fast, no-training subcommands."""


from muonet.cli import build_parser, main


def test_parser_builds():
    parser = build_parser()
    args = parser.parse_args(["recon", "--scene", "tungsten_cube", "--method", "poca"])
    assert args.command == "recon"
    assert args.method == ["poca"]


def test_info_runs(capsys):
    assert main(["info"]) == 0
    out = capsys.readouterr().out
    assert "tungsten" in out
    assert "realistic_sipm" in out


def test_recon_poca_runs(capsys):
    code = main(["recon", "--scene", "tungsten_cube", "--dosage", "128", "--method", "poca"])
    assert code == 0
    assert "PSNR" in capsys.readouterr().out


def test_recon_learned_without_checkpoint_errors():
    code = main(["recon", "--scene", "tungsten_cube", "--dosage", "128", "--method", "learned"])
    assert code == 2


def test_recon_with_config(tmp_path, capsys):
    cfg = tmp_path / "run.yaml"
    cfg.write_text("scene: lead_sphere\ndetector: ideal\ndosage: 128\nseed: 1\n")
    assert main(["recon", "--config", str(cfg), "--method", "poca"]) == 0
    assert "lead_sphere" in capsys.readouterr().out


def test_export_poca_only(tmp_path):
    code = main(
        [
            "export",
            "--scenes",
            "tungsten_cube",
            "--dosages",
            "128",
            "256",
            "--out",
            str(tmp_path),
        ]
    )
    assert code == 0
    manifest = tmp_path / "manifest.json"
    assert manifest.exists()
    import json

    data = json.loads(manifest.read_text())
    assert data["methods"] == ["poca"]
    assert len(data["scenes"]) == 1
    # A ground-truth and a poca volume exist for the scene.
    assert (tmp_path / "tungsten_cube_gt.bin").exists()
    assert (tmp_path / "tungsten_cube_poca_128.bin").exists()
