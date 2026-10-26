import argparse
import json
import subprocess
import sys

from taxi import config


def run_dbt(args, target="dev", extra_vars=None):
    cmd = list(args) + ["--project-dir", ".", "--profiles-dir", ".", "--target", target]
    if extra_vars:
        cmd += ["--vars", json.dumps(extra_vars)]
    # run in a subprocess so dbt lets go of the duckdb file when it's done
    # (profiles.yml paths are relative to dbt/)
    code = "import sys; from dbt.cli.main import cli; sys.exit(cli())"
    if subprocess.run([sys.executable, "-c", code] + cmd, cwd=config.DBT_DIR).returncode != 0:
        sys.exit(f"dbt {args[0]} failed")


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m taxi")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("download", help="fetch trips, zones and weather")
    sub.add_parser("load", help="parse weather, load raw tables into duckdb")
    sub.add_parser("build", help="dbt deps + build")
    bt = sub.add_parser("backtest", help="train and score models for each test month")
    bt.add_argument("--models", nargs="+")
    bt.add_argument("--horizons", nargs="+", type=int)
    bt.add_argument("--folds", nargs="+", help="test months, e.g. 2026-01")
    bt.add_argument("--tag", help="name for the run files (default: the model names)")
    sub.add_parser("tune", help="recheck LightGBM settings on Oct-Nov 2025")
    sub.add_parser("report", help="pool backtest runs into metrics")
    sub.add_parser("weather", help="hourly weather regressions")
    sub.add_parser("export", help="write dashboard tables")
    s = sub.add_parser("sample", help="whole pipeline on the committed sample")
    s.add_argument("--make", action="store_true", help="rebuild sample_data/ from the full data first")
    args = p.parse_args(argv)

    if args.cmd == "download":
        from taxi import ingest
        ingest.download_zones()
        ingest.download_weather()
        ingest.download_trips()
    elif args.cmd == "load":
        from taxi import ingest
        ingest.build_weather_csv()
        ingest.load()
    elif args.cmd == "build":
        run_dbt(["deps"])
        run_dbt(["build"])
    elif args.cmd == "backtest":
        from taxi import backtest
        backtest.run(models=args.models, horizons=args.horizons, folds=args.folds, tag=args.tag)
    elif args.cmd == "tune":
        from taxi import tune
        tune.run()
    elif args.cmd == "report":
        from taxi import backtest
        backtest.report()
    elif args.cmd == "weather":
        from taxi import weather_effect
        weather_effect.run()
    elif args.cmd == "export":
        from taxi import export
        export.run()
    elif args.cmd == "sample":
        from taxi import sample
        if args.make:
            sample.make()
        sample.run()
