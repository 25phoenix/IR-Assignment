from __future__ import annotations
import argparse
from src.pipeline import load_config, run


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Zipf/tokenization research pipeline")
    parser.add_argument("--mode", choices=("demo", "full"), default="demo")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    corpus, vocab, sweet, errors = run(args.mode, config)
    print(f"Completed {args.mode}: {len(corpus)} languages, {len(vocab)} vocabulary rows, {len(sweet)} sweet-spot rows")
    if errors:
        print("Optional experiment notes:")
        for error in errors:
            print(f"- {error}")


if __name__ == "__main__":
    main()
