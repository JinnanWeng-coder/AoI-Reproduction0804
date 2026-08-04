"""CLI entry point for the reproduction runner."""

from __future__ import annotations

import json
import sys

from config import build_parser, config_from_args, matrix_specs


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    config = config_from_args(args)
    if args.dry_run:
        print(json.dumps(config.to_dict(), indent=2, sort_keys=True))
        print(f"state_dim={config.state_dim} action_dim={config.action_dim}")
        print(f"run_name={config.run_name}")
        print(f"output_root={config.output_root}")
        print(f"eval_purpose={args.eval_purpose}")
        print("validation_eval_seeds=201,202,203,204,205,206")
        print("final_test_eval_seeds=101,102,103,104,105,106")
        if args.matrix:
            specs = matrix_specs(profile=args.profile)
            print(json.dumps(specs, indent=2, sort_keys=True))
            print(f"matrix_count={len(specs)} unique_count={len({(s['profile'], s['scenario'], s['seed']) for s in specs})}")
        return 0

    from runner import evaluate_from_checkpoint, train

    if args.eval_only:
        if not args.resume:
            raise SystemExit("--eval-only requires --resume <checkpoint>")
        eval_seeds = None if args.eval_seeds is None else [int(item) for item in args.eval_seeds.split(",") if item.strip()]
        result = evaluate_from_checkpoint(config, args.resume, args.eval_episodes, eval_seeds, args.eval_purpose)
    else:
        if args.eval_purpose != "final_test":
            raise SystemExit("--eval-purpose is only meaningful with --eval-only")
        result = train(config, resume=args.resume)
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
