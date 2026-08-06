"""CLI entry point for the reproduction runner."""

from __future__ import annotations

import json
import sys

from config import build_parser, config_from_args, matrix_specs


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    config = config_from_args(args)
    if args.diagnostic_eval and not args.eval_only:
        raise SystemExit("--diagnostic-eval is only valid with --eval-only")
    if args.dry_run:
        print(json.dumps(config.to_dict(), indent=2, sort_keys=True))
        print(f"state_dim={config.state_dim} action_dim={config.action_dim}")
        print(f"run_name={config.run_name}")
        print(f"output_root={config.output_root}")
        print(f"scope={args.scope}")
        print(f"eval_purpose={args.eval_purpose}")
        print(f"eval_noise={args.eval_noise}")
        print(f"diagnostic_eval={args.diagnostic_eval}")
        print(f"recover_empty_run={args.recover_empty_run}")
        print("validation_eval_seeds=201,202,203,204,205,206")
        print("final_test_eval_seeds=101,102,103,104,105,106")
        if args.matrix:
            specs = matrix_specs(profile=args.profile)
            print(json.dumps(specs, indent=2, sort_keys=True))
            print(f"matrix_count={len(specs)} unique_count={len({(s['profile'], s['scenario'], s['seed']) for s in specs})}")
        return 0

    from runner import evaluate_from_checkpoint, train

    if args.eval_only:
        if args.recover_empty_run:
            raise SystemExit("--recover-empty-run is only valid for training")
        if not args.resume:
            raise SystemExit("--eval-only requires --resume <checkpoint>")
        if args.scope == "train":
            raise SystemExit("--eval-only requires --scope validation or --scope final_release")
        if args.eval_purpose is None:
            raise SystemExit("--eval-only requires an explicit --eval-purpose")
        expected_scope = "validation" if args.eval_purpose == "validation" else "final_release"
        if args.scope != expected_scope:
            raise SystemExit(f"--scope {args.scope} does not match --eval-purpose {args.eval_purpose}")
        eval_seeds = None if args.eval_seeds is None else [int(item) for item in args.eval_seeds.split(",") if item.strip()]
        result = evaluate_from_checkpoint(
            config,
            args.resume,
            args.eval_episodes,
            eval_seeds,
            args.eval_purpose,
            scope=args.scope,
            eval_noise=args.eval_noise,
            diagnostic_eval=args.diagnostic_eval,
        )
    else:
        if args.scope != "train":
            raise SystemExit("training requires --scope train")
        if args.eval_purpose is not None:
            raise SystemExit("--eval-purpose is only meaningful with --eval-only")
        if args.recover_empty_run and args.resume:
            raise SystemExit("--recover-empty-run cannot be combined with --resume")
        result = train(config, resume=args.resume, recover_empty_run=args.recover_empty_run)
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
