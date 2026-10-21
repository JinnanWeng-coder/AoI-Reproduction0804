"""Action-repeat streams, independent of the environment's private generator."""
from __future__ import annotations

import copy
import numpy as np

MAPPO_TAG = 0x4D415050
MADDPG_TAG = 0xA01
REPEAT_TAG = 0x44504C59
RANDOM_TAG = 0x52414E44


def random_action_seed_words(action_stream_id: int, world: int) -> list[int]:
    """Version1 Uniform Random deployment stream, not a training seed/repeat0."""
    if type(action_stream_id) is not int or type(world) is not int or min(action_stream_id, world) < 0:
        raise ValueError("invalid Random action stream/world identity")
    return [action_stream_id, world, RANDOM_TAG, 1]


def action_seed_words(seed: int, world: int, repeat: int, family: str) -> list[int]:
    if family not in {"mappo", "maddpg"} or repeat < 0 or min(seed, world) < 0:
        raise ValueError("invalid action RNG identity")
    words = [int(seed), int(world), MAPPO_TAG if family == "mappo" else MADDPG_TAG]
    # repeat 0 exactly preserves the original evaluators' SeedSequence inputs.
    return words if repeat == 0 else words + [REPEAT_TAG, int(repeat)]


def policy_seed(seed: int, world: int, repeat: int) -> int:
    return int(np.random.SeedSequence(action_seed_words(seed, world, repeat, "mappo"))
               .generate_state(1, dtype=np.uint32)[0])


def exogenous_probe(environment) -> dict:
    """Small exact probe; excludes action-dependent AoI/interference/observations."""
    return {"rng_state": copy.deepcopy(environment.rng.bit_generator.state),
            "vehicles": [[list(v.position), v.direction, float(v.velocity)] for v in environment.vehicles],
            "v2i_shadowing": environment.v2i_shadowing.tolist(),
            "v2v_shadowing": environment.v2v_shadowing.tolist(),
            "v2i_channels_fast": environment.v2i_channels_fast.tolist()}
