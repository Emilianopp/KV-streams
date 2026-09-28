# SPDX-License-Identifier: Apache-2.0
"""Wire types for the kv-eviction integration layer.

CompactionEventWire transports admission metadata and TurnCompactionStateWire
anchors SGLang's cumulative client-carried replay state.

Architecture note: the authoritative struct definition lives in prime-rl at
prime_rl.transport.types, because it's a field on TrainingSample / MicroBatch
and those structs are owned by prime-rl. We re-export it here so the rest of
the kv-eviction integration (env wrapper, trainer dispatch, segmented forward)
has a single canonical import path that doesn't leak prime-rl internals:

    from kv_eviction.types import CompactionEventWire, TurnCompactionStateWire

Keeping the definition in prime-rl means:
1. prime-rl stays free of vllm imports (prime-rl never sees
   vllm.v1.core.compaction.types.CompactionEvent).
2. No inverted dependency: prime-rl doesn't import from kv_eviction.
3. TrainingSample's msgspec field type is a local type, not a foreign import.

The env wrapper is the single translation boundary: it reads vllm's
pydantic CompactionEventPayload from ChatCompletionResponse and converts
to CompactionEventWire before building TrainingSamples.
"""

from prime_rl.transport.types import (
    CompactionEventWire,
    TurnCompactionStateWire,
    compute_turn_compaction_state_id,
    training_effective_temperature,
    turn_compaction_state_to_json,
    validate_prefill_trim_event_field_types,
    validate_turn_compaction_state,
)

__all__ = [
    "CompactionEventWire",
    "TurnCompactionStateWire",
    "compute_turn_compaction_state_id",
    "training_effective_temperature",
    "turn_compaction_state_to_json",
    "validate_prefill_trim_event_field_types",
    "validate_turn_compaction_state",
]
