# kv_eviction: Native vLLM KV Cache Compaction for RL
#
# Importing kv_eviction (or anything under it like kv_eviction.types,
# which prime-rl's orchestrator does at startup) triggers kv_eviction.env's
# module-level monkey-patches that plumb vLLM's KV-eviction extension fields
# through verifiers' client adapter and MultiTurnEnv.add_model_response into
# the trajectory step's extras dict.
# Without these patches, verifiers silently drops compaction_events during
# response conversion, the trainer never sees compaction boundaries, and saved
# rollout outputs miss attention-matching robustness fields like shuffle_events
# and noise_events.
from . import env as env  # noqa: F401
