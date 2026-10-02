# Table I representation codecs

This directory freezes the action-representation assets used by the seven-hand Table I experiments.

- `codecs/faas7_project_v3_sided.json`: sided-v3 FAAS contract, SHA-256 `57ec9c7b73d57e866b816924f3bea70b93a058ab379bd084bb8a1bb4526cc51c`.
- `codecs/selected.pt`: private frozen DexLatent checkpoint, SHA-256 `73b1c80497a493835e58ace3e3abe509906d63916a4c263f5191d585a38bbd7b`; the binary is not distributed in this source repository.
- `codecs/frozen_seal.json`: checkpoint-selection seal.
- `faas7.py` and `dexlatent7.py`: codec implementations matching the experiment server.

The contracts, implementation and selection seal are retained for version matching. Supply a matching local encoder to reproduce DexLatent; all encoder and policy weights remain private. A later shared-v4 engineering contract exists in the historical v8 package, but it was not used to produce the Table I results.
