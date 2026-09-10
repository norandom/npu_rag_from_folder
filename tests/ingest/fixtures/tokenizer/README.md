# Offline tokenizer fixture

Committed tokenizer files for `Alibaba-NLP/gte-modernbert-base` (Apache-2.0,
ungated), the embedding runtime's third candidate. Requirement 6.7's contract
test loads these with no network and must fail rather than skip if they are
absent.

Only files named in `npu_rag.embedding.tokenize.TOKENIZER_FILE_PATTERNS` are
stored here. Model weights (`model.onnx`, `model.safetensors`,
`pytorch_model.bin`) are not committed.

Source snapshot: `e7f32e3c00f91d699e8c43b53106206bcc72bb22`.
