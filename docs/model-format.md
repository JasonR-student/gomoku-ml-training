# Model and Data Format

The model file is a Java-only binary artifact with the following logical fields:

1. magic string
2. format version
3. input and output dimensions
4. named network parameters
5. Adam optimizer step and moments

The model manifest is JSON and contains the model version, board dimensions,
SHA-256 checksum, training game count, and search configuration.

Completed games are stored as one JSON object per line in `games.jsonl`.
Each line contains a game id, result, and ordered moves. The HTTP download
endpoint returns gzip-compressed JSONL in complete groups of 100 games.
