# Gomoku ML Training

This is a headless Java 21 training project for Gomoku. It contains:

- a small policy-value MLP;
- PUCT/MCTS self-play;
- Renju-style black forbidden move checks;
- append-only replay and completed-game storage;
- checkpoints and resumable training;
- an HTTP data channel for complete 100-game batches.

## Build

```bash
mvn test
mvn -q -DskipTests package
```

## Run training on the server

```bash
export DATA_TOKEN='replace-with-a-long-random-token'
java -jar target/gomoku-ml-training-1.0.0.jar train \
  --hours 24 \
  --output ./runs \
  --serve-data \
  --data-host 0.0.0.0 \
  --data-port 5000 \
  --data-port-end 5005 \
  --data-token "$DATA_TOKEN"
```

The server tries ports 5000 through 5005 and prints the selected port.
Open that port in the server firewall/security group.

## Download complete games locally

```bash
java -jar gomoku-ml-training-1.0.0.jar download-data \
  --url http://SERVER_IP:5000 \
  --output ./downloads \
  --token "$DATA_TOKEN"
```

The client downloads only complete groups of 100 games. If the server has
4578 completed games, it downloads 4500. It stores progress in
`downloads/.downloaded-games`, so a later command downloads only new complete
batches.

Each file is named like:

```text
games-0000000-0000100.jsonl.gz
```

## Resume training

```bash
java -jar target/gomoku-ml-training-1.0.0.jar train \
  --resume ./runs/run-YYYYMMDD-HHmmss \
  --hours 24 \
  --serve-data \
  --data-token "$DATA_TOKEN"
```
