package com.jasonrhan.gomoku.ml;

import com.jasonrhan.gomoku.ml.data.DataDownloadServer;
import com.jasonrhan.gomoku.ml.data.GameArchiveStore;
import com.jasonrhan.gomoku.ml.data.ReplayBufferStore;
import com.jasonrhan.gomoku.ml.domain.GameResult;
import com.jasonrhan.gomoku.ml.domain.PlayerColor;
import com.jasonrhan.gomoku.ml.selfplay.SelfPlayRunner;
import com.jasonrhan.gomoku.ml.selfplay.TrainingExample;
import org.junit.jupiter.api.Test;

import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.zip.GZIPInputStream;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class ReplayAndDownloadTest {
    @Test
    void replayStoreReloadsExamples() throws Exception {
        Path directory = Files.createTempDirectory("gomoku-replay");
        TrainingExample example = example();
        try (ReplayBufferStore store = new ReplayBufferStore(directory, 100)) {
            store.append(example);
            assertEquals(1, store.size());
        }
        try (ReplayBufferStore store = new ReplayBufferStore(directory, 100)) {
            assertEquals(1, store.size());
            assertEquals(112, store.sample(1, new java.util.Random(1)).get(0).action());
        }
    }

    @Test
    void downloadEndpointReturnsOnlyCompleteHundredGameBatches() throws Exception {
        Path directory = Files.createTempDirectory("gomoku-archive");
        try (GameArchiveStore archive = new GameArchiveStore(
                directory.resolve("games.jsonl")
        )) {
            for (int i = 0; i < 101; i++) {
                archive.append(new SelfPlayRunner.GameRecord(
                        GameResult.DRAW,
                        0,
                        List.of(),
                        List.of()
                ));
            }
            try (DataDownloadServer server = new DataDownloadServer(archive, "secret")) {
                int port = server.start("127.0.0.1", 51000, 51005);
                HttpClient client = HttpClient.newHttpClient();
                HttpRequest request = HttpRequest.newBuilder(
                                URI.create("http://127.0.0.1:" + port
                                        + "/api/download?from=0&count=101")
                        )
                        .header("Authorization", "Bearer secret")
                        .GET()
                        .build();
                HttpResponse<byte[]> response = client.send(
                        request,
                        HttpResponse.BodyHandlers.ofByteArray()
                );
                assertEquals(200, response.statusCode());
                assertEquals("100", response.headers().firstValue(
                        "X-Game-Count"
                ).orElse(""));
                String body = gunzip(response.body());
                assertEquals(100, body.lines().count());
            }
        }
    }

    private TrainingExample example() {
        float[] policy = new float[225];
        policy[112] = 1.0f;
        return new TrainingExample(
                new float[675],
                policy,
                0.0f,
                112,
                PlayerColor.BLACK
        );
    }

    private String gunzip(byte[] value) throws IOException {
        try (GZIPInputStream gzip = new GZIPInputStream(
                new ByteArrayInputStream(value)
        )) {
            return new String(gzip.readAllBytes(), StandardCharsets.UTF_8);
        }
    }
}
