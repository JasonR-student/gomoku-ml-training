package com.jasonrhan.gomoku.ml.model;

import com.jasonrhan.gomoku.ml.domain.BoardState;
import com.jasonrhan.gomoku.ml.selfplay.TrainingExample;

import java.util.List;
import java.util.Random;

public final class PolicyValueNetwork {
    public static final int INPUT_SIZE = BoardState.INPUT_SIZE;
    public static final int HIDDEN_ONE = 128;
    public static final int HIDDEN_TWO = 128;
    public static final int POLICY_SIZE = BoardState.POLICY_SIZE;

    private final float[] w1 = new float[INPUT_SIZE * HIDDEN_ONE];
    private final float[] b1 = new float[HIDDEN_ONE];
    private final float[] w2 = new float[HIDDEN_ONE * HIDDEN_TWO];
    private final float[] b2 = new float[HIDDEN_TWO];
    private final float[] policyWeights = new float[HIDDEN_TWO * POLICY_SIZE];
    private final float[] policyBias = new float[POLICY_SIZE];
    private final float[] valueWeights = new float[HIDDEN_TWO];
    private final float[] valueBias = new float[1];
    private final List<TrainableParameter> parameters;

    public PolicyValueNetwork(long seed) {
        initialize(new Random(seed));
        parameters = createParameters();
    }

    private PolicyValueNetwork(boolean empty) {
        parameters = createParameters();
    }

    public static PolicyValueNetwork empty() {
        return new PolicyValueNetwork(true);
    }

    private void initialize(Random random) {
        initializeArray(w1, random, Math.sqrt(2.0 / INPUT_SIZE));
        initializeArray(w2, random, Math.sqrt(2.0 / HIDDEN_ONE));
        initializeArray(policyWeights, random, Math.sqrt(2.0 / HIDDEN_TWO));
        initializeArray(valueWeights, random, Math.sqrt(2.0 / HIDDEN_TWO));
    }

    private void initializeArray(float[] values, Random random, double scale) {
        for (int i = 0; i < values.length; i++) {
            values[i] = (float) (random.nextGaussian() * scale);
        }
    }

    private List<TrainableParameter> createParameters() {
        return List.of(
                new TrainableParameter("w1", w1),
                new TrainableParameter("b1", b1),
                new TrainableParameter("w2", w2),
                new TrainableParameter("b2", b2),
                new TrainableParameter("policy_weights", policyWeights),
                new TrainableParameter("policy_bias", policyBias),
                new TrainableParameter("value_weights", valueWeights),
                new TrainableParameter("value_bias", valueBias)
        );
    }

    public List<TrainableParameter> parameters() {
        return parameters;
    }

    public Prediction predict(float[] input) {
        Forward forward = forward(input);
        return new Prediction(forward.policy, forward.value);
    }

    public BatchMetrics trainBatch(
            List<TrainingExample> examples,
            AdamOptimizer optimizer,
            double learningRate,
            double l2,
            double valueWeight
    ) {
        if (examples.isEmpty()) {
            return new BatchMetrics(0.0, 0.0, 0.0);
        }
        for (TrainableParameter parameter : parameters) {
            parameter.clearGradient();
        }

        double policyLoss = 0.0;
        double valueLoss = 0.0;
        for (TrainingExample example : examples) {
            Forward forward = forward(example.featuresUnsafe());
            float[] dLogits = new float[POLICY_SIZE];
            for (int action = 0; action < POLICY_SIZE; action++) {
                float target = example.policyTargetUnsafe()[action];
                dLogits[action] = forward.policy[action] - target;
                if (target > 0.0f) {
                    policyLoss -= target * Math.log(Math.max(1.0e-8, forward.policy[action]));
                }
            }
            double valueError = forward.value - example.valueTarget();
            valueLoss += 0.5 * valueError * valueError;

            float dValue = (float) (
                    valueWeight * valueError * (1.0 - forward.value * forward.value)
            );
            accumulatePolicyGradients(forward.hiddenTwo, dLogits);
            accumulateValueGradients(forward.hiddenTwo, dValue);

            float[] dHiddenTwo = new float[HIDDEN_TWO];
            for (int hidden = 0; hidden < HIDDEN_TWO; hidden++) {
                float gradient = dValue * valueWeights[hidden];
                for (int action = 0; action < POLICY_SIZE; action++) {
                    gradient += policyWeights[hidden * POLICY_SIZE + action] * dLogits[action];
                }
                dHiddenTwo[hidden] = forward.z2[hidden] > 0.0f ? gradient : 0.0f;
            }

            float[] dHiddenOne = new float[HIDDEN_ONE];
            for (int hidden = 0; hidden < HIDDEN_ONE; hidden++) {
                float gradient = 0.0f;
                for (int next = 0; next < HIDDEN_TWO; next++) {
                    gradient += w2[hidden * HIDDEN_TWO + next] * dHiddenTwo[next];
                }
                dHiddenOne[hidden] = forward.z1[hidden] > 0.0f ? gradient : 0.0f;
            }

            TrainableParameter w2Parameter = parameters.get(2);
            TrainableParameter b2Parameter = parameters.get(3);
            TrainableParameter w1Parameter = parameters.get(0);
            TrainableParameter b1Parameter = parameters.get(1);
            for (int hidden = 0; hidden < HIDDEN_ONE; hidden++) {
                b1Parameter.gradients()[hidden] += dHiddenOne[hidden];
                for (int input = 0; input < INPUT_SIZE; input++) {
                    w1Parameter.gradients()[input * HIDDEN_ONE + hidden]
                            += dHiddenOne[hidden] * forward.input[input];
                }
            }
            for (int hidden = 0; hidden < HIDDEN_TWO; hidden++) {
                b2Parameter.gradients()[hidden] += dHiddenTwo[hidden];
                for (int previous = 0; previous < HIDDEN_ONE; previous++) {
                    w2Parameter.gradients()[previous * HIDDEN_TWO + hidden]
                            += dHiddenTwo[hidden] * forward.hiddenOne[previous];
                }
            }
        }

        float scale = 1.0f / examples.size();
        for (TrainableParameter parameter : parameters) {
            for (int i = 0; i < parameter.gradients().length; i++) {
                parameter.gradients()[i] *= scale;
            }
        }
        optimizer.update(parameters, learningRate, 0.9, 0.999, 1.0e-8, l2);
        double averagePolicyLoss = policyLoss / examples.size();
        double averageValueLoss = valueLoss / examples.size();
        return new BatchMetrics(
                averagePolicyLoss,
                averageValueLoss,
                averagePolicyLoss + valueWeight * averageValueLoss
        );
    }

    private void accumulatePolicyGradients(float[] hiddenTwo, float[] dLogits) {
        TrainableParameter weightParameter = parameters.get(4);
        TrainableParameter biasParameter = parameters.get(5);
        for (int action = 0; action < POLICY_SIZE; action++) {
            biasParameter.gradients()[action] += dLogits[action];
            for (int hidden = 0; hidden < HIDDEN_TWO; hidden++) {
                weightParameter.gradients()[hidden * POLICY_SIZE + action]
                        += hiddenTwo[hidden] * dLogits[action];
            }
        }
    }

    private void accumulateValueGradients(float[] hiddenTwo, float dValue) {
        TrainableParameter weightParameter = parameters.get(6);
        TrainableParameter biasParameter = parameters.get(7);
        biasParameter.gradients()[0] += dValue;
        for (int hidden = 0; hidden < HIDDEN_TWO; hidden++) {
            weightParameter.gradients()[hidden] += hiddenTwo[hidden] * dValue;
        }
    }

    private Forward forward(float[] input) {
        if (input.length != INPUT_SIZE) {
            throw new IllegalArgumentException("Expected " + INPUT_SIZE + " input values");
        }
        float[] z1 = new float[HIDDEN_ONE];
        float[] hiddenOne = new float[HIDDEN_ONE];
        for (int hidden = 0; hidden < HIDDEN_ONE; hidden++) {
            float sum = b1[hidden];
            for (int index = 0; index < INPUT_SIZE; index++) {
                sum += input[index] * w1[index * HIDDEN_ONE + hidden];
            }
            z1[hidden] = sum;
            hiddenOne[hidden] = Math.max(0.0f, sum);
        }

        float[] z2 = new float[HIDDEN_TWO];
        float[] hiddenTwo = new float[HIDDEN_TWO];
        for (int hidden = 0; hidden < HIDDEN_TWO; hidden++) {
            float sum = b2[hidden];
            for (int previous = 0; previous < HIDDEN_ONE; previous++) {
                sum += hiddenOne[previous] * w2[previous * HIDDEN_TWO + hidden];
            }
            z2[hidden] = sum;
            hiddenTwo[hidden] = Math.max(0.0f, sum);
        }

        float[] logits = new float[POLICY_SIZE];
        float maxLogit = Float.NEGATIVE_INFINITY;
        for (int action = 0; action < POLICY_SIZE; action++) {
            float sum = policyBias[action];
            for (int hidden = 0; hidden < HIDDEN_TWO; hidden++) {
                sum += hiddenTwo[hidden] * policyWeights[hidden * POLICY_SIZE + action];
            }
            logits[action] = sum;
            maxLogit = Math.max(maxLogit, sum);
        }

        float[] policy = new float[POLICY_SIZE];
        double total = 0.0;
        for (int action = 0; action < POLICY_SIZE; action++) {
            policy[action] = (float) Math.exp(Math.max(-60.0f, logits[action] - maxLogit));
            total += policy[action];
        }
        for (int action = 0; action < POLICY_SIZE; action++) {
            policy[action] /= (float) total;
        }

        float value = valueBias[0];
        for (int hidden = 0; hidden < HIDDEN_TWO; hidden++) {
            value += hiddenTwo[hidden] * valueWeights[hidden];
        }
        value = (float) Math.tanh(value);
        return new Forward(input, z1, hiddenOne, z2, hiddenTwo, policy, value);
    }

    public record BatchMetrics(
            double policyLoss,
            double valueLoss,
            double totalLoss
    ) {
    }

    public static final class LoadedNetwork {
        private final PolicyValueNetwork network;
        private final AdamOptimizer optimizer;

        public LoadedNetwork(PolicyValueNetwork network, AdamOptimizer optimizer) {
            this.network = network;
            this.optimizer = optimizer;
        }

        public PolicyValueNetwork network() {
            return network;
        }

        public AdamOptimizer optimizer() {
            return optimizer;
        }
    }

    private record Forward(
            float[] input,
            float[] z1,
            float[] hiddenOne,
            float[] z2,
            float[] hiddenTwo,
            float[] policy,
            float value
    ) {
    }
}
