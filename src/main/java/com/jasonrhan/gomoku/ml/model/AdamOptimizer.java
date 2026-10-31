package com.jasonrhan.gomoku.ml.model;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public final class AdamOptimizer {
    private final Map<String, float[]> firstMoment = new HashMap<>();
    private final Map<String, float[]> secondMoment = new HashMap<>();
    private long step;

    public void update(
            List<TrainableParameter> parameters,
            double learningRate,
            double beta1,
            double beta2,
            double epsilon,
            double l2
    ) {
        step++;
        double correction1 = 1.0 - Math.pow(beta1, step);
        double correction2 = 1.0 - Math.pow(beta2, step);
        for (TrainableParameter parameter : parameters) {
            float[] values = parameter.values();
            float[] gradients = parameter.gradients();
            float[] m = firstMoment.computeIfAbsent(
                    parameter.name(), key -> new float[values.length]
            );
            float[] v = secondMoment.computeIfAbsent(
                    parameter.name(), key -> new float[values.length]
            );
            for (int i = 0; i < values.length; i++) {
                double gradient = gradients[i] + l2 * values[i];
                m[i] = (float) (beta1 * m[i] + (1.0 - beta1) * gradient);
                v[i] = (float) (beta2 * v[i] + (1.0 - beta2) * gradient * gradient);
                double mHat = m[i] / correction1;
                double vHat = v[i] / correction2;
                values[i] -= (float) (learningRate * mHat / (Math.sqrt(vHat) + epsilon));
            }
            parameter.clearGradient();
        }
    }

    public long step() {
        return step;
    }

    public void writeTo(DataOutput output, List<TrainableParameter> parameters)
            throws IOException {
        output.writeLong(step);
        for (TrainableParameter parameter : parameters) {
            float[] m = firstMoment.getOrDefault(
                    parameter.name(), new float[parameter.values().length]
            );
            float[] v = secondMoment.getOrDefault(
                    parameter.name(), new float[parameter.values().length]
            );
            output.writeUTF(parameter.name());
            output.writeInt(m.length);
            for (float value : m) {
                output.writeFloat(value);
            }
            for (float value : v) {
                output.writeFloat(value);
            }
        }
    }

    public static AdamOptimizer readFrom(
            DataInput input,
            List<TrainableParameter> parameters
    ) throws IOException {
        AdamOptimizer optimizer = new AdamOptimizer();
        optimizer.step = input.readLong();
        Map<String, TrainableParameter> known = new HashMap<>();
        for (TrainableParameter parameter : parameters) {
            known.put(parameter.name(), parameter);
        }
        for (int i = 0; i < parameters.size(); i++) {
            String name = input.readUTF();
            int length = input.readInt();
            float[] m = new float[length];
            float[] v = new float[length];
            for (int j = 0; j < length; j++) {
                m[j] = input.readFloat();
            }
            for (int j = 0; j < length; j++) {
                v[j] = input.readFloat();
            }
            TrainableParameter parameter = known.get(name);
            if (parameter == null || parameter.values().length != length) {
                throw new IOException("Optimizer parameter mismatch: " + name);
            }
            optimizer.firstMoment.put(name, m);
            optimizer.secondMoment.put(name, v);
        }
        return optimizer;
    }
}
