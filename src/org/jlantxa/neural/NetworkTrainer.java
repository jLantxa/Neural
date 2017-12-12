/*
 * Copyright 2017 Javier Lancha Vázquez
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
*/

package org.jlantxa.neural;

import java.util.ArrayList;

public class NetworkTrainer
{
    private final NeuralNetwork mNetwork;

    // Learning parameters
    private final double mLearningRate;
    private final double mMomentum;

    private double mNetError;

    private double[] mTeachingInput;
    private double[] mTeachingOutput;

    /* The indexing of these vector considers index 0 as the first hidden layer,
     * as the input layer does not have to be trained
    */
    private final ArrayList<double[]> mDelta;
    private final ArrayList<double[]> mWeightDelta;

    public NetworkTrainer(NeuralNetwork net, double etta, double momentum) {
        mNetwork = net;
        mLearningRate = etta;
        mMomentum = momentum;

        int numLayers = net.getNumberOfLayers();

        mNetError = 0.0;

        mTeachingOutput = new double[net.getInputSize()];
        mTeachingOutput = new double[net.getOutputSize()];

        mDelta = new ArrayList<>(numLayers-1);
        mWeightDelta = new ArrayList<>(numLayers-1);

        for (int l = 1; l < numLayers; l++) {
            NeuralNetwork.Layer layer = net.mLayers.get(l);
            mDelta.add(new double[layer.getSize()]);
            mWeightDelta.add(new double[layer.getSize()]);
        }
    }

    public void inputTeachingData(double[] in, double[] out) {
        mTeachingInput = in;
        mTeachingOutput = out;
    }

    private void propagateNet() {
        mNetwork.execute(mTeachingInput);
    }

    private void backPropagateNet() {
        // TODO
    }

    private void update() {
        propagateNet();
        updateError();
        backPropagateNet();

        // TODO
    }

    private void updateError() {
        double[] out = mNetwork.getOutput();
        double err = 0;
        for (int o = 0; o < mNetwork.getOutputSize(); o++) {
            err += Math.pow(mTeachingOutput[o] - out[o], 2);
        }
        mNetError = 0.5 * err;
    }

    public void propagateForward(double[] input) {
        mNetwork.execute(input);
    }

    public double getLearningRate() {
        return mLearningRate;
    }

    public double getMomentum() {
        return mMomentum;
    }

    public double getNetError() {
        return mNetError;
    }
}
