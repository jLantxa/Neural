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

import org.jlantxa.neural.behaviour.Behaviour;

import java.util.ArrayList;

public class NeuralNetwork
{
    private ArrayList<Layer> mLayers;
    private ArrayList<double[][]> mConnections;

    public NeuralNetwork() {
        mLayers = new ArrayList<>();
        mConnections = new ArrayList<>();
    }


    public boolean addLayer(Layer layer, double[][] connections) {
        if (layer == null) {
            return false;
        }

        int layerSize = layer.getSize();
        if (layerSize <= 0) {
            return false;
        }

        // Is this the first layer?
        if (mLayers.size() <= 0) {
            // Add layer, don't mind the connections
            mLayers.add(layer);
            return true;
        }
        // or a hidden / output layer?
        else {
            if (connections == null) {
                return false;
            }

            int hSize = connections.length;
            int kSize = connections[0].length;

            // Connection matrix has wrong dimensions?
            int netSize = mLayers.size();
            if (layerSize != kSize || hSize != mLayers.get(netSize - 1).getSize()) {
                return false;
            }

            // Add layer and connections
            mLayers.add(layer);
            mConnections.add(connections);
            return true;
        }
    }

    /**
     * Propagate output of layer k-1 through layer k
     * @param k layer index
     */
    private void propagateLayer(int k) {
        // h := previous layer index
        // k := current layer index
        int h = k - 1;

        // Check index bounds
        if (h <= 0) return; // Input layer or previous
        if (k >= mLayers.size()) return;    // Beyond output layer

        double[] hOutput = mLayers.get(h).output;
        Layer layer_k = mLayers.get(k);

        int size_h = mLayers.get(h).getSize();
        int size_k = layer_k.getSize();
        double[][] connection = mConnections.get(k);
        double[] k_input = new double[size_k];

        // Weighted sum
        for (int kn = 0; kn < size_k; kn++) {
            k_input[kn] = 0;
            for (int hn = 0; hn < size_h; hn++) {
                k_input[kn] += connection[hn][kn] * hOutput[hn];
            }
        }

        // Propagate the layer
        layer_k.propagate(k_input);
    }

    /**
     * Execute a cycle of the network
     * @param netInput Network input vector
     * @return Network output vector
     */
    public double[] execute(double[] netInput) {
        int netSize = mLayers.size();
        mLayers.get(0).propagate(netInput);
        for (int l = 1; l < netSize; l++) {
            propagateLayer(l);
        }

        return getOutput();
    }

    /**
     * Get output vector
     * @return output vector
     */
    public double[] getOutput() {
        return mLayers.get(mLayers.size() - 1).getOutput();
    }


    /**
     * A layer is a collection of neurons of equal behaviour each of them equally distant from the input layer.
     */
    private class Layer
    {
        private int size;
        private Behaviour behaviour;
        private double[] bias;
        private double[] output;


        public Layer(int size, Behaviour behaviour) {
            this.size = size;
            this.behaviour = behaviour;

            this.bias = new double[size];
            this.output = new double[size];
        }

        public Layer(double[] biases, Behaviour behaviour) {
            this.size = biases.length;
            this.behaviour = behaviour;

            this.bias = biases;
            this.output = new double[this.size];
        }


        /**
         * Propagate input and activate all neurons
         * @param layerInput input vector to the layer after applying connection weights
         */
        public void propagate(double[] layerInput) {
            for (int n = 0; n < size; n++) {
                output[n] = behaviour.activation(layerInput[n] - bias[n]);
            }
        }

        /**
         * Get the size of the layer
         * @return size of the layer
         */
        public int getSize() {
            return this.size;
        }

        /**
         * Get the activation function of the layer
         * @return activation function (Behaviour) of the layer
         */
        public Behaviour getBehaviour() {
            return behaviour;
        }

        /**
         * Get the bias of neuron at index n
         * @param n neuron index
         * @return bias of neuron n within the layer
         */
        public double getBias(int n) {
            return this.bias[n];
        }

        /**
         * Get the output vector of the whole layer
         * @return Output vector of the layer
         */
        public double[] getOutput() {
            return this.output;
        }

        /**
         * Get the output from neuron at index n within the layer
         * @param n neuron index
         * @return Output of neuron n
         */
        public double getOutputAt(int n) {
            return this.output[n];
        }
    }
}
