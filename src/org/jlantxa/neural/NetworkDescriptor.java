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
import org.jlantxa.neural.behaviour.IdentityFunction;
import org.jlantxa.neural.behaviour.LogisticFunction;

import java.util.ArrayList;

/**
 * A NetworkDescriptor describes the topology of a NeuralNetwork
 */
public class NetworkDescriptor
{
    /**
     * Activation functions. This enum contains simple representations of the activation function classes
     */
    enum BehaviourType {
        IDENTITY,
        LOGISTIC
    }

    final ArrayList<LayerDescriptor> mLayers;
    final ArrayList<double[][]> mConnections;

    /**
     * Create an empty NetworkDescriptor. Layers must be added sequentially.
     */
    public NetworkDescriptor() {
        mLayers = new ArrayList<>();
        mConnections = new ArrayList<>();
    }

    /**
     * Add a layer, including connection matrix and activation function
     * @param biases vector of biases
     * @param behaviour activation function
     * @param connections connection matrix
     * @throws TopologyException Throws TopologyException when the new layer does not match with
     * the previous layer and/or connection matrix.
     */
    public void addLayer(double[] biases, BehaviourType behaviour, double[][] connections) throws TopologyException
    {
        if (biases == null || behaviour == null) {
            throw new TopologyException("Bias and/or behaviour objects are null.");
        }

        if (biases.length <= 0) {
            throw new TopologyException("Bias vector is empty.");
        }

        LayerDescriptor layer = new LayerDescriptor(biases, behaviour);

        // Is this the first layer?
        if (mLayers.size() <= 0) {
            // Add layer, don't mind the connections
            mLayers.add(layer);
        }
        // or a hidden / output layer?
        else {
            if (connections == null) {
                throw new TopologyException("The connection matrix is null.");
            }

            int hSize = connections.length;
            int kSize = connections[0].length;

            // Connection matrix has wrong dimensions?
            int netSize = mLayers.size();
            if (layer.size() != kSize || hSize != mLayers.get(netSize - 1).size()) {
                throw new TopologyException("The dimensions of the connection matrix " +
                        "do not match the net's topology.\n" +
                        "Lhk = (" + layer.size() + ", " + mLayers.get(netSize - 1).size() + "), " +
                        "hk = (" + hSize + ", " + kSize + ")");
            }

            // Add layer and connections
            mLayers.add(layer);
            mConnections.add(connections);
        }
    }

    /**
     * Remove the last layer from the descriptor
     */
    public void removeLayer() {
        int numLayers = mLayers.size();
        if (numLayers <= 0) {
            return;
        }

        mLayers.remove(numLayers - 1);
        int numConnections = mConnections.size();
        if (numConnections > 0) {
            mConnections.remove(numConnections - 1);
        }
    }

    /**
     * Get an ArrayList of all layer descriptors in this NetworkDescriptor
     * @return ArrayList of LayerDescriptors
     */
    ArrayList<LayerDescriptor> getLayerDescriptors() {
        return mLayers;
    }

    /**
     * Get an ArrayList of all connection matrices in this NetworkDescriptor
     * @return ArrayList of connection matrices
     */
    ArrayList<double[][]> getConnectionDescriptors() {
        return mConnections;
    }

    /**
     * Get behaviour descriptor constant from a Behaviour instance
     * @param behaviour An implementation of the Behaviour interface
     * @return A constant describing the type of behaviour instance
     */
    static NetworkDescriptor.BehaviourType parseBehaviourType(Behaviour behaviour) {
        if (behaviour instanceof IdentityFunction) {
            return NetworkDescriptor.BehaviourType.IDENTITY;
        } else if (behaviour instanceof LogisticFunction) {
            return NetworkDescriptor.BehaviourType.LOGISTIC;
        }

        // Return LOGISTIC as default
        return NetworkDescriptor.BehaviourType.LOGISTIC;
    }

    /**
     * A LayerDescriptor is a simple representation of a network layer. It contains the biases of all neurons
     * and the behaviour type of the layer.
     */
    class LayerDescriptor
    {
        final double[] biases;
        final BehaviourType behaviourType;

        LayerDescriptor(double[] biases, BehaviourType behaviourType) {
            this.biases = biases;
            this.behaviourType = behaviourType;
        }

        int size() {
            return this.biases.length;
        }
    }
}
