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

package org.jlantxa.neural.test;

import org.jlantxa.neural.NeuralNetwork;
import org.jlantxa.neural.TopologyException;
import org.jlantxa.neural.behaviour.IdentityFunction;
import org.jlantxa.neural.behaviour.LogisticFunction;

import java.util.ArrayList;

public class NetPerformanceTest
{
    private NetPerformanceTest() {

    }

    /**
     * Measure average performance of an arbitrary neural network
     * @param nTests Number of executions
     * @param layers Sizes of layers
     * @param bPrintVectors Print input and output vectors
     */
    public static void execute(int nTests, int[] layers, boolean bPrintVectors) {
        long nanoTimer, createNanos;
        ArrayList<Long> execute = new ArrayList<>();

        try {
            nanoTimer = System.nanoTime();
            NeuralNetwork net = new NeuralNetwork();
            for (int l = 0; l < layers.length; l++) {
                double[] biases = new double[layers[l]];

                for (int b = 0; b < biases.length; b++) {
                    biases[b] = Math.random();
                }


                if (l == 0) {
                    net.addLayer(biases, new IdentityFunction(), null);
                } else {
                    double[][] connections = new double[layers[l - 1]][layers[l]];

                    for (int h = 0; h < connections.length; h++) {
                        for (int k = 0; k < connections[0].length; k++) {
                            connections[h][k] = Math.random();
                        }
                    }

                    net.addLayer(biases, new LogisticFunction(), connections);
                }
            }

            createNanos = System.nanoTime() - nanoTimer;
            System.out.println("Create [ns] = " + createNanos);

            for (int t = 0; t < nTests; t++) {
                double[] input = new double[layers[0]];

                for (int i = 0; i < input.length; i++) {
                    input[i] = Math.random();
                }

                nanoTimer = System.nanoTime();
                double[] output = net.execute(input);
                execute.add(System.nanoTime() - nanoTimer);

                if (bPrintVectors) {
                    printVector(input, "Net " + t + " input:");
                    printVector(output, "Net " + t + " output:");
                }
            }

            long executeAvg = 0;
            for (Long anExecute : execute) {
                executeAvg += anExecute;
            }
            executeAvg /= execute.size();
            double throughput = 1 / (executeAvg/1000000000.0);
            System.out.println("Avg. throughput: " + throughput + " cycles per second");

        } catch (TopologyException te) {
            te.printStackTrace();
        }
    }

    private static void printVector(double[] vector, String msg) {
        System.out.println(msg);
        for (double element : vector) {
            System.out.println(element);
        }
        System.out.println();
    }
}
