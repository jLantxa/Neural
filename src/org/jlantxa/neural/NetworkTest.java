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

import org.xml.sax.SAXException;

import javax.xml.parsers.ParserConfigurationException;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;

/**
 * This class contains simple test functions and utilities for testing
 */
public abstract class NetworkTest
{
    private static final long NSEC_TO_SEC = 1000000000;

    /**
     * Measure average performance of an arbitrary neural network
     * @param nTests Number of executions
     * @param layers Sizes of layers
     * @param bPrintVectors Print input and output vectors
     */
    public static void performanceTest(int nTests, int[] layers, boolean bPrintVectors) {
        long nanoTimer, createNanos;
        ArrayList<Long> execute = new ArrayList<>();

        try {
            nanoTimer = System.nanoTime();
            NetworkDescriptor netDescriptor = new NetworkDescriptor();
            for (int l = 0; l < layers.length; l++) {
                double[] biases = new double[layers[l]];

                for (int b = 0; b < biases.length; b++) {
                    biases[b] = Math.random();
                }


                if (l == 0) {
                    netDescriptor.addLayer(biases, NetworkDescriptor.BehaviourType.IDENTITY, null);
                } else {
                    double[][] connections = new double[layers[l - 1]][layers[l]];

                    for (int h = 0; h < connections.length; h++) {
                        for (int k = 0; k < connections[0].length; k++) {
                            connections[h][k] = Math.random();
                        }
                    }

                    netDescriptor.addLayer(biases, NetworkDescriptor.BehaviourType.LOGISTIC, connections);
                }
            }

            NeuralNetwork net = new NeuralNetwork(netDescriptor);

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
            for (Long anExecution : execute) {
                executeAvg += anExecution;
            }
            executeAvg /= execute.size();
            double throughput = 1.0 / (executeAvg/(1.0*NSEC_TO_SEC));
            System.out.println("Avg. throughput: " + throughput + " cycles per second");

        } catch (TopologyException te) {
            te.printStackTrace();
        }
    }

    /**
     * Execute a cycle of the neural network described in an XML file
     * @param xmlFile XML file containing the network description
     * @param input Input vector. If null, a random input vector will be used
     * @return Network output
     * @throws TopologyException  TopologyException
     * @throws ParserConfigurationException ParserConfigurationException
     * @throws SAXException SAXException
     * @throws IOException IOException
     */
    public static double[] executeNetFromXML(File xmlFile, double[] input)
            throws TopologyException, IOException, SAXException, ParserConfigurationException
    {
        NetworkDescriptor netDescriptor = NetworkXmlParser.getNetworkDescriptor(xmlFile);
        NeuralNetwork network = new NeuralNetwork(netDescriptor);
        double[] netInput;
        double[] output;

        if (input == null) {
            netInput = new double[network.getInputSize()];
            for (int i = 0; i < netInput.length; i++) {
                netInput[i] = Math.random();
            }
        } else {
            netInput = input;
        }

        output = network.execute(netInput);

        NetworkTest.printVector(netInput, "Input:");
        NetworkTest.printVector(output, "Output:");

        return output;
    }

    public static void printVector(double[] vector, String msg) {
        System.out.println(msg);
        for (double element : vector) {
            System.out.println(element);
        }
        System.out.println();
    }
}
