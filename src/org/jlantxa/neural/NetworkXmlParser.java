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

import org.w3c.dom.Document;
import org.w3c.dom.Element;
import org.w3c.dom.Node;
import org.w3c.dom.NodeList;
import org.xml.sax.SAXException;

import javax.xml.parsers.DocumentBuilder;
import javax.xml.parsers.DocumentBuilderFactory;
import javax.xml.parsers.ParserConfigurationException;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;

/**
 * This class provides functions to read and write network topologies from and to XML files
 */
public abstract class NetworkXmlParser
{
    private static final String LAYER_NODE_NAME = "Layer";
    private static final String CONNECTION_NODE_NAME = "Connection";

    private static final String LAYER_NAME_ATTRIBUTE = "name";
    private static final String LAYER_BEHAVIOUR_ATTRIBUTE = "behaviour";

    private static final String NEURON_ELEMENT_NAME = "Neuron";
    private static final String NODE_ELEMENT_NAME = "Node";
    private static final String WEIGHT_ELEMENT_NAME = "Weight";

    /**
     * Get a network descriptor from an XML file
     * @param xmlFile XML file containing the network description
     * @return NetworkDescriptor object which describes the network topology
     * @throws ParserConfigurationException ParserConfigurationException
     * @throws IOException IOException
     * @throws SAXException SAXException
     * @throws TopologyException TopologyException
     */
    public static NetworkDescriptor getNetworkDescriptor(File xmlFile)
            throws ParserConfigurationException, IOException, SAXException, TopologyException
    {
        NetworkDescriptor networkDescriptor = new NetworkDescriptor();
        ArrayList<double[]> layerBiases = new ArrayList<>();
        ArrayList<NetworkDescriptor.BehaviourType> behaviours = new ArrayList<>();
        ArrayList<double[][]> connectionMatrices = new ArrayList<>();

        DocumentBuilderFactory dbFactory = DocumentBuilderFactory.newInstance();
        DocumentBuilder dBuilder = dbFactory.newDocumentBuilder();
        Document doc = dBuilder.parse(xmlFile);
        doc.getDocumentElement().normalize();

        NodeList layerList = doc.getElementsByTagName(LAYER_NODE_NAME);
        NodeList connectionList = doc.getElementsByTagName(CONNECTION_NODE_NAME);

        // Parse layers
        for (int l = 0; l < layerList.getLength(); l++) {
            Node layerNode = layerList.item(l);

            if (layerNode.getNodeType() == Node.ELEMENT_NODE) {
                String name = ((Element) layerNode).getAttribute(LAYER_NAME_ATTRIBUTE);
                String behaviourString = ((Element) layerNode).getAttribute(LAYER_BEHAVIOUR_ATTRIBUTE);

                NetworkDescriptor.BehaviourType behaviour;
                switch (behaviourString) {
                    case "identity":
                        behaviour = NetworkDescriptor.BehaviourType.IDENTITY;
                        break;

                    case "logistic":
                    default:
                        behaviour = NetworkDescriptor.BehaviourType.LOGISTIC;
                        break;
                }

                NodeList neuronList = ((Element) layerNode).getElementsByTagName(NEURON_ELEMENT_NAME);

                double[] biases = new double[neuronList.getLength()];
                for (int n = 0; n < neuronList.getLength(); n++) {
                    double bias = Double.parseDouble(neuronList.item(n).getTextContent());
                    biases[n] = bias;
                }

                // Add biases and behaviour to list
                layerBiases.add(biases);
                behaviours.add(behaviour);
            }
        }

        // Parse connections
        for (int c = 0; c < connectionList.getLength(); c++) {
            Node connectionNode = connectionList.item(c);

            if (connectionNode.getNodeType() == Node.ELEMENT_NODE) {
                Element element = (Element) connectionNode;
                NodeList neuronList = element.getElementsByTagName(NODE_ELEMENT_NAME);

                ArrayList<double[]> synapsesList = new ArrayList<>();

                for (int n = 0; n < neuronList.getLength(); n++) {
                    Node neuronNode = neuronList.item(n);

                    if (neuronNode.getNodeType() == Node.ELEMENT_NODE) {
                        Element neuronElement = (Element) neuronNode;
                        NodeList weightList = neuronElement.getElementsByTagName(WEIGHT_ELEMENT_NAME);

                        double[] nodeSynapses = new double[weightList.getLength()];
                        for (int w = 0; w < weightList.getLength(); w++) {
                            double weight = Double.parseDouble(weightList.item(w).getTextContent());
                            nodeSynapses[w] = weight;
                        }

                        synapsesList.add(nodeSynapses);
                    }
                }

                // Check that every node has the same number of synapses
                int lastNodeSynapsesNum = synapsesList.get(0).length;
                for (int n = 1; n < synapsesList.size(); n++) {
                    int synapsesNum = synapsesList.get(n).length;
                    if (lastNodeSynapsesNum != synapsesNum) {
                        throw new TopologyException("Layer connection number " + c +
                                " as described in the XML does not form a rectangular matrix.");
                    }

                    lastNodeSynapsesNum = synapsesNum;
                }

                // Create matrix
                double[][] connection = new double[synapsesList.size()][synapsesList.get(0).length];
                for (int h = 0; h < synapsesList.size(); h++) {
                    double[] synapses = synapsesList.get(h);
                    System.arraycopy(synapses, 0, connection[h], 0, synapses.length);
                }

                // Add matrix to list
                connectionMatrices.add(connection);
            }
        }

        // Add layers to network descriptor
        int layerNumTotal = layerBiases.size();
        int behavioursNumTotal = behaviours.size();
        int connectionsNumTotal = connectionMatrices.size();

        if (layerNumTotal != behavioursNumTotal) {
            throw new TopologyException("Every layer (including the input layer) must define a behaviour");
        }

        if (layerNumTotal != connectionsNumTotal + 1) {
            throw new TopologyException("There must be one connection matrix for every layer after the input layer");
        }

        // Input layer has no connection matrix
        networkDescriptor.addLayer(layerBiases.get(0), behaviours.get(0), null);

        for (int l = 1; l < layerBiases.size(); l++) {
            networkDescriptor.addLayer(layerBiases.get(l), behaviours.get(l), connectionMatrices.get(l - 1));
        }

        // Return the network descriptor if reached this point
        return networkDescriptor;
    }

    /**
     * Save a NetworkDescriptor to an XML file
     * @param networkDescriptor Network descriptor
     */
    public static void writeXML(NetworkDescriptor networkDescriptor) {

    }
}
