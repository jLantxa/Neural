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

import org.w3c.dom.*;
import org.xml.sax.SAXException;

import javax.xml.parsers.DocumentBuilder;
import javax.xml.parsers.DocumentBuilderFactory;
import javax.xml.parsers.ParserConfigurationException;
import javax.xml.transform.Transformer;
import javax.xml.transform.TransformerFactory;
import javax.xml.transform.dom.DOMSource;
import javax.xml.transform.stream.StreamResult;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;

/**
 * This class provides functions to read and write network topologies from and to XML files
 */
public abstract class NetworkXmlParser
{
    private static final String NEURAL_NETWORK_ROOT_ELEMENT = "NeuralNetwork";

    private static final String LAYER_NODE_NAME = "Layer";
    private static final String LAYER_NAME_ATTRIBUTE = "name";
    private static final String LAYER_BEHAVIOUR_ATTRIBUTE = "behaviour";
    private static final String NEURON_ELEMENT_NAME = "Neuron";

    private static final String CONNECTION_NODE_NAME = "Connection";
    private static final String NODE_ELEMENT_NAME = "Node";
    private static final String WEIGHT_ELEMENT_NAME = "Weight";

    /**
     * Get a network descriptor from an XML file
     *
     * @param xmlFile XML file containing the network description
     * @return NetworkDescriptor object which describes the network topology
     * @throws ParserConfigurationException ParserConfigurationException
     * @throws IOException                  IOException
     * @throws SAXException                 SAXException
     * @throws TopologyException            TopologyException
     */
    public static NetworkDescriptor getNetworkDescriptor(File xmlFile)
            throws ParserConfigurationException, IOException, SAXException, TopologyException {
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

                NetworkDescriptor.BehaviourType behaviour = getBehaviourTypeFromAttribute(behaviourString);

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
     * Get a NeuralNetwork instance from an XML file
     *
     * @param xmlFile XML file containing the network description
     * @return NetworkDescriptor object which describes the network topology
     * @throws ParserConfigurationException ParserConfigurationException
     * @throws IOException                  IOException
     * @throws SAXException                 SAXException
     * @throws TopologyException            TopologyException
     */
    public static NeuralNetwork getNetwork(File xmlFile)
            throws ParserConfigurationException, IOException, SAXException, TopologyException {
        return new NeuralNetwork(getNetworkDescriptor(xmlFile));
    }

    /**
     * Save a NetworkDescriptor to an XML file
     *
     * @param networkDescriptor Network descriptor
     * @param xmlFile Output file
     */
    public static void writeXML(NetworkDescriptor networkDescriptor, File xmlFile) {
        try {
            DocumentBuilderFactory dbFactory = DocumentBuilderFactory.newInstance();
            DocumentBuilder dBuilder = dbFactory.newDocumentBuilder();
            Document doc = dBuilder.newDocument();

            // NeuralNetwork element
            Element neuralNetworkElement = doc.createElement(NEURAL_NETWORK_ROOT_ELEMENT);
            doc.appendChild(neuralNetworkElement);

            // Layers
            for (NetworkDescriptor.LayerDescriptor layerDescriptor : networkDescriptor.getLayerDescriptors()) {
                Element layerElement = doc.createElement(LAYER_NODE_NAME);
                neuralNetworkElement.appendChild(layerElement);

                NetworkDescriptor.BehaviourType behaviourType = layerDescriptor.behaviourType;
                Attr attr = doc.createAttribute("behaviour");
                attr.setValue(getBehaviourAttributeFromType(behaviourType));
                layerElement.setAttributeNode(attr);

                for (double bias : layerDescriptor.biases) {
                    Element neuronElement = doc.createElement(NEURON_ELEMENT_NAME);
                    neuronElement.appendChild(doc.createTextNode(String.valueOf(bias)));
                    layerElement.appendChild(neuronElement);
                }
            }

            // Connections
            for (double[][] connectionMatrix : networkDescriptor.getConnectionDescriptors()) {
                Element connectionElement = doc.createElement(CONNECTION_NODE_NAME);
                neuralNetworkElement.appendChild(connectionElement);

                // Nodes
                for (double[] synapseList : connectionMatrix) {
                    Element nodeElement = doc.createElement(NODE_ELEMENT_NAME);
                    connectionElement.appendChild(nodeElement);

                    // Weights
                    for (double weight : synapseList) {
                        Element weightElement = doc.createElement(WEIGHT_ELEMENT_NAME);
                        weightElement.appendChild(doc.createTextNode(String.valueOf(weight)));
                        nodeElement.appendChild(weightElement);
                    }
                }
            }

            // Write the content into xml file
            TransformerFactory transformerFactory = TransformerFactory.newInstance();
            Transformer transformer = transformerFactory.newTransformer();
            DOMSource source = new DOMSource(doc);
            StreamResult result = new StreamResult(xmlFile);
            transformer.transform(source, result);

            // Output to console for testing
            StreamResult consoleResult = new StreamResult(System.out);
            transformer.transform(source, consoleResult);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private static String getBehaviourAttributeFromType(NetworkDescriptor.BehaviourType type) {
        switch (type) {
            case IDENTITY:
                return "identity";

            case LOGISTIC:
            default:
                return "logistic";
        }
    }

    private static NetworkDescriptor.BehaviourType getBehaviourTypeFromAttribute(String attribute) {
        switch (attribute) {
            case "identity":
                return NetworkDescriptor.BehaviourType.IDENTITY;

            case "logistic":
            default:
                return NetworkDescriptor.BehaviourType.LOGISTIC;
        }
    }
}
