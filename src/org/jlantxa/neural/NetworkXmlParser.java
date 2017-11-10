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

import javax.xml.parsers.DocumentBuilder;
import javax.xml.parsers.DocumentBuilderFactory;
import java.io.File;

public class NetworkXmlParser
{
    private NetworkXmlParser() {

    }

    public static NetworkDescriptor parseXML(File xmlFile) {
        NetworkDescriptor networkDescriptor = new NetworkDescriptor();

        try {
            DocumentBuilderFactory dbFactory = DocumentBuilderFactory.newInstance();
            DocumentBuilder dBuilder = dbFactory.newDocumentBuilder();
            Document doc = dBuilder.parse(xmlFile);
            doc.getDocumentElement().normalize();
            System.out.println("Root element :" + doc.getDocumentElement().getNodeName());
            System.out.println("----------------------------");

            NodeList layerList = doc.getElementsByTagName("Layer");
            NodeList connectionList = doc.getElementsByTagName("Connection");

            // Parse layers
            for (int l = 0; l < layerList.getLength(); l++) {
                Node layerNode = layerList.item(l);
                System.out.println("\nCurrent node: " + layerNode.getNodeName());

                if (layerNode.getNodeType() == Node.ELEMENT_NODE) {
                    String name = ((Element)layerNode).getAttribute("name");
                    System.out.println("Layer name: " + name);
                    String behaviourString = ((Element)layerNode).getAttribute("behaviour");
                    System.out.println("Behaviour name: " + behaviourString);

                    NodeList neuronList = ((Element)layerNode).getElementsByTagName("Neuron");

                    for (int n = 0; n < neuronList.getLength(); n++) {
                        double bias = Double.parseDouble(neuronList.item(n).getTextContent());
                        System.out.println("Bias: " + bias);
                    }
                }
            }

            // Parse connections
            for (int c = 0; c < connectionList.getLength(); c++) {
                Node connectionNode = connectionList.item(c);
                System.out.println("\nCurrent node: " + connectionNode.getNodeName());

                if (connectionNode.getNodeType() == Node.ELEMENT_NODE) {
                    Element element = (Element) connectionNode;
                    NodeList neuronList = element.getElementsByTagName("Node");

                    for (int n = 0; n < neuronList.getLength(); n++) {
                        Node neuronNode = neuronList.item(n);
                        System.out.println("Neuron node: " + n);

                        if (neuronNode.getNodeType() == Node.ELEMENT_NODE) {
                            Element neuronElement = (Element) neuronNode;
                            NodeList weightList = neuronElement.getElementsByTagName("Weight");

                            for (int w = 0; w < weightList.getLength(); w++) {
                                double weight = Double.parseDouble(weightList.item(w).getTextContent());
                                System.out.println("Weight: " + weight);
                            }
                        }
                    }
                }
            }
        } catch (Exception e) {
            e.printStackTrace();
        }

        return networkDescriptor;
    }
}
