package org.jlantxa.neural.test;

import org.jlantxa.neural.NetworkDescriptor;
import org.jlantxa.neural.NetworkXmlParser;

import java.io.File;

public class NetworkXmlParserTest
{
    public static void main(String[] args) {
       NetworkDescriptor networkDescriptor = NetworkXmlParser.parseXML(new File("res/NetExample.xml"));
    }
}
