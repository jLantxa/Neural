package org.jlantxa.neural.test;

import org.jlantxa.neural.NeuralNetwork;
import org.jlantxa.neural.behaviour.IdentityFunction;
import org.jlantxa.neural.behaviour.LogisticFunction;

public class NetTest
{
    public static void main(String[] args) {
        NeuralNetwork net = new NeuralNetwork();

        net.addLayer(new double[] {0.0, 0.0}, new IdentityFunction(), null);
        net.addLayer(new double[] {0.0, 0.0, 0.0}, new LogisticFunction(), new double[][] {{1.0, 1.0, 1.0}, {1.0, 1.0, 1.0}});
        net.addLayer(new double[] {0.0, 0.0, 0.0}, new LogisticFunction(), new double[][] {{1.0}, {1.0}, {1.0}});

        double[] input = {1.0, -1.0};
        double[] output = net.execute(input);

        System.out.println("Net size: " + net.getSize());

        System.out.println("Net input:");
        for (double in : input) {
            System.out.println(in);
        }
        System.out.println();

        System.out.println("Net output:");
        for (double out : output) {
            System.out.println(out);
        }
        System.out.println();
    }
}
