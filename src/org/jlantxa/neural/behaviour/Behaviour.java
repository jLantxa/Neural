package org.jlantxa.neural.behaviour;

/**
 * The Behaviour interface models the activation function of a neuron
 * Activation functions must implement this interface so they can be instantiated regardless of
 * their class.
 */
public interface Behaviour
{
    /**
     * Evaluate the function at value x
     * @param x input value to evaluate the function
     * @return output of the function for value x
     */
    double activation(double x);

    /**
     * Calculate the derivative of the function at input value x
     * @param x input value to calculate the derivative of the function
     * @return derivative of the function at input value x
     */
    double detivative(double x);
}
