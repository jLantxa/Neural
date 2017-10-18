package org.jlantxa.neural.behaviour;

/**
 * The identity function outputs exactly its input when evaluated. Its derivative is therefore 0.
 */
public class IdentityFunction implements Behaviour
{
    public IdentityFunction() {

    }


    @Override
    public double activation(double x) {
        return x;
    }

    @Override
    public double detivative(double x) {
        return 0.0;
    }
}