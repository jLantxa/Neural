package org.jlantxa.neural.behaviour;

/**
 * The logistic function compresses any input value from [-inf, inf] to [0, 1].
 * The constant parameter permits the adjustment of the slope. A logistic function with slope 0.0
 * outputs a constant 0.5. A logistic function with infinite slope behaves as the heaviside step function.
 */
public class LogisticFunction implements Behaviour
{
    private final double constant;

    public LogisticFunction(double constant) {
        this.constant = constant;
    }

    public LogisticFunction() {
        this(1.0);
    }


    @Override
    public double activation(double x) {
        return 1 / (1 + Math.exp(-x * constant));
    }

    @Override
    public double detivative(double x) {
        return activation(x) * (1 - activation(x));
    }
}