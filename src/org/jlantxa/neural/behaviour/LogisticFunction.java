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
    public double derivative(double x) {
        return activation(x) * (1 - activation(x));
    }
}