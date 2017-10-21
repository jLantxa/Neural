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
    double derivative(double x);
}
