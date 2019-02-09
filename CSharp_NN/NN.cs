using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CSharp_NN
{
    public class NN
    {

        // Feedforward Neural Network
        // (C) 2019 Brandon Anderson  brando.slc@gmail.com

        public enum ActivationFunction { None, ReLU, Sigmoid, Tanh, Step, Softmax }

        // Nodes ( w/ biases, weights and activation functions)
        public double[][] nodes;
        public double[][] biases;
        public double[][][] weights;
        public ActivationFunction[] hiddenLayerFunctions;
        public ActivationFunction outputLayerFunction;

        public double cost;

        public NN(int[] nodeCount, ActivationFunction[] hiddenLayerActivationFunction, ActivationFunction outputLayerActivationFunction)
        {

            nodes = new double[nodeCount.Length][];
            biases = new double[nodeCount.Length][];
            weights = new double[nodeCount.Length][][];
            hiddenLayerFunctions = hiddenLayerActivationFunction;
            outputLayerFunction = outputLayerActivationFunction;

            // Create nodes (with biases and activation functions) for every layer
            for (int layer = 0; layer < nodeCount.Length; layer++)
            {
                nodes[layer] = new double[nodeCount[layer]];
                biases[layer] = new double[nodeCount[layer]];
                for (int i = 0; i < nodeCount[layer]; i++)
                {
                    double node = 0;
                    double bias = 0;

                    nodes[layer][i] = node;
                    biases[layer][i] = bias;     
                }
            }
            // For every layer except the last layer
            for (int layer = 0; layer < nodeCount.Length - 1; layer++)
            {
                weights[layer] = new double[nodeCount[layer]][];
                // For every node in the layer
                for (int node = 0; node < nodeCount[layer]; node++)
                {
                    weights[layer][node] = new double[nodeCount[layer + 1]];
                    // For every node in the next layer
                    for (int w = 0; w < nodeCount[layer + 1]; w++)
                    {
                        // Create a weight
                        weights[layer][node][w] = 0;
                    }
                }
            }
        }

        public bool Iterate()
        {
            try
            {        
                // For every layer except the first layer
                for (int layer = 1; layer < nodes.Length; layer++)
                {
                    // For each node in the layer
                    for (int node = 0; node < nodes[layer].Length; node++)
                    {
                        // Reset the value
                        nodes[layer][node] = 0;

                        // For each node in the previous layer..
                        // Sum the product of every node in that layer and the weight that connects it to the current node
                        for (int nodeInPreviousLayer = 0; nodeInPreviousLayer < nodes[layer - 1].Length; nodeInPreviousLayer++)
                        {
                            nodes[layer][node] += nodes[layer - 1][nodeInPreviousLayer] * weights[layer - 1][nodeInPreviousLayer][node];
                        }

                        // Add the bias
                        nodes[layer][node] += biases[layer][node];               

                    }
                    // Activate
                    if (layer < nodes.Length - 1)
                    {
                        nodes[layer] = Activate(nodes[layer], hiddenLayerFunctions[layer - 1]);
                    }
                    else
                    {
                        nodes[layer] = Activate(nodes[layer], outputLayerFunction);
                    }
                 
                }

                return true;
            }
            catch (Exception ex) { Debug.WriteLine(ex.Message); return false; }
        }

        public static double[] Activate(double[] values, ActivationFunction function)
        {
            try
            {              
                if (function == ActivationFunction.ReLU)
                {
                    return MathTools.ReLU(values);
                }
                else if (function == ActivationFunction.Sigmoid)
                {
                    return MathTools.Sigmoid(values);             
                }
                else if (function == ActivationFunction.Softmax)
                {
                    return MathTools.Softmax(values);
                }
                else { return values; }
            }
            catch (Exception ex)
            {
                Debug.WriteLine(ex.Message);
                return values;
            }
        }

       
    }
}
