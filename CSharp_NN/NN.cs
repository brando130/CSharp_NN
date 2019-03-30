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
        // (C) 2019 Brandon Anderson brando.slc@gmail.com (MIT License)

        public enum Activation { None, LeakyReLU, ReLU, Sigmoid, Softmax, Step, Tanh }
        public Activation[] hLA;
        public Activation oLA;
        public float[][] nodes, biases, nets;
        public float[][][] weights;
        public float[][][] updateToWeights;

        public float[] errors;
        public float cost;
        public float[][] gradientOfTotalErrorWithRespectToOutput;
        public float[][] gradientOfTotalErrorWithRespectToNet;
        public float[][] gradientOfOutputWithRespectToNet;

        public NN(int[] nodeCounts, Activation[] hiddenLayerActivations, Activation outputLayerActivation)
        {
            hLA = hiddenLayerActivations;
            oLA = outputLayerActivation;

            nodes = new float[nodeCounts.Length][];
            errors = new float[nodeCounts[nodeCounts.Length - 1]];
            nets = new float[nodeCounts.Length - 1][];
            biases = new float[nodeCounts.Length - 1][];
            gradientOfTotalErrorWithRespectToOutput = new float[nodeCounts.Length][];
            gradientOfTotalErrorWithRespectToNet = new float[nodeCounts.Length][];
            gradientOfOutputWithRespectToNet = new float[nodeCounts.Length][];
            weights = new float[nodeCounts.Length - 1][][];
            updateToWeights = new float[nodeCounts.Length - 1][][];

            for (int layer = 0; layer < nodeCounts.Length; layer++)
            {
                nodes[layer] = new float[nodeCounts[layer]];
                if (layer > 0)
                {
                    biases[layer - 1] = new float[nodeCounts[layer]];
                    nets[layer - 1] = new float[nodeCounts[layer]];
                }
                if (layer < nodeCounts.Length - 1)
                {
                    weights[layer] = new float[nodeCounts[layer + 1]][];
                    updateToWeights[layer] = new float[nodeCounts[layer + 1]][];
                }

                for (int node = 0; node < nodeCounts[layer]; node++)
                {
                    nodes[layer][node] = 0f;
                    if (layer > 0)
                    {
                        biases[layer - 1][node] = 0f;
                        nets[layer - 1][node] = 0f;
                    }
                    if (layer < nodeCounts.Length - 1)
                    {
                        weights[layer][node] = new float[nodeCounts[layer + 1]];
                        updateToWeights[layer][node] = new float[nodeCounts[layer + 1]];
                        for (int nodeInNextLayer = 0; nodeInNextLayer < nodeCounts[layer + 1]; nodeInNextLayer++)
                        {
                            weights[layer][node][nodeInNextLayer] = 0f;
                            updateToWeights[layer][node][nodeInNextLayer] = 0f;
                        }
                    }
                }
            }
        }

        public bool Iterate()
        {
            for (int layer = 1; layer < nodes.Length; layer++)
            {
                for (int node = 0; node < nodes[layer].Length; node++)
                {
                    float sum = 0f;
                    for (int nodeInPreviousLayer = 0; nodeInPreviousLayer < nodes[layer - 1].Length; nodeInPreviousLayer++)
                    {
                        sum += nodes[layer - 1][nodeInPreviousLayer] * weights[layer - 1][nodeInPreviousLayer][node];
                    }
                    nodes[layer][node] = sum + biases[layer - 1][node];
                    nets[layer - 1][node] = nodes[layer][node];
                }
                if (layer < nodes.Length - 1)
                    nodes[layer] = Activate(nodes[layer], hLA[layer - 1]);
                else
                    nodes[layer] = Activate(nodes[layer], oLA);
            }
            return true;
        }

        public bool Backpropagate(float[] target, float learningRate)
        {

            cost = 0f;

            for (int i = 0; i < nodes[nodes.Length - 1].Length; i++)
            {
                errors[i] = 0.5f * (float)Math.Pow(target[i] - nodes[nodes.Length - 1][i], 2f);
                cost += errors[i];
            }

            for (int layer = nodes.Length - 1; layer > 0; layer--)
            {
                // Output layer
                if (layer == nodes.Length - 1)
                {
                    gradientOfTotalErrorWithRespectToOutput[layer - 1] = new float[target.Length];
                    gradientOfTotalErrorWithRespectToNet[layer - 1] = new float[target.Length];
                    gradientOfOutputWithRespectToNet[layer - 1] = new float[target.Length];

                    // For each output node
                    for (int i = 0; i < nodes[nodes.Length - 1].Length; i++)
                    {
                        gradientOfTotalErrorWithRespectToOutput[layer - 1][i] = (target[i] - nodes[nodes.Length - 1][i]) * -1f;
                        gradientOfOutputWithRespectToNet[layer - 1][i] = nodes[nodes.Length - 1][i] * (1 - nodes[nodes.Length - 1][i]);
                        for (int j = 0; j < weights[layer - 1][i].Length; j++)
                        {
                            updateToWeights[layer - 1][j][i] = weights[layer - 1][j][i] - learningRate * (gradientOfTotalErrorWithRespectToOutput[layer - 1][i] * gradientOfOutputWithRespectToNet[layer - 1][i] * nodes[layer - 1][i]);
                        }
                    }
                }

                // Hidden Layers
                else
                {
                    gradientOfTotalErrorWithRespectToNet[layer - 1] = new float[nodes[layer].Length];
                    gradientOfTotalErrorWithRespectToOutput[layer - 1] = new float[nodes[layer].Length];
                    gradientOfOutputWithRespectToNet[layer - 1] = new float[nodes[layer].Length];

                    for (int i = 0; i < nodes[layer].Length; i++)
                    {
                        gradientOfTotalErrorWithRespectToOutput[layer - 1][i] = 0f;
                        float[] gradientOfOutputInNextLayerWithRespectToOutput = new float[nodes[layer + 1].Length];
                        for (int j = 0; j < nodes[layer + 1].Length; j++)
                        {
                            gradientOfTotalErrorWithRespectToNet[layer - 1][j] = gradientOfTotalErrorWithRespectToOutput[layer][j] * gradientOfOutputWithRespectToNet[layer][j];
                            gradientOfOutputInNextLayerWithRespectToOutput[j] = gradientOfTotalErrorWithRespectToNet[layer - 1][j] * weights[layer][i][j];
                            gradientOfTotalErrorWithRespectToOutput[layer - 1][i] += gradientOfOutputInNextLayerWithRespectToOutput[j];
                        }
                    }
                    for (int i = 0; i < nodes[layer].Length; i++)
                    {
                        gradientOfOutputWithRespectToNet[layer - 1][i] = nodes[layer][i] * (1 - nodes[layer][i]);
                        for (int j = 0; j < weights[layer - 1][i].Length; j++)
                        {
                            updateToWeights[layer - 1][j][i] = weights[layer - 1][j][i] - learningRate * (gradientOfTotalErrorWithRespectToOutput[layer - 1][i] * gradientOfOutputWithRespectToNet[layer - 1][i] * nodes[layer - 1][i]);
                        }
                    }
                }
            }

            // Update weights
            for (int l = 0; l < nodes.Length - 1; l++)
            {
                for (int n = 0; n < nodes[l].Length; n++)
                {
                    for (int w = 0; w < nodes[l + 1].Length; w++)
                    {
                        weights[l][n][w] = updateToWeights[l][n][w];
                    }
                }
            }

            return true;
        }

        public static float[] Activate(float[] vector, Activation function)
        {
            if (function == Activation.ReLU)
                return ReLU(vector);
            else if (function == Activation.Sigmoid)
                return Sigmoid(vector);
            else if (function == Activation.Softmax)
                return Softmax(vector);
            else
                return vector;
        }


        public static double NextDouble(Random rand, double max)
        {
            return rand.NextDouble() * max;
        }

        public static double NextDouble(Random rand, double min, double max)
        {
            return min + (rand.NextDouble() * (max - min));
        }

        public static float[] Sigmoid(float[] vector)
        {
            float[] rVector = new float[vector.Length];
            for (int i = 0; i < vector.Length; i++)
            {
                float eToNumber = (float)Math.Exp(vector[i]);
                rVector[i] = eToNumber / (1 + eToNumber);
            }
            return rVector;
        }

        public static float[] ReLU(float[] vector)
        {
            float[] rVector = new float[vector.Length];
            for (int i = 0; i < vector.Length; i++)
            {
                if (vector[i] > 0)
                    rVector[i] = vector[i];
                else
                    rVector[i] = 0f;
            }
            return rVector;
        }

        public static float[] Softmax(float[] vector)
        {

            float sum = 0;

            // Calculate the exponetiated vector
            float[] eVector = new float[vector.Length];
            float[] rVector = new float[vector.Length];

            for (int i = 0; i < vector.Length; i++)
            {
                float eNum = (float)Math.Pow(Math.E, vector[i]);
                eVector[i] = eNum;
                sum += eNum;
            }

            // Calculate the softmax for each number in the vector
            for (int i = 0; i < vector.Length; i++)
            {
                rVector[i] = eVector[i] / sum;
            }

            return rVector;
        }
    }
}
