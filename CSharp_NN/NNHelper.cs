using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CSharp_NN
{
    public static class NNHelper
    {
        public static NN CopyNN(NN original)
        {
            NN copy;

            int[] nodeCount = new int[original.nodes.Length];
            for (int l = 0; l < original.nodes.Length; l++)
            {
                nodeCount[l] = original.nodes[l].Length;
            }

            copy = new NN(nodeCount, original.hiddenLayerFunctions, original.outputLayerFunction);

            for (int l = 0; l < original.nodes.Length; l++)
            {
                for (int n = 0; n < original.nodes[l].Length; n++)
                {
                    copy.nodes[l][n] = original.nodes[l][n];
                    copy.biases[l][n] = original.biases[l][n];

                    if (l < original.nodes.Length - 1)
                    {
                        for (int w = 0; w < original.nodes[l + 1].Length; w++)
                        {
                            copy.weights[l][n][w] = original.weights[l][n][w];
                        }
                    }
                }

            }
            return copy;
        }

        public static bool RandomizeWeights(NN nn, Random rnd, double minValue, double maxValue)
        {
            for (int layer = 0; layer < nn.nodes.Length - 1; layer++)
            {
                for (int node = 0; node < nn.nodes[layer].Length; node++)
                {
                    for (int weight = 0; weight < nn.nodes[layer + 1].Length; weight++)
                    {
                        nn.weights[layer][node][weight] = MathTools.NextDouble(rnd, minValue, maxValue);
                    }
                }
            }
            return true;
        }
        public static void PrintNetwork(NN nn)
        {
            Debug.Write("Inputs: ");
            for (int l = 0; l < nn.nodes.Length; l++)
            {
                if (l > 0 && l < nn.nodes.Length - 1)
                {
                    Debug.Write("Hidden Layer " + l + " Values: ");
                }
                if (l == nn.nodes.Length - 1)
                {
                    Debug.Write("Outputs: ");
                }
                for (int n = 0; n < nn.nodes[l].Length; n++)
                {
                    Debug.Write(nn.nodes[l][n] + ", ");
                }
                Debug.Write(Environment.NewLine);
            }
            for (int layer = 0; layer < nn.nodes.Length - 1; layer++)
            {
                for (int node = 0; node < nn.nodes[layer].Length; node++)
                {
                    for (int weight = 0; weight < nn.nodes[layer + 1].Length; weight++)
                    {
                       Debug.WriteLine("Weights for layer " + layer + " node " + node + " weight " + nn.weights[layer][node][weight]);
                    }
                }
            }
        }

        public static void MutateWeights(NN nn, Random rnd, double minMutation, double maxMutation)
        {
            for (int l = 0; l < nn.nodes.Length - 1; l++)
            {
                for (int n = 0; n < nn.nodes[l].Length; n++)
                {
                    for (int w = 0; w < nn.nodes[l + 1].Length; w++)
                    {
                        nn.weights[l][n][w] *= MathTools.NextDouble(rnd, minMutation, maxMutation);
                    }
                }
            }
        }

    }
}
