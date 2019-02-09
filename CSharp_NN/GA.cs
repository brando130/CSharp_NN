using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CSharp_NN
{
    public class GA
    {

        // Simple genetic algorithm for training DNNs with reinforcement
        // (C) 2019 Brandon Anderson  brando.slc@gmail.com
        // Implementation of ideas published in Such, et. al 2017 (Uber AI Labs)
        // https://arxiv.org/abs/1712.06567

        public NN[] population;
        private int g = 0;
        private readonly Func<NN, double> cost;
        private readonly Random rnd = new Random();
        private readonly int elites;
        private readonly double minMutation, maxMutation;

        public GA(int populationSize, Func<NN, double> costFunction, int inputCount, int outputCount, NN.ActivationFunction activationFunctionHiddenLayers, NN.ActivationFunction activationFunctionOutputLayer, int minHiddenLayers, int maxHiddenLayers, int minHiddenNodes, int maxHiddenNodes, double weightMinValue, double weightMaxValue, double biasMinValue, double biasMaxValue, int elitesToCopy, double minMutationPercentage, double maxMutationPercentage)
        {

            population = new NN[populationSize];
            cost = costFunction;
            elites = elitesToCopy;
            minMutation = minMutationPercentage;
            maxMutation = maxMutationPercentage;

            // For each new unit to be created
            for (int i = 0; i < populationSize; i++)
            {
                // Specify input, hidden, and output layer node counts
                int[] nodeCounts = new int[2 + rnd.Next(minHiddenLayers, maxHiddenLayers + 1)];
                nodeCounts[0] = inputCount;
                for (int j = 1; j < nodeCounts.Length - 1; j++)
                {
                    nodeCounts[j] = rnd.Next(minHiddenNodes, maxHiddenNodes + 1);
                }
                nodeCounts[nodeCounts.Length - 1] = outputCount;

                // Create the neural network
                NN.ActivationFunction[] functions = new NN.ActivationFunction[nodeCounts.Length - 1];
                for (int j = 0; j < nodeCounts.Length - 2; j++)
                {
                    functions[j] = activationFunctionHiddenLayers;
                }
                population[i] = new NN(nodeCounts, functions, activationFunctionOutputLayer);

                // Initialize weights
                bool success = NNHelper.RandomizeWeights(population[i], rnd, weightMinValue, weightMaxValue);
            }
        }

        public bool Iterate()
        {
            try
            {
          
                // Initial cost and sort
                if (g == 0)
                {
                    foreach (NN nn in population)
                    {
                        nn.cost = cost(nn);
                    }
                    population = population.OrderBy(p => p.cost).ToArray();
                }

                // Copy non-elite networks and mutate them. 
                for (int i = 1; i < population.Length; i++)
                {          
                    if (i >= elites)
                    {              
                        population[i] = NNHelper.CopyNN(population[rnd.Next(0, elites - 1)]);
                        NNHelper.MutateWeights(population[i], rnd, minMutation, maxMutation);
                    }
                }

                // Cost and sort
                foreach (NN nn in population)
                {
                    nn.cost = cost(nn);
                }
                population = population.OrderBy(p => p.cost).ToArray();

                // Increment generation
                g++;

                return true;
            }
            catch (Exception ex)
            {
                Debug.WriteLine(ex.Message);
                return false;
            }
        }
    }
}
