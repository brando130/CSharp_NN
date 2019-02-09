using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CSharp_NN
{
    public class Training
    {

        public double[][][] trainingData;

        public double QuadraticLoss(NN nn)
        {

            double sum = 0;
            
            for (int i = 0; i < trainingData.Length; i++)
            {
                for (int j = 0; j < trainingData[0][0].Length; j++)
                {
                    nn.nodes[0][j] = trainingData[i][0][j];
                }
                nn.Iterate();
                for (int j =0; j < trainingData[0][1].Length; j++)
                {
                    sum += Math.Pow(nn.nodes[nn.nodes.Length - 1][j] - trainingData[i][1][j], 2);             
                }
            }

            return sum;
           // return 1 / nn.nodes[0].Length * sum;
        }

        public static double[][][] GenerateTrainingData(Random rnd)
        {
            // Generate 10k samples
            double[][][] tD = new double[10000][][];

            // For each sample
            for (int i = 0; i < 10000; i++)
            {
                // Create a 2d array (one dimension for inputs, one for outputs)
                tD[i] = new double[2][];

                // Create the inputs (single-entry array of one random number)
                tD[i][0] = new double[] { MathTools.NextDouble(rnd, -1, 1) };
                
                // Create the outputs (input / 10)
                tD[i][1] = new double[] { tD[i][0][0] / 10 };
            }
            return tD;
        }

        public static double CostOfSquaredFunction(NN nn)
        {
            // Return the first node of the last layer squared.
            nn.Iterate();
            return Math.Pow(nn.nodes[nn.nodes.Length - 1][0], 2);
        }
        public static double CostOfCosineFunction(NN nn)
        {
            nn.Iterate();
            return Math.Cos(nn.nodes[nn.nodes.Length - 1][0]);
        }

    }
}
