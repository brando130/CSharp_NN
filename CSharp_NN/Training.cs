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

          //  Debug.WriteLine(g);

            double sum = 0;
            
            for (int i = 0; i < trainingData.Length; i++)
            {
                for (int j = 0; j < trainingData[0][0].Length; j++)
                {
                 //   Debug.WriteLine("Setting input node to: " + trainingData[i][0][j]);
                    nn.nodes[0][j] = trainingData[i][0][j];
                }
                nn.Iterate();
               // NNHelper.PrintNetwork(nn);
                for (int j =0; j < trainingData[0][1].Length; j++)
                {
                    
              //      Debug.WriteLine("Output length:" + trainingData[0][1].Length);
               //     Debug.WriteLine("j " + j);
               //     Debug.WriteLine("Expected value: " + trainingData[i][1][j]);
               //     Debug.WriteLine("Output value: " + nn.nodes[nn.nodes.Length - 1][j]);
               //     Debug.WriteLine("Old Sum: " + sum);
                    sum += Math.Pow(nn.nodes[nn.nodes.Length - 1][j] - trainingData[i][1][j], 2);
               //     Debug.WriteLine("New Sum: " + sum);
                   
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
                
                // Create the outputs (input % 2)
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
        public static double CostOfSingleVariableQuadraticLoss(NN nn)
        {
            nn.Iterate();
            double x = nn.nodes[nn.nodes.Length - 1][0];
            double t = 0.53453;
            return Math.Pow(t - x, 2);
        }
    }
}
