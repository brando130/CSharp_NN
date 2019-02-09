using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CSharp_NN
{
    public static class MathTools
    {

        public static double NextDouble(Random rand, double max)
        {
            return rand.NextDouble() * max;
        }

        public static double NextDouble(Random rand, double min, double max)
        {
            return min + (rand.NextDouble() * (max - min));
        }

        public static double[] Sigmoid(double[] vector)
        {
            double[] rVector = new double[vector.Length];

            for (int i = 0; i < vector.Length; i++)
            {
                double eToNumber = Math.Pow(Math.E, vector[i]);
                rVector[i] = eToNumber / (1 + eToNumber);
            }

            return rVector;
        }
        public static double[] Softmax(double[] vector)
        {

            double sum = 0;

            // Calculate the exponetiated vector
            double[] eVector = new double[vector.Length];
            double[] rVector = new double[vector.Length];

            for (int i = 0; i < vector.Length; i++)
            {
                double eNum = Math.Pow(Math.E, vector[i]);
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

        public static double[] ReLU(double[] vector)
        {
            double[] returnValues = new double[vector.Length];
            for (int i = 0; i < vector.Length; i++)
            {
                if (vector[i] < 0) { returnValues[i] = 0; } else { returnValues[i] = vector[i]; }
            }
            return returnValues;
        }

        public static double QuadraticCost(double[] x, double[] y, double[] a)
        {
            double n = x.Length;
            double sum = 0;
            for (int i = 0; i < x.Length; i++)
            {
                sum += Math.Pow(y[i] - a[i], 2);            
            }
            // return 1 /  x.Length * sum;
            return sum;
        }

    }
}
