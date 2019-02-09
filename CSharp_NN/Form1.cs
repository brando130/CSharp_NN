using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Diagnostics;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace CSharp_NN
{

    public partial class Form1 : Form
    {
        public GA genAlg;
        public Random rnd = new Random();

        public Form1()
        {
            InitializeComponent();
        }

        private void Form1_Load(object sender, EventArgs e)
        {

        }



        private void button4_Click(object sender, EventArgs e)
        {
            Training session = new Training();
            Random rnd = new Random();
            session.trainingData = Training.GenerateTrainingData(rnd);
            GA ga = new GA(100, session.QuadraticLoss, 1, 1, NN.ActivationFunction.Sigmoid, NN.ActivationFunction.Sigmoid, 1, 3, 2, 10, -1, 1, -1, 1, 5, 0.9, 1.1);
            Debug.WriteLine("Initialized");
            while (true)
            {
                for (int i = 0; i < 10; i++)
                {
                    bool success = ga.Iterate();
                }
                Debug.Write("cost: " + ga.population[0].cost);
                Debug.Write(" y: " + session.trainingData[0][1][0]);
                ga.population[0].nodes[0][0] = session.trainingData[0][0][0];
                ga.population[0].Iterate();
                Debug.Write(" a: " + ga.population[0].nodes[ga.population[0].nodes.Length - 1][0]);
                Debug.Write(Environment.NewLine);
                double x = MathTools.NextDouble(rnd, 0, 1);
                Debug.Write("test: x: " + x);
                ga.population[0].nodes[0][0] = x;
                ga.population[0].Iterate();
                Debug.Write(" a: " + ga.population[0].nodes[ga.population[0].nodes.Length - 1][0]);
                Debug.WriteLine(Environment.NewLine);
            }
        }
    }
}
