using System;
using System.Diagnostics;
using System.Text;
using System.Windows.Forms;
using System.Numerics;
using System.Collections.Generic;
using Alea;
using System.Linq;
using Alea.Parallel;

namespace CSharp_NN
{

    public partial class Form1 : Form
    {
        public GA genAlg;
        public Random rnd = new Random();
        public double[][][] trainingData;

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
            GA ga = new GA(100, session.QuadraticLoss, 1, 1, NN.Activation.Sigmoid, NN.Activation.Sigmoid, 1, 3, 2, 10, -1, 1, -1, 1, 5, 0.9, 1.1);
            Debug.WriteLine("Initialized");
            while (true)
            {
                for (int i = 0; i < 10; i++)
                {
                    bool success = ga.Iterate();
                }
                Debug.Write("cost: " + ga.population[0].cost);
                Debug.Write(" y: " + session.trainingData[0][1][0]);
                ga.population[0].nodes[0][0] = (float)session.trainingData[0][0][0];
                ga.population[0].Iterate();
                Debug.Write(" a: " + ga.population[0].nodes[ga.population[0].nodes.Length - 1][0]);
                Debug.Write(Environment.NewLine);
                double x = NN.NextDouble(rnd, 0, 1);
                Debug.Write("test: x: " + x);
                ga.population[0].nodes[0][0] = (float)x;
                ga.population[0].Iterate();
                Debug.Write(" a: " + ga.population[0].nodes[ga.population[0].nodes.Length - 1][0]);
                Debug.WriteLine(Environment.NewLine);
            }
        }

        private void button1_Click(object sender, EventArgs e)
        {
            Training session = new Training();
            session.trainingData = Training.LoadTrainingData(Environment.CurrentDirectory + @"\training-articles.csv");

            GA ga = new GA(250, session.QuadraticLoss, 128, 4, NN.Activation.Sigmoid, NN.Activation.Sigmoid, 1, 4, 16, 64, -1, 1, -1, 1, 10, 0.8, 1.2);

            while (true)
            {
                ga.Iterate();
                Debug.WriteLine("Iterated. Cost: " + ga.population[0].cost);


                string test = "Some progress made on Medicaid expansion governor says";
                double[] encodedTest = Training.EncodeString(test);
                for (int i = 0; i < encodedTest.Length; i++)
                {
                    ga.population[0].nodes[0][i] = (float)encodedTest[i];
                }
                ga.population[0].Iterate();
                Debug.WriteLine("test string predicted category 1: " + ga.population[0].nodes[ga.population[0].nodes.Length - 1][0] + ga.population[0].nodes[ga.population[0].nodes.Length - 1][1] + ga.population[0].nodes[ga.population[0].nodes.Length - 1][2] + ga.population[0].nodes[ga.population[0].nodes.Length - 1][3]);

                string test2 = "Top Banana Chiquita And Fyffe Announce Merger";
                double[] encodedTest2 = Training.EncodeString(test2);
                for (int i = 0; i < encodedTest2.Length; i++)
                {
                    ga.population[0].nodes[0][i] = (float)encodedTest2[i];
                }
                ga.population[0].Iterate();
                Debug.WriteLine("test string predicted category 2: " + ga.population[0].nodes[ga.population[0].nodes.Length - 1][0] + ga.population[0].nodes[ga.population[0].nodes.Length - 1][1] + ga.population[0].nodes[ga.population[0].nodes.Length - 1][2] + ga.population[0].nodes[ga.population[0].nodes.Length - 1][3]);


                string test3 = "The Flash TV series casts Godzilla actor";
                double[] encodedTest3 = Training.EncodeString(test3);
                for (int i = 0; i < encodedTest3.Length; i++)
                {
                    ga.population[0].nodes[0][i] = (float)encodedTest3[i];
                }
                ga.population[0].Iterate();
                Debug.WriteLine("test string predicted category 3: " + ga.population[0].nodes[ga.population[0].nodes.Length - 1][0] + ga.population[0].nodes[ga.population[0].nodes.Length - 1][1] + ga.population[0].nodes[ga.population[0].nodes.Length - 1][2] + ga.population[0].nodes[ga.population[0].nodes.Length - 1][3]);


            }

        }
    }
}
