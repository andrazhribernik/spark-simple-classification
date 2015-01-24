import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf
import org.apache.spark.rdd.RDD

import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.classification.SVMWithSGD

object SimpleLogisticRegression {
  
  def readFile (sc: SparkContext, file: String): RDD[LabeledPoint] = {
  	val trainData = sc.textFile(file)
	val filteredData = trainData.filter(line => !line.startsWith("#"))
	val parsedData = filteredData.map { line =>
	  val parts = line.split(' ')
	  val label = if (parts.head.toInt == 1) 1 else 0
	  LabeledPoint(label, Vectors.sparse(
	  	10000,
	  	parts.tail.map{
	  	  element =>
	  	  val elementParts = element.split(':')
	  	  (elementParts(0).toInt, elementParts(1).toDouble)
	  	}
	  ))
	}
	parsedData
  } 

  def main(args: Array[String]) {
    // Load and parse the data
    val conf = new SparkConf().setAppName("Simple Logistic Regression")
    val sc = new SparkContext(conf)

	val parsedData = readFile(sc, args(0)).cache()

	println("Number of samples: " + parsedData.count )
	
	// Building the model
	val numIterations = 100
	val model = SVMWithSGD.train(parsedData, numIterations)

	// Evaluate model on training examples and compute training error
	val trainingSet = readFile(sc, args(1)).cache()
	val valuesAndPreds = trainingSet.map { point =>
	  val prediction = model.predict(point.features)
	  (point.label, prediction)
	}
    valuesAndPreds.cache()
    
    val numOfCorrect = valuesAndPreds.map{point =>
      if (point._1 == point._2) 1 else 0
    }.sum()
    val numOfAll = valuesAndPreds.count()
	println("All samples " + numOfAll)
	println("Correct predictions " + numOfCorrect)
	println("Accuracy " + numOfCorrect * 100.0 / numOfAll + "%")

  }
}