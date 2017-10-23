package example

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.{CategoricalBayes}
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler}
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.udf

object AcuteInflammations {
  def main(args: Array[String]): Unit = {
    val spark: SparkSession = SparkSession
      .builder()
      .appName("AcuteInflammations")
      .config("spark.master", "local[4]")
      .getOrCreate()

    import spark.implicits._

    val rawData = spark.sparkContext.textFile("data/diagnosis.data")
        .map(line => line.trim)
        .map(line => line.split("\t"))
        .filter(cols => cols.size == 8)
        .map(cols => (cols(0), cols(1), cols(2), cols(3), cols(4), cols(5), cols(6), cols(7)))
      .toDF(
        "temp", // Temperature of patient
        "nausea", //"Occurrence of nausea
        "lumbar", //Lumbar pain
        "urine", // Urine pushing
        "micturition", //Micturition pains
        "urethra", //Burning of urethra
        "bladder", //decision: Inflammation of urinary bladder
        "renal" //decision: Nephritis of renal pelvis origin
      )

    val discretizeTempUdf = udf { temp: String => temp.replaceAll(",", ".").toDouble.round.toString }
    val data = rawData.withColumn("dtemp", discretizeTempUdf('temp)).drop("temp").drop("renal")
    data.show()

    val tempIndexer = new StringIndexer().setInputCol("dtemp").setOutputCol("idtemp")
    val nauseaIndexer = new StringIndexer().setInputCol("nausea").setOutputCol("inausea")
    val lumbarIndexer = new StringIndexer().setInputCol("lumbar").setOutputCol("ilumbar")
    val urineIndexer = new StringIndexer().setInputCol("urine").setOutputCol("iurine")
    val micturitionIndexer = new StringIndexer().setInputCol("micturition").setOutputCol("imicturition")
    val urethraIndexer = new StringIndexer().setInputCol("urethra").setOutputCol("iurethra")
    val vectorAssembler = new VectorAssembler()
      .setInputCols(Array("idtemp", "inausea", "ilumbar", "iurine", "imicturition", "iurethra"))
      .setOutputCol("features")
    val bladderIndexer = new StringIndexer().setInputCol("bladder").setOutputCol("label")
    val categoricalBayes = new CategoricalBayes().setLabelCol("label")
      .setFeaturesCol("features")
      .setFeatureCardinalities(Array(7, 2, 2, 2, 2, 2))
    val pipeline = new Pipeline().setStages(Array(
      tempIndexer,
      nauseaIndexer,
      lumbarIndexer,
      urineIndexer,
      micturitionIndexer,
      urethraIndexer,
      vectorAssembler,
      bladderIndexer,
      categoricalBayes))

    val evaluator = new BinaryClassificationEvaluator()
    val crossValidator = new CrossValidator()
      .setEstimator(pipeline)
      .setEvaluator(evaluator)
      .setEstimatorParamMaps(new ParamGridBuilder().build())
      .setNumFolds(5)
    val accuracy = crossValidator.fit(data).avgMetrics.head
    println(s"Cross-validated accuracy: $accuracy")

    val Array(trainData, testData) = data.randomSplit(Array(0.8, 0.2))
    trainData.cache()
    testData.cache()

    val model = pipeline.fit(trainData)
    val predictedData = model.transform(testData)
    predictedData.show()
    val predictionAccuracy = evaluator.evaluate(predictedData)
    println(s"Prediction accuracy: $predictionAccuracy")
  }
}
