package org.apache.spark.ml.classification

import org.apache.spark.ml.linalg.{ Vector, Vectors }
import org.apache.spark.ml.{ Pipeline, PipelineModel }
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.col
import org.scalatest.Matchers._
import org.scalatest._

class CategoricalBayesSpec extends FlatSpec {
  val spark: SparkSession = SparkSession
    .builder()
    .appName("CategoricalBayesTests")
    .config("spark.master", "local[4]")
    .getOrCreate()

  "CategoricalBayes model with a single boolean feature" should "predict class 1 among classes {0, 1} with  {0.33, 0.66} probabilities" in {
    val training = spark.createDataFrame(Seq(
      (0.0, Vectors.dense(0.0)),
      (0.0, Vectors.dense(1.0)),
      (0.0, Vectors.dense(1.0)),
      (1.0, Vectors.dense(0.0)),
      (1.0, Vectors.dense(0.0)),
      (1.0, Vectors.dense(1.0))
    )).toDF("label", "features")

    val categoricalBayes = new CategoricalBayes()
      .setLabelCol("label")
      .setFeaturesCol("features")
      .setFeatureCardinalities(Array(2))
      .setSmoothing(0.0)
      .setPredictionCol("MB-predicted")
      .setProbabilityCol("MB-probability")
      .setRawPredictionCol("MB-rawPrediction")

    val pipeline = new Pipeline().setStages(Array(
      categoricalBayes
    ))

    val model: PipelineModel = pipeline.fit(training)
    val test = spark.createDataFrame(Seq(
      (1L, Vectors.dense(0.0))
    )).toDF("label", "features")
    val predicted = model.transform(test)
    predicted.show(truncate = false)

    val predictedClass = predicted.select(col(categoricalBayes.getPredictionCol)).head().getDouble(0)
    predictedClass should equal(1.0)

    val rawPredictions = predicted.select(col(categoricalBayes.getRawPredictionCol)).head().getAs[Vector](0)
    rawPredictions(0) should equal(-1.79 +- 0.1)
    rawPredictions(1) should equal(-1.09 +- 0.1)

    val classProbabilities = predicted.select(col(categoricalBayes.getProbabilityCol)).head().getAs[Vector](0)
    classProbabilities(0) should equal(0.33 +- 0.1)
    classProbabilities(1) should equal(0.66 +- 0.1)
  }

  "CategoricalBayes model with two boolean features" should "predict class 1 among classes {0, 1} with  {0.22, 0.88} probabilities" in {
    val training = spark.createDataFrame(Seq(
      (0.0, Vectors.dense(0.0, 1.0)),
      (0.0, Vectors.dense(1.0, 0.0)),
      (0.0, Vectors.dense(1.0, 0.0)),
      (1.0, Vectors.dense(0.0, 1.0)),
      (1.0, Vectors.dense(0.0, 1.0)),
      (1.0, Vectors.dense(1.0, 0.0))
    )).toDF("label", "features")

    val categoricalBayes = new CategoricalBayes()
      .setLabelCol("label")
      .setFeaturesCol("features")
      .setFeatureCardinalities(Array(2, 2))
      .setSmoothing(0.0)
      .setPredictionCol("MB-predicted")
      .setProbabilityCol("MB-probability")
      .setRawPredictionCol("MB-rawPrediction")

    val pipeline = new Pipeline().setStages(Array(
      categoricalBayes
    ))

    val model: PipelineModel = pipeline.fit(training)
    val test = spark.createDataFrame(Seq(
      (1L, Vectors.dense(0.0, 1.0))
    )).toDF("label", "features")
    val predicted = model.transform(test)
    predicted.show(truncate = false)

    val predictedClass = predicted.select(col(categoricalBayes.getPredictionCol)).head().getDouble(0)
    predictedClass should equal(1.0)

    val rawPredictions = predicted.select(col(categoricalBayes.getRawPredictionCol)).head().getAs[Vector](0)
    rawPredictions(0) should equal(-2.89 +- 0.1)
    rawPredictions(1) should equal(-1.50 +- 0.1)

    val classProbabilities = predicted.select(col(categoricalBayes.getProbabilityCol)).head().getAs[Vector](0)
    classProbabilities(0) should equal(0.22 +- 0.1)
    classProbabilities(1) should equal(0.88 +- 0.1)
  }

  "CategoricalBayes model with default (1.0) smoothing with a single boolean feature and missing training example" should "predict class 1 among classes {0, 1} with  {0.33, 0.66} probabilities" in {
    val training = spark.createDataFrame(Seq(
      //(0.0, Vectors.dense(0.0)),
      (0.0, Vectors.dense(1.0)),
      (0.0, Vectors.dense(1.0)),
      (1.0, Vectors.dense(0.0)),
      (1.0, Vectors.dense(0.0)),
      (1.0, Vectors.dense(1.0))
    )).toDF("label", "features")

    val categoricalBayes = new CategoricalBayes()
      .setLabelCol("label")
      .setFeaturesCol("features")
      .setFeatureCardinalities(Array(2))
      .setSmoothing(0.001)
      .setPredictionCol("MB-predicted")
      .setProbabilityCol("MB-probability")
      .setRawPredictionCol("MB-rawPrediction")

    val pipeline = new Pipeline().setStages(Array(
      categoricalBayes
    ))

    val model: PipelineModel = pipeline.fit(training)
    val test = spark.createDataFrame(Seq(
      (1L, Vectors.dense(0.0))
    )).toDF("label", "features")
    val predicted = model.transform(test)
    predicted.show(truncate = false)

    val predictedClass = predicted.select(col(categoricalBayes.getPredictionCol)).head().getDouble(0)
    predictedClass should equal(1.0)

    val rawPredictions = predicted.select(col(categoricalBayes.getRawPredictionCol)).head().getAs[Vector](0)
    rawPredictions(0) should equal(-8.51 +- 0.1)
    rawPredictions(1) should equal(-0.91 +- 0.1)

    val classProbabilities = predicted.select(col(categoricalBayes.getProbabilityCol)).head().getAs[Vector](0)
    classProbabilities(0) should equal(0.0005 +- 0.0001)
    classProbabilities(1) should equal(0.9995 +- 0.0001)
  }

  "CategoricalBayes" should "save and load the trained model" in {
    val training = spark.createDataFrame(Seq(
      (0.0, Vectors.dense(0.0, 1.0)),
      (0.0, Vectors.dense(1.0, 0.0)),
      (0.0, Vectors.dense(1.0, 0.0)),
      (1.0, Vectors.dense(0.0, 1.0)),
      (1.0, Vectors.dense(0.0, 1.0)),
      (1.0, Vectors.dense(1.0, 0.0))
    )).toDF("label", "features")

    val categoricalBayes = new CategoricalBayes()
      .setLabelCol("label")
      .setFeaturesCol("features")
      .setPredictionCol("MB-predicted")
      .setProbabilityCol("MB-probability")
      .setRawPredictionCol("MB-rawPrediction")
      .setFeatureCardinalities(Array(2, 2))
      .setSmoothing(0.0)

    val pipeline = new Pipeline().setStages(Array(
      categoricalBayes
    ))

    val model: PipelineModel = pipeline.fit(training)
    val modelPath = "bayes.model"
    model.write.overwrite.save(modelPath)

    val loadedModel = PipelineModel.load(modelPath)

    val test = spark.createDataFrame(Seq((1L, Vectors.dense(0.0, 1.0)))).toDF("label", "features")
    val predicted = loadedModel.transform(test)
    predicted.show(truncate = false)

    val predictedClass = predicted.select(col(categoricalBayes.getPredictionCol)).head().getDouble(0)
    predictedClass should equal(1.0)

    val rawPredictions = predicted.select(col(categoricalBayes.getRawPredictionCol)).head().getAs[Vector](0)
    rawPredictions(0) should equal(-2.89 +- 0.1)
    rawPredictions(1) should equal(-1.50 +- 0.1)

    val classProbabilities = predicted.select(col(categoricalBayes.getProbabilityCol)).head().getAs[Vector](0)
    classProbabilities(0) should equal(0.22 +- 0.1)
    classProbabilities(1) should equal(0.88 +- 0.1)

    import org.apache.hadoop.fs.FileSystem
    import org.apache.hadoop.fs.Path
    val fs = FileSystem.get(spark.sparkContext.hadoopConfiguration)

    if (fs.exists(new Path(modelPath)))
      fs.delete(new Path(modelPath), true)
  }
}

