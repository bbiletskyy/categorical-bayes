package org.apache.spark.ml.classification

import org.apache.hadoop.fs.Path
import org.apache.spark.ml.PredictorParams
import org.apache.spark.ml.linalg.{ BLAS, DenseMatrix, DenseVector, Matrix, SparseVector, Vector, Vectors }
import org.apache.spark.ml.param.{ DoubleParam, IntArrayParam, ParamMap, ParamValidators }
import org.apache.spark.ml.util._
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.sql.functions.col
import org.apache.spark.sql.{ DataFrame, Dataset }

/**
 * Params for Categorical Bayes Classifiers.
 */
private[classification] trait CategoricalBayesParams extends PredictorParams {

  /**
   * The smoothing parameter.
   * (default = 1.0).
   *
   * @group param
   */
  final val smoothing: DoubleParam = new DoubleParam(this, "smoothing", "The smoothing parameter.",
    ParamValidators.gtEq(0))

  /** @group getParam */
  final def getSmoothing: Double = $(smoothing)

  /**
   * Feature cardinalities parameter: i-th elememt of the array corresponds to
   * the number of distinct values that i-th feature can take.
   *
   * @group param
   */
  final val featureCardinalities: IntArrayParam = new IntArrayParam(this, "featureCardinalities", "Number of values per each feature",
    featureCardinalitiesValidator)

  /** @group getParam */
  final def getFeatureCardinalities: Array[Int] = $(featureCardinalities)

  private def featureCardinalitiesValidator: Array[Int] => Boolean = { (value: Array[Int]) =>
    value.size > 0 && value.forall(_ > 0)
  }
}

/**
 * Categorical version of Naive Bayes method.
 *
 * @param uid
 */
class CategoricalBayes(override val uid: String)
    extends ProbabilisticClassifier[Vector, CategoricalBayes, CategoricalBayesModel]
    with CategoricalBayesParams with DefaultParamsWritable {

  def this() = this(Identifiable.randomUID("cb"))

  def setSmoothing(value: Double): this.type = set(smoothing, value)

  setDefault(smoothing -> 1.0)

  def setFeatureCardinalities(value: Array[Int]): this.type = set(featureCardinalities, value)

  override protected def train(dataset: Dataset[_]): CategoricalBayesModel = {
    val labelFeatureIndexCounts: Array[((Double, Double, Int), Double)] = dataset.select(col($(labelCol)), col($(featuresCol))).rdd
      .map(row => (row.getDouble(0), row.getAs[Vector](1)))
      .flatMap { case (label, features) => features.toArray.zipWithIndex.map { case (feature, index) => ((label, feature, index), 1.0) } }
      .aggregateByKey(0.0)(
        seqOp = (localValue1, localValue2) => localValue1 + localValue2,
        combOp = (partitionVal1, partitionVal2) => partitionVal1 + partitionVal2
      ).collect()

    val labelCounts: Map[Double, Double] = labelFeatureIndexCounts.filter { case ((label, feature, index), count) => index == 0 }
      .map { case ((label, feature, index), count) => (label, count) }
      .groupBy { case (label, count) => label }
      .map { case (label, results) => (label, results.map(_._2).sum) }

    val totalTrainingExamples = labelCounts.map(_._2).sum
    val labelsNumber = labelCounts.size
    val lambda = getSmoothing
    val smoothedLogTotal = math.log(totalTrainingExamples + labelCounts.size * lambda)
    val smoothedLabelLogProbs: Map[Double, Double] = labelCounts.map {
      case (label, count) =>
        val smoothedLabelLogProb = math.log(count + lambda) - smoothedLogTotal
        (label, smoothedLabelLogProb)
    }

    val labelFeatureCounts: Array[Map[(Double, Double), Double]] =
      labelFeatureIndexCounts.groupBy { case ((label, feature, index), count) => index }.toArray
        .sortBy { case (index, grouped) => index }
        .map {
          case (index, grouped) =>
            grouped.map { case ((label, feature, index), count) => ((feature, label), count) }.toMap.withDefaultValue(0.0)
        }
    val pi = Vectors.dense(smoothedLabelLogProbs.toArray.sortBy { case (label, logProb) => label }.map(_._2))

    val thetas: Array[Matrix] = getFeatureCardinalities.map { featureCardinality =>
      val matrixElements = Array.fill(featureCardinality * labelsNumber)(lambda)
      new DenseMatrix(featureCardinality, labelsNumber, matrixElements, true)
    }.zip(labelFeatureCounts).map {
      case (matrix, labelFeatureCount) =>
        for {
          feature <- 0 to matrix.numRows - 1;
          label <- 0 to matrix.numCols - 1
        } {
          val smoothedCondProbNumerator = matrix(feature, label) + labelFeatureCount((feature, label))
          val featureCardinality = matrix.numRows
          val smoothedCondProbDenominator = lambda * featureCardinality + labelCounts(label)
          val smoothedCondProb = math.log(smoothedCondProbNumerator) - math.log(smoothedCondProbDenominator)
          matrix.update(feature, label, smoothedCondProb)
        }
        matrix
    }

    new CategoricalBayesModel(uid, pi, thetas)
  }

  override def copy(extra: ParamMap): CategoricalBayes = defaultCopy(extra)
}

class CategoricalBayesModel private[ml] (
  override val uid: String,
  val pi: Vector,
  val thetas: Array[Matrix])
    extends ProbabilisticClassificationModel[Vector, CategoricalBayesModel]
    with CategoricalBayesParams with MLWritable {

  override val numFeatures: Int = thetas.size

  override val numClasses: Int = pi.size

  override protected def predictRaw(features: Vector): Vector = {
    assert(features.size == thetas.size)
    val rawPredictionsVec = features.toArray.zip(thetas)
      .map { case (feature, theta) => theta.rowIter.toList(feature.toInt) }
      .foldLeft(pi.copy) { (acc, labelCondLogProbs) =>
        BLAS.axpy(1.0, labelCondLogProbs, acc)
        acc
      }
    rawPredictionsVec
  }

  override protected def raw2probabilityInPlace(rawPrediction: Vector): Vector = {
    rawPrediction match {
      case dv: DenseVector =>
        var i = 0
        val size = dv.size
        val maxLog = dv.values.max
        while (i < size) {
          dv.values(i) = math.exp(dv.values(i) - maxLog)
          i += 1
        }
        val probSum = dv.values.sum
        i = 0
        while (i < size) {
          dv.values(i) = dv.values(i) / probSum
          i += 1
        }
        dv
      case sv: SparseVector =>
        throw new RuntimeException("Unexpected error in CategoricalBayesModel:" +
          " raw2probabilityInPlace encountered SparseVector")
    }
  }

  override def copy(extra: ParamMap): CategoricalBayesModel = {
    copyValues(new CategoricalBayesModel(uid, pi, thetas).setParent(this.parent), extra)
  }

  override def toString: String = {
    s"CategoricalBayesModel (uid=$uid) with ${pi.size} classes"
  }

  override def write: MLWriter = new CategoricalBayesModel.CategoricalBayesModelWriter(this)
}

object CategoricalBayesModel extends MLReadable[CategoricalBayesModel] {

  override def read: MLReader[CategoricalBayesModel] = new CategoricalBayesModelReader

  override def load(path: String): CategoricalBayesModel = super.load(path)

  private[CategoricalBayesModel] class CategoricalBayesModelWriter(instance: CategoricalBayesModel) extends MLWriter {

    private case class Data(m: Matrix)

    override protected def saveImpl(path: String): Unit = {
      DefaultParamsWriter.saveMetadata(instance, path, sc)
      val pi = Data(new DenseMatrix(1, instance.pi.toArray.length, instance.pi.toArray, isTransposed = true))
      val dataPath = new Path(path, "data").toString

      val data: Seq[Data] = Seq(pi) ++ instance.thetas.map(m => Data(m))
      sparkSession.createDataFrame(data).repartition(1).write.parquet(dataPath)
    }
  }

  private class CategoricalBayesModelReader extends MLReader[CategoricalBayesModel] {

    /** Checked against metadata when loading model */
    private val className = classOf[CategoricalBayesModel].getName

    override def load(path: String): CategoricalBayesModel = {
      val metadata = DefaultParamsReader.loadMetadata(path, sc, className)

      val dataPath = new Path(path, "data").toString
      val data: DataFrame = sparkSession.read.parquet(dataPath)
      val dataML = MLUtils.convertMatrixColumnsToML(data, "m")

      val head :: tail = dataML.select("m").collect().map(r => r.getAs[Matrix]("m")).toList
      val pi: Vector = head.rowIter.next()
      val thetas: Array[Matrix] = tail.toArray

      val model = new CategoricalBayesModel(metadata.uid, pi, thetas)

      DefaultParamsReader.getAndSetParams(model, metadata)
      model
    }
  }

}
