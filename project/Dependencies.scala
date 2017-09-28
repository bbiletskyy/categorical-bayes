import sbt._

object Version {
  val spark        = "2.2.0"
  val hadoop       = "2.7.4"
  val slf4j        = "1.7.21"
  val scalaTest    = "3.0.1"
  val scalaXml     = "1.0.6"
}

object Library {
  val sparkCore      = "org.apache.spark"       %% "spark-core"      % Version.spark exclude("org.scala-lang" , "scala-compiler") exclude("org.scala-lang", "scala-reflect") exclude("com.google.guava", "guava") exclude("org.scalatest", "scalatest_2.11")
  val sparkSql       = "org.apache.spark"       %% "spark-sql"       % Version.spark
  val sparkMlLib     = "org.apache.spark"       %% "spark-mllib"     % Version.spark
  val hadoopClient   = "org.apache.hadoop"      %  "hadoop-client"   % Version.hadoop
  val slf4jApi       = "org.slf4j"              %  "slf4j-api"       % Version.slf4j
  val scalaXml       = "org.scala-lang.modules" % "scala-xml_2.11"   % Version.scalaXml
  val scalaTest      = "org.scalatest"          %% "scalatest"       % Version.scalaTest
}

object Dependencies {
  import Library._

  val sparkHadoop = Seq(
    sparkCore,
    sparkMlLib,
    sparkSql,
    hadoopClient,
    scalaXml,
    scalaTest % "test"
  )
}
