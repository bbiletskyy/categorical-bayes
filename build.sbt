name := """categorical-bayes"""

scalaVersion := "2.11.8"

libraryDependencies ++= Dependencies.sparkHadoop

releaseSettings

scalariformSettings

fork in run := true
