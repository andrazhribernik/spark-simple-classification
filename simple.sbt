name := "Simple Logistic Regression"

version := "1.0"

scalaVersion := "2.10.4"

libraryDependencies ++= Seq(
  "org.apache.spark"  %% "spark-core"              % "1.2.0",
  "org.apache.spark"  %% "spark-mllib"             % "1.2.0"
)